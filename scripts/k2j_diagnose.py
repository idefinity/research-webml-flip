"""K2j - separate memorisation from compounding in run-1's student.

Run-1's numbers: train loss 0.213 -> 0.088, held-out endpoint error unchanged
or worse. Two explanations, and they want opposite fixes:

  A. memorisation - the velocity field was learned on 30 images and nowhere
     else. Fix: more images. More epochs will make it worse.
  B. compounding - each segment is nearly right, but the student's own rollout
     walks into states the teacher never visited, so the errors add up across
     four steps. Fix: targets taken from the student's own rollout (true
     consistency distillation). More data barely helps.

One measurement separates them. Take states from the TEACHER's trajectory, so
no compounding can hide inside them, and measure the student's error on a
single segment. Compare it with the error of its full four-step rollout.

    per-segment error small, rollout bad   -> B
    per-segment error large                -> A

Run-1 never measured the first one, so the verdict it recorded is not actually
a verdict. It also never evaluated on a training image, which is why
"memorisation" is still a guess rather than a finding.

The same pass also produces the teacher-4-step-Euler baseline on the frozen
grid, for free, from the teacher's own velocities at the grid states - so the
student's rollout is judged against the thing it is actually trying to beat
rather than against a number remembered from another run.

Branch pairing is not assumed. The sampler makes two calls per step (cond and
uncond) and the trace does not say which is which; the combined velocity is
`g*a - (g-1)*b` for one ordering and `g*b - (g-1)*a` for the other. Both are
tried and the one whose single Euler step lands closer to the teacher's actual
next state wins. That self-check is what makes the teacher-instantaneous
column trustworthy instead of a coin flip.

Env:
  FLIP_STUDENT_URL  signed output URL of run-1's k2e/student.pt
  FLIP_GRID_URL     signed output URL of run-1's k2e/grid.json
  FLIP_N            how many upstream example images (default 4)
  FLIP_RUN_K2G      1 = fetch and run k2g afterwards for the PLYs (default 1)
"""

import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch
from PIL import Image

warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k2j"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_IMG = int(os.environ.get("FLIP_N", "4"))
STUDENT_URL = os.environ.get("FLIP_STUDENT_URL", "").strip()
GRID_URL = os.environ.get("FLIP_GRID_URL", "").strip()
RUN_K2G = os.environ.get("FLIP_RUN_K2G", "1") == "1"
SEED = int(os.environ.get("FLIP_SEED", "42"))
N_GAUSS = 65536
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
# dt is 0.001 x the recorded t difference: measured in run 3 of the smoke
# tests, not assumed. The recorded t values sit in the hundreds, so the
# sampler works in sigma units times 1000.
DT_SCALE = float(os.environ.get("FLIP_T_SCALE", "0.001"))
GUIDANCE = 3.0

R = {"notes": [], "images": [], "verdict": {}}


def note(msg=""):
    print(msg, flush=True)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=2))


def fetch_url(url, dst, min_bytes=1):
    """Atomic download, skipped when a complete file is already there."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size >= min_bytes:
        note(f"{dst.name} already on disk ({dst.stat().st_size/1e6:.1f} MB)")
        return True
    import urllib.request  # noqa: PLC0415
    note(f"fetching {url[:64]}...")
    t0 = time.time()
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=1800) as r:
            with open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    fh.write(chunk)
        os.replace(tmp, dst)
    except Exception:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        note(f"fetch FAILED: {traceback.format_exc()[-500:]}")
        return False
    note(f"  {dst.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s")
    return True


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def build_pipeline():
    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415

    files = sorted(CKPT.rglob("*.safetensors"))

    def pick(*keys):
        for f in files:
            n = f.name.lower()
            if all(k in n for k in keys):
                return f
        return None

    dino_p = pick("dino")
    vae_p = pick("flux")
    dec_p = pick("decoder") or pick("octree")
    rmbg_p = pick("biref") or pick("rmbg")
    taken = {p for p in (dino_p, vae_p, rmbg_p, dec_p) if p}
    ck = [f for f in files if f not in taken][0]
    missing = [k for k, v in (("dino", dino_p), ("flux", vae_p),
                              ("decoder", dec_p), ("rmbg", rmbg_p)) if v is None]
    if missing:
        raise RuntimeError(f"missing ckpt parts {missing}")
    return T.TripoSplatPipeline(
        ckpt_path=str(ck), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))


def as_map(state):
    if torch.is_tensor(state):
        return {"latent": state}
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


def cond_key(cond):
    """Cheap identity for a cond dict, so 40 calls do not store 40 copies."""
    parts = []
    for k, v in sorted(cond.items()):
        if torch.is_tensor(v):
            parts.append(f"{k}:{tuple(v.shape)}:{float(v.float().sum()):.4f}")
    return "|".join(parts)


class Recorder:
    """Record every flow-model call during one real sampler run.

    States and outputs are cloned (they are small); cond dicts are deduped by
    content because CFG makes two calls per step with two conds, and cloning
    the real one forty times is 400 MB for nothing.
    """

    def __init__(self, flow):
        self.flow = flow
        self.cls = type(flow)
        self.orig = self.cls.forward
        self.calls = []
        self.conds = {}

    def __enter__(self):
        rec = self

        def spy(_self, state, t, condf, *a, **k):
            st = as_map(state)
            call = {
                "t": float(torch.as_tensor(t).flatten()[0].float()),
                "state": {kk: v.detach().clone() for kk, v in st.items()},
                "out": None,
            }
            key = cond_key(as_map(condf) if not torch.is_tensor(condf) else
                           {"x": condf})
            if key not in rec.conds:
                rec.conds[key] = {
                    kk: v.detach().clone() for kk, v in as_map(condf).items()}
            call["cond_key"] = key
            # orig is the plain class function, not a bound method: the
            # instance has to go back in explicitly, or state lands in self's
            # slot and cond goes missing. That is the TypeError this script's
            # first run died on, in all four images at once.
            res = rec.orig(_self, state, t, condf, *a, **k)
            call["out"] = {kk: v.detach().clone()
                           for kk, v in as_map(res).items()}
            rec.calls.append(call)
            return res

        self.cls.forward = spy
        return self

    def __exit__(self, *exc):
        self.cls.forward = self.orig
        return False


def trajectory(calls):
    """Collapse calls into distinct states, exactly as k2e does on traces."""
    states, ts, seen = [], [], None
    for c in calls:
        key = (c["t"], round(sum(float(v.float().sum())
                                 for v in c["state"].values()), 3))
        if key == seen:
            continue
        seen = key
        states.append(c["state"])
        ts.append(c["t"])
    return states, ts


def pick_grid(ts, s_steps):
    """Nearest recorded state to each equal-sigma boundary, as k2e does."""
    start = ts[0]
    idx, last = [], -1
    for i in range(s_steps):
        want = start - start * i / s_steps
        best = min(range(len(ts)), key=lambda j: abs(ts[j] - want))
        if best <= last:
            best = min(last + 1, len(ts) - 1)
        idx.append(best)
        last = best
    return idx + [len(ts)]


def rel(a, b):
    return round(((a - b).norm() / b.norm()).item(), 4)


def cos(a, b):
    return round(torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0).item(), 4)


def forward_velocity(model, state, t, cond):
    """Call the flow module the way the sampler does. t is passed through
    untouched - the recorded value is already in the module's convention."""
    t_in = torch.as_tensor([t], device=DEV, dtype=torch.float32)
    sm = {k: v.to(DEV).float() for k, v in state.items()}
    cd = {k: v.to(DEV).float() for k, v in cond.items()}
    with torch.autocast("cuda", torch.float16, cache_enabled=False):
        out = model(sm, t_in, cd)
    return as_map(out)


def pair_calls(calls, t, k_index, all_states, dts, guidance):
    """Return the CFG-combined velocity at boundary t, branch order verified.

    Two calls share each t. `guidance*a - (guidance-1)*b` is right for one
    ordering; the other ordering is `guidance*b - (guidance-1)*a`. The one
    whose single Euler step reproduces the teacher's actual next state is the
    right one, and that check is cheaper than trusting the call order.
    """
    pair = [c for c in calls if abs(c["t"] - t) < 1e-3]
    if len(pair) != 2:
        return None, "notfound", 0.0
    ref = (all_states[k_index + 1] if k_index + 1 < len(all_states) else None)
    best, best_err, which = None, None, None
    for name, a, b in (("ab", pair[0]["out"], pair[1]["out"]),
                       ("ba", pair[1]["out"], pair[0]["out"])):
        v = {ch: guidance * a[ch].float() - (guidance - 1) * b[ch].float()
             for ch in a if ch in b}
        if ref is None:
            return v, "nocheck", 0.0
        x = {ch: all_states[k_index][ch].float() + v[ch] * dts[k_index]
             for ch in v}
        err = sum(rel(x[ch].cpu(), ref[ch].float().cpu())
                  for ch in v if ch in ref) / max(len(v), 1)
        note(f"      branch {name}: one-step rel err {err:.4f}")
        if best_err is None or err < best_err:
            best, best_err, which = v, err, name
    return best, which, best_err


def analyse(pipe, flow, student, image_path, grid):
    note("-" * 72)
    note(Path(image_path).name)
    prepared = pipe.preprocess_image(Image.open(image_path).convert("RGB"))
    cond_full = pipe.encode_image(prepared)

    with Recorder(flow) as rec:
        gen = torch.Generator(device=DEV).manual_seed(SEED)
        lat = pipe.sample_latent(cond_full, steps=20, guidance_scale=GUIDANCE,
                                 shift=3.0, generator=gen)
    final = as_map(lat)
    states, ts = trajectory(rec.calls)
    note(f"  teacher run: {len(rec.calls)} calls -> {len(states)} states")

    n = len(states)
    dts = [(ts[i + 1] - ts[i]) * DT_SCALE for i in range(n - 1)]
    dts.append((0.0 - ts[-1]) * DT_SCALE)
    all_states = states + [final]

    g = pick_grid(ts, len(grid["t"]))
    got_t = [round(ts[i], 2) for i in g[:-1]]
    ok = all(abs(a - b) < 1.0 for a, b in zip(got_t, grid["t"]))
    note(f"  grid indices {g}  t {got_t}  "
         f"{'matches grid.json' if ok else 'MISMATCH vs ' + str(grid['t'])}")
    if not ok:
        return {"image": Path(image_path).name, "grid_ok": False}

    cond_real = None
    for c in rec.calls[:4]:
        if abs(c["t"] - ts[0]) < 1e-3:
            cond_real = rec.conds[c["cond_key"]]
            break
    if cond_real is None:
        cond_real = next(iter(rec.conds.values()))

    segs = []
    for k in range(len(g) - 1):
        i, j = g[k], g[k + 1]
        span = sum(dts[i:j])
        v_star = {ch: (all_states[j][ch].float() - all_states[i][ch].float())
                  / span for ch in all_states[i] if ch in all_states[j]}
        t_k = ts[i]
        v_stu = forward_velocity(student, all_states[i], t_k, cond_real)
        v_tea, branch, branch_err = pair_calls(rec.calls, t_k, i, all_states,
                                               dts, GUIDANCE)
        row = {"k": k, "t": round(t_k, 2), "span": round(span, 4),
               "branch": branch, "branch_one_step_err": round(branch_err, 4)}
        for ch in v_star:
            if ch not in v_stu:
                continue
            a = v_stu[ch].float().cpu()
            b = v_star[ch].cpu()
            r = {"student_vs_segment": {"rel": rel(a, b), "cos": cos(a, b)}}
            if ch in v_tea:
                c2 = v_tea[ch].float().cpu()
                r["teacher_inst_vs_segment"] = {"rel": rel(c2, b),
                                                "cos": cos(c2, b)}
            row[ch] = r
        segs.append(row)
        line = f"  seg{k} t={row['t']:<7} span={row['span']}"
        for ch in ("latent", "camera"):
            if ch in row:
                m = row[ch]
                line += f"  {ch}: stu {m['student_vs_segment']['rel']:.3f}"
                if "teacher_inst_vs_segment" in m:
                    line += f"/tea {m['teacher_inst_vs_segment']['rel']:.3f}"
        note(line)

    def rollout(velocities):
        x = {ch: all_states[g[0]][ch].float().clone().to(DEV)
             for ch in all_states[g[0]]}
        for k in range(len(g) - 1):
            v = velocities[k]
            for ch in x:
                if ch in v:
                    x[ch] = x[ch] + v[ch].float() * sum(dts[g[k]:g[k + 1]])
        return x

    stu_roll = rollout([forward_velocity(student, all_states[g[k]], ts[g[k]],
                                         cond_real)
                        for k in range(len(g) - 1)])
    stu_vs = {ch: {"rel": rel(stu_roll[ch].cpu(), final[ch].float().cpu()),
                   "cos": cos(stu_roll[ch], final[ch])} for ch in stu_roll
              if ch in final}

    tea_roll = rollout([pair_calls(rec.calls, ts[g[k]], g[k], all_states,
                                   dts, GUIDANCE)[0]
                        for k in range(len(g) - 1)])
    tea_vs = {ch: {"rel": rel(tea_roll[ch].cpu(), final[ch].float().cpu()),
                   "cos": cos(tea_roll[ch], final[ch])} for ch in tea_roll
              if ch in final}

    note(f"  rollout to teacher endpoint (rel-l2):")
    note(f"    teacher 4-step euler (baseline)  " +
         "  ".join(f"{ch} {m['rel']:.3f}" for ch, m in sorted(tea_vs.items())))
    note(f"    student 4 steps                  " +
         "  ".join(f"{ch} {m['rel']:.3f}" for ch, m in sorted(stu_vs.items())))

    return {"image": Path(image_path).name, "grid_ok": True, "segments": segs,
            "rollout": {"student": stu_vs, "teacher_4step": tea_vs}}


def pick_images():
    ex = REPO / "static" / "example_inputs"
    cands = sorted(q for q in ex.glob("*") if q.suffix.lower() in IMG_EXT)
    if not cands:
        cands = sorted(q for q in (REPO / "static").rglob("*")
                       if q.suffix.lower() in IMG_EXT)
    return cands[:N_IMG]


def main():
    note("=" * 72)
    note("K2j memorisation vs compounding")
    note("=" * 72)
    grid_path = WORK / "k2j/grid.json"
    if GRID_URL:
        fetch_url(GRID_URL, grid_path, 10)
    grid = (json.loads(grid_path.read_text()) if grid_path.exists()
            else {"t": [1000.0, 750.0, 500.0, 250.0], "dt": [-0.25] * 4})
    note(f"grid {grid['t']}  dt {grid['dt']}")

    student_path = WORK / "k2j/student.pt"
    if STUDENT_URL:
        fetch_url(STUDENT_URL, student_path, int(1e6))

    pipe = build_pipeline()
    flow = pipe.flow_model
    student = None
    if student_path.exists():
        blob = torch.load(str(student_path), map_location="cpu")
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        student = flow
        student.load_state_dict(sd)
        student = student.to(DEV).float()
        note(f"student loaded ({len(sd)} tensors, fp32)")
    else:
        note("NO STUDENT - only the teacher columns will mean anything")

    for img in pick_images():
        try:
            R["images"].append(analyse(pipe, flow, student, img, grid))
        except Exception:  # noqa: BLE001
            note("  image FAILED")
            note(traceback.format_exc()[-1200:])
            R["images"].append({"image": Path(img).name, "error":
                                traceback.format_exc()[-800:]})
        save()
        torch.cuda.empty_cache()

    segs = [s for r in R["images"] if r.get("grid_ok")
            for s in r.get("segments", [])]
    if segs and student is not None:
        for ch in ("latent", "camera"):
            vals = [s[ch]["student_vs_segment"]["rel"] for s in segs if ch in s]
            if vals:
                R["verdict"][f"per_segment_student_rel_{ch}"] = {
                    "median": sorted(vals)[len(vals) // 2],
                    "min": min(vals), "max": max(vals), "n": len(vals)}
        imgs = [r for r in R["images"] if r.get("grid_ok")]
        for ch in ("latent", "camera"):
            roll = [r["rollout"]["student"][ch]["rel"] for r in imgs
                    if ch in r.get("rollout", {}).get("student", {})]
            if roll:
                R["verdict"][f"rollout_student_rel_{ch}"] = {
                    "median": sorted(roll)[len(roll) // 2],
                    "min": min(roll), "max": max(roll), "n": len(roll)}
        ps = R["verdict"].get("per_segment_student_rel_latent", {})
        ro = R["verdict"].get("rollout_student_rel_latent", {})
        if ps and ro:
            note("=" * 72)
            note(f"per-segment median rel {ps['median']}   "
                 f"rollout median rel {ro['median']}")
            if ps["median"] < 0.15 and ro["median"] > 0.5:
                R["verdict"]["call"] = ("B: compounding - single segments are "
                                        "fine, the rollout is what drifts")
            elif ps["median"] > 0.4:
                R["verdict"]["call"] = ("A: memorisation - the field itself is "
                                        "wrong off the training images")
            else:
                R["verdict"]["call"] = ("mixed - read the table, "
                                        "do not trust a rule")
            note(R["verdict"]["call"])
    save()


if __name__ == "__main__":
    try:
        main()
        if RUN_K2G:
            note("=" * 72)
            note("handing over to k2g for the PLYs")
            import base64  # noqa: PLC0415
            import urllib.request  # noqa: PLC0415
            sha = os.environ.get("FLIP_SHA", "")
            src = None
            for u in (f"https://raw.githubusercontent.com/idefinity/"
                      f"research-webml-flip/{sha}/scripts/k2g_student_ply.py",
                      f"https://api.github.com/repos/idefinity/"
                      f"research-webml-flip/contents/scripts/k2g_student_ply.py"
                      f"?ref={sha}"):
                try:
                    req = urllib.request.Request(
                        u, headers={"User-Agent": "flip-k2j",
                                    "Accept": "application/vnd.github.raw"})
                    with urllib.request.urlopen(req, timeout=60) as fh:
                        raw = fh.read()
                    if "api.github.com" in u:
                        raw = base64.b64decode(json.loads(raw)["content"])
                    src = raw.decode("utf-8")
                    break
                except Exception as exc:  # noqa: BLE001
                    note(f"  {type(exc).__name__} on {u[:60]}")
            if src is None:
                note("could not fetch k2g - PLYs skipped")
            else:
                os.environ["FLIP_STUDENT"] = str(
                    WORK / "k2j/student.pt")
                os.environ["FLIP_GRID_URL"] = GRID_URL
                (WORK / "k2g_script.py").write_text(src)
                rc = subprocess.run([sys.executable, str(WORK / "k2g_script.py")])
                note(f"k2g rc={rc.returncode}")
    except Exception:  # noqa: BLE001
        note(traceback.format_exc())
        save()
        raise
