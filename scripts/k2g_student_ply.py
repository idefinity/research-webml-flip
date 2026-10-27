"""K2g - put the student's weights back into the pipeline and dump PLYs.

Everything before this judged the student by numbers in latent space: cosine,
rel-L2, MSE. Those are the numbers that were easy to compute, not the numbers
that decide anything. What decides is whether the Gaussians look right when
they are rasterised, which is how K2b settled the 4-step question and how K2c
found the pose drift.

So: load the pipeline, swap the flow model for the student, run it, decode,
save_ply, and put the PLYs next to each other -

    teacher 20 steps / guidance 3     what we are trying to stay near
    teacher  8 steps / guidance 3     where the error curve sits between
    teacher  4 steps / guidance 3     the naive shortcut, the K2b baseline
    teacher  2 steps / guidance 3     what truncating the teacher's own field costs
    student  4 steps / no guidance    what distillation bought

The teacher columns are controls, not ceilings. An earlier version of this note
said the best a 2-step student can do is whatever the teacher does at 2 steps,
and that is wrong: the student is a different field, trained to land on the
teacher's 20-step endpoint, so it can beat the teacher at the same call count -
and it does, on screen, at 4 steps.

Two reasons the same-count comparison is not a bound. The student's per-call
velocity is a chord (segment start to segment end), which by Taylor is the
midpoint velocity plus O(dt^2); truncating the teacher uses the tangent at the
segment start, which is O(dt). On a trajectory that bends, the gap is large -
k2j measured it at 0.63-0.75 for the first segment. And the teacher columns run
with guidance 3, so at large dt the 3a-2b combination pushes the state off the
manifold on top of the integration error.

What is a real bound: the teacher's 20-step endpoint (the target - the student
cannot fix errors that are already in it), and the error of fitting that whole
trajectory with n straight pieces, which grows like dt^2 * curvature. At 2 steps
dt doubles and the first piece covers the most curved part of the path, so the
floor rises about 4x. That makes 2 steps expensive, not impossible, and the
teacher@2 column does not measure it because guidance is still on.

The runtime schedule matters and is worth stating precisely. The sampler is

    t_seq = shift * linspace(1, 0, steps+1) / (1 + (shift-1) * linspace(1, 0, steps+1))
    dt    = t - t_prev
    x     = x - v * dt

so with shift=3 and steps=20 the recorded t values are 3u/(1+2u) for u on a
21-point grid from 1 to 0 - which is what the trace showed. The student was
trained on boundaries at sigma = 1.0, 0.75, 0.5, 0.25, 0, which is exactly
linspace(1, 0, 5). That is what the sampler produces with steps=4 and
shift=1.0. No custom sampler, no rewritten loop: the student runs on the
stock pipeline with two arguments changed, and guidance_scale=1 turns CFG off
because the sampler skips the unconditional pass when the scale is <= 1.

That identity is asserted against k2e/grid.json rather than trusted, because
a mismatch here would silently evaluate a model on a schedule it never saw.

Env:
  FLIP_STUDENT      student checkpoint (default /kaggle/working/k2e/student_best.pt)
  FLIP_STUDENT_URL  fetch it over HTTP instead. This is the only way to get the
                    weights out of the training kernel: a kernel's output is
                    not the next kernel's input, and attaching a dataset is
                    silently ignored. Kaggle's signed output URLs are readable
                    without cookies, so paste one here.
  FLIP_IMAGE_DIR    images to run; falls back to the repo's examples
  FLIP_N            how many images (default 3)
  FLIP_SEED         sampling seed - shared by every run so they all start
                    from the same noise, which is what makes them comparable
"""

import json
import os
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
OUT = WORK / "k2g"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STUDENT = Path(os.environ.get("FLIP_STUDENT", str(WORK / "k2e/student_best.pt")))
N_IMG = int(os.environ.get("FLIP_N", "3"))
# The student's call count. Must match what it was trained for, or check_grid
# rejects the run before any PLY is written.
S_STEPS = int(os.environ.get("FLIP_STEPS", "4"))
# Which rows to render. A 20-step teacher costs 40 forward calls per image, so
# re-rendering the whole table to add one student row is a waste; pass
# FLIP_TAGS=t20,s02 when only those two matter.
TAGS = [t.strip() for t in os.environ.get("FLIP_TAGS", "").split(",") if t.strip()]
SEED = int(os.environ.get("FLIP_SEED", "42"))
N_GAUSS = 65536
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
R = {"images": [], "notes": []}


def note(msg=""):
    print(msg, flush=True)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=2))


# ---------------------------------------------------------------------------
# the three runs. Kept as data, not control flow: adding a configuration is
# one line, and the report says which PLY came from which.
# ---------------------------------------------------------------------------
RUNS = [
    {"tag": "t20", "label": "teacher 20 steps, guidance 3",
     "steps": 20, "shift": 3.0, "guidance": 3.0, "student": False},
    {"tag": "t08", "label": "teacher 8 steps, guidance 3",
     "steps": 8, "shift": 3.0, "guidance": 3.0, "student": False},
    {"tag": "t04", "label": "teacher 4 steps, guidance 3",
     "steps": 4, "shift": 3.0, "guidance": 3.0, "student": False},
    # Truncating the teacher's own field. A control, not a ceiling - see the
    # note at the top.
    {"tag": "t02", "label": "teacher 2 steps, guidance 3",
     "steps": 2, "shift": 3.0, "guidance": 3.0, "student": False},
    {"tag": f"s{S_STEPS:02d}", "label": f"student {S_STEPS} steps, no guidance",
     "steps": S_STEPS, "shift": 1.0, "guidance": 1.0, "student": True},
]
if TAGS:
    RUNS = [c for c in RUNS if c["tag"] in TAGS]


def fetch_url(url, dst, min_bytes=1):
    """Download to dst atomically, skipping if a complete file is there.

    Re-running a kernel after a partial failure is normal here, and 741 MB is
    not worth paying twice.
    """
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
        note(f"fetch FAILED: {traceback.format_exc()[-600:]}")
        return False
    note(f"  {dst.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s")
    return True


def fetch_student():
    """Get the student weights, by URL if the file is not already on disk."""
    url = os.environ.get("FLIP_STUDENT_URL", "").strip()
    if not url:
        return STUDENT
    dst = STUDENT if STUDENT.suffix == ".pt" else WORK / "k2g/student.pt"
    return dst if fetch_url(url, dst, int(1e6)) else None


def pick_images():
    root = os.environ.get("FLIP_IMAGE_DIR", "").strip()
    if root:
        p = Path(root)
        found = ([p] if p.is_file() else
                 sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT))
        if found:
            return found[:N_IMG]
        note(f"FLIP_IMAGE_DIR {root}: nothing usable")
    ex = REPO / "static" / "example_inputs"
    cands = sorted(q for q in ex.glob("*") if q.suffix.lower() in IMG_EXT)
    if not cands:
        cands = sorted(q for q in (REPO / "static").rglob("*")
                       if q.suffix.lower() in IMG_EXT)
    return cands[:N_IMG]


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

    pipe = T.TripoSplatPipeline(
        ckpt_path=str(ck), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))
    return pipe


def check_grid():
    """Assert the stock sampler reproduces the grid the student was trained on.

    The student only ever saw four (t, dt) pairs. Running it on anything else
    is evaluating a model on a schedule it never saw, and the failure mode is
    quiet: the PLYs would still come out, just bad.
    """
    gj = WORK / "k2e/grid.json"
    gurl = os.environ.get("FLIP_GRID_URL", "").strip()
    if gurl and not gj.exists():
        fetch_url(gurl, gj, 1)
    if not gj.exists():
        note("no k2e/grid.json - cannot check the schedule matches")
        return
    g = json.loads(gj.read_text())
    import numpy as np
    steps, shift = S_STEPS, 1.0
    u = np.linspace(1, 0, steps + 1)
    t_seq = shift * u / (1 + (shift - 1) * u)
    want_t = [round(x * 1000, 4) for x in t_seq[:-1]]
    want_dt = [round(-(t_seq[i] - t_seq[i + 1]), 6) for i in range(steps)]
    got_t = [round(x, 4) for x in g["t"]]
    got_dt = [round(x, 6) for x in g["dt"]]
    note(f"grid.json   t {got_t}  dt {got_dt}")
    note(f"sampler     t {want_t}  dt {want_dt}")
    ok = want_t == got_t and all(abs(a - b) < 2e-6 for a, b in zip(want_dt, got_dt))
    note("schedule MATCHES the frozen grid" if ok else
         "schedule MISMATCH - the student is being run on a grid it never saw")
    R["grid_check"] = {"ok": ok, "grid": g, "sampler_t": want_t,
                       "sampler_dt": want_dt}
    if not ok:
        raise RuntimeError("student schedule does not match grid.json")


def main():
    note("=" * 72)
    note("K2g student PLY")
    note("=" * 72)
    note(f"student {STUDENT}  exists {STUDENT.exists()}")
    check_grid()

    pipe = build_pipeline()
    teacher_sd = {k: v.detach().clone()
                  for k, v in pipe.flow_model.state_dict().items()}
    student_path = fetch_student()
    student_sd = None
    if student_path is not None and student_path.exists():
        blob = torch.load(str(student_path), map_location="cpu")
        student_sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        note(f"loaded student: {len(student_sd)} tensors from {student_path.name}")
    else:
        note("no student checkpoint - only the teacher runs will be produced")

    imgs = pick_images()
    note(f"{len(imgs)} images: {[p.name for p in imgs]}")

    for img_path in imgs:
        rec = {"image": img_path.name, "runs": {}}
        note("-" * 72)
        note(img_path.name)
        image = Image.open(img_path).convert("RGB")
        prepared = pipe.preprocess_image(image)
        prepared.save(OUT / f"prepared_{Path(img_path).stem}.png")

        t0 = time.time()
        cond = pipe.encode_image(prepared)
        rec["encode_s"] = round(time.time() - t0, 2)

        finals = {}
        for cfg in RUNS:
            if cfg["student"] and student_sd is None:
                continue
            note(f"  {cfg['tag']}: {cfg['label']}")
            if cfg["student"]:
                pipe.flow_model.load_state_dict(student_sd)
            else:
                pipe.flow_model.load_state_dict(teacher_sd)
            run = {"label": cfg["label"], "steps": cfg["steps"],
                   "shift": cfg["shift"], "guidance": cfg["guidance"]}
            try:
                # Same seed for every configuration: the three runs start from
                # the same noise, so any difference is the sampler, not luck.
                gen = torch.Generator(device=DEV).manual_seed(SEED)
                torch.cuda.synchronize()
                t0 = time.time()
                lat = pipe.sample_latent(cond, steps=cfg["steps"],
                                         guidance_scale=cfg["guidance"],
                                         shift=cfg["shift"], generator=gen)
                torch.cuda.synchronize()
                run["sample_s"] = round(time.time() - t0, 2)
                finals[cfg["tag"]] = {k: v.detach().float().cpu()
                                      for k, v in lat.items()}

                t0 = time.time()
                g = pipe.decode_latent(lat["latent"], num_gaussians=N_GAUSS)
                torch.cuda.synchronize()
                run["decode_s"] = round(time.time() - t0, 2)
                run["n_gaussians"] = int(g.get_xyz.shape[0])
                run["opacity_mean"] = round(
                    g.get_opacity.detach().float().mean().item(), 4)
                run["scale_mean"] = round(
                    g.get_scaling.detach().float().mean().item(), 5)
                ply = OUT / f"{Path(img_path).stem}_{cfg['tag']}.ply"
                g.save_ply(str(ply))
                run["ply"] = ply.name
                run["ply_mb"] = round(ply.stat().st_size / 1e6, 2)
                note(f"    sample {run['sample_s']}s  decode {run['decode_s']}s  "
                     f"{run['ply_mb']} MB  opacity {run['opacity_mean']}  "
                     f"scale {run['scale_mean']}")
            except Exception:  # noqa: BLE001
                run["error"] = traceback.format_exc()[-1200:]
                note("    FAILED")
                note(traceback.format_exc()[-800:])
            rec["runs"][cfg["tag"]] = run
            save()
            torch.cuda.empty_cache()

        # latent-space distance to the teacher's endpoint, per channel. Useful
        # because it is cheap, and useless as a verdict - the PLYs are that.
        if "t20" in finals:
            for tag, lat in finals.items():
                if tag == "t20":
                    continue
                d = {}
                for ch in lat:
                    a = lat[ch].float().flatten()
                    b = finals["t20"][ch].float().flatten()
                    d[ch] = {
                        "cos": round(torch.nn.functional.cosine_similarity(
                            a, b, dim=0).item(), 4),
                        "rel_l2": round(((a - b).norm() / b.norm()).item(), 4),
                    }
                rec.setdefault("vs_t20", {})[tag] = d
                note(f"    {tag} vs t20  " + "  ".join(
                    f"{ch} cos {m['cos']:.4f} rel-l2 {m['rel_l2']:.3f}"
                    for ch, m in d.items()))
        R["images"].append(rec)
        save()

    note("=" * 72)
    note("SUMMARY")
    for rec in R["images"]:
        for tag, run in rec["runs"].items():
            note(f"  {rec['image']:<28} {tag}  {run.get('sample_s')}s  "
                 f"{run.get('ply_mb')} MB  opacity {run.get('opacity_mean')}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        note(traceback.format_exc())
        save()
        raise
