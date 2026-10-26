"""K2e - turn the recorded teacher trajectory into a 4-step student.

The naive distillation target is "match the teacher's velocity at (x, t)". That
target is wrong for our purpose, and it is worth being explicit about why: the
teacher's velocity field is already correct at 20 steps, and a student trained
to reproduce it call-for-call would reproduce it just as correctly at 4 calls,
where the ODE integration is too coarse. The blur in K2b/K2c is an integration
error, not a field error.

What the student actually has to learn is: given the state at t_k, move the
whole distance that the teacher covers in the next FIVE steps, in one call.
That is the average velocity over the segment,

    v* = (x_{k+n} - x_k) / sum(dt_i)      n = teacher_steps / student_steps

and Euler-integrating that with the same dt lands exactly on the teacher's
endpoint. Three consequences, all good:

  * CFG folds in for free. The recorded states came out of the guided sampler,
    so the segments already contain the guidance. No branch bookkeeping, no
    gain constant to tune - the trace's two conditional branches become
    irrelevant to training.
  * No teacher forward pass in the training loop. Every target is a subtraction
    of two states already on disk. Training is one student forward+backward per
    sample, ~11 s on a T4 instead of ~50.
  * The student only ever sees the grid it will be run on, so no train/test
    mismatch at inference - but it also means the grid is frozen. The same
    constants must be used by the browser sampler; they are written to
    grid.json for exactly that reason.

The state is a dict: latent is [1, 8192, 16] and camera is [1, 1, 5], and the
sampler integrates both. Both are advanced here, because freezing the camera
would feed the student a state at steps 2..4 that the teacher never visited.

Writes k2e/student.pt, grid.json, and report.json.

Env:
  FLIP_TRACE_DIR  where k2d wrote its traces (default /kaggle/working/k2d)
  FLIP_STEPS      student calls (default 4)
  FLIP_EPOCHS     passes over the pair set (default 40)
  FLIP_LR         AdamW lr (default 5e-5)
  FLIP_ACCUM      samples per optimiser step (default 4)
  FLIP_DRY        1 = print the pair table and exit, no training
"""

import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
TRACE_DIR = Path(os.environ.get("FLIP_TRACE_DIR", str(WORK / "k2d")))
OUT = WORK / "k2e"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S_STEPS = int(os.environ.get("FLIP_STEPS", "4"))
EPOCHS = int(os.environ.get("FLIP_EPOCHS", "40"))
LR = float(os.environ.get("FLIP_LR", "5e-5"))
ACCUM = int(os.environ.get("FLIP_ACCUM", "4"))
DRY = os.environ.get("FLIP_DRY", "0") == "1"

R = {"notes": [], "training": [], "eval": []}


def note(msg):
    line = str(msg)
    print(line, flush=True)
    R["notes"].append(line)


def save(name="report.json"):
    (OUT / name).write_text(json.dumps(R, indent=2, default=str))


# ---------------------------------------------------------------------------
# reading the trace
# ---------------------------------------------------------------------------

def load_trace(path):
    """Same shape as k2d writes; `$ref` markers resolve against `shared`.

    Duplicated from the trace script on purpose: on Kaggle only the fetched
    script exists on disk, so there is nothing to import. It is fifteen lines.
    """
    p = torch.load(path, map_location="cpu", weights_only=False)

    def rehydrate(x):
        if isinstance(x, dict):
            if "$ref" in x and len(x) == 1:
                return p["shared"][x["$ref"]]
            return {k: rehydrate(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rehydrate(v) for v in x]
        return x

    p["calls"] = [{"args": rehydrate(c["args"]),
                   "kwargs": rehydrate(c["kwargs"]),
                   "out": rehydrate(c["out"]),
                   **{k: v for k, v in c.items() if k not in ("args", "kwargs", "out")}}
                  for c in p["calls"]]
    return p


def as_map(state):
    """States come as {'latent', 'camera'} or, hypothetically, a bare tensor."""
    if torch.is_tensor(state):
        return {"latent": state}
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


def state_latent(state):
    m = as_map(state)
    for k in ("latent", "x", "sample"):
        if k in m:
            return m[k]
    raise KeyError(f"no latent in state: {list(m)}")


def split_call(call):
    """Recover (state, t, cond) from whatever the sampler actually passed.

    The trace stores raw args, so this has to tolerate positional and keyword
    calls. k2d printed the real signature on its first run - args were
    ({latent, camera}, t, {feature1, feature2}) - precisely so this could be
    written against fact instead of guesswork.
    """
    args, kwargs = call["args"], call["kwargs"]
    state = t = cond = None
    for a in args:
        if isinstance(a, dict) and set(a) & {"latent", "x", "sample"}:
            state = a
        elif torch.is_tensor(a) and a.numel() == 1:
            t = a
        elif isinstance(a, dict):
            cond = a
        elif torch.is_tensor(a) and a.numel() > 1 and state is None:
            state = {"latent": a}
        elif isinstance(a, (int, float)) and t is None:
            t = torch.tensor(float(a))
    for k, v in kwargs.items():
        lk = k.lower()
        if "state" in lk or lk in ("x", "inputs"):
            state = v
        elif lk in ("t", "timestep", "time"):
            t = v
        elif "cond" in lk or lk == "c":
            cond = v
    if state is None or t is None:
        raise RuntimeError(f"cannot split call: args={[type(a).__name__ for a in args]} "
                           f"kwargs={list(kwargs)}")
    return state, t, cond


def trajectory(trace):
    """Collapse the call list into the sequence of distinct states.

    CFG runs the module twice per step on the same state, so consecutive
    identical states have to be dropped or the paired target would be a zero
    displacement. Ordering is preserved; nothing else is assumed.
    """
    states, ts, seen_last = [], [], None
    for call in trace["calls"]:
        state, t, _ = split_call(call)
        m = as_map(state)
        key = (float(torch.as_tensor(t).flatten()[0].float()),
               round(sum(float(v.float().sum()) for v in m.values()), 3))
        if key == seen_last:
            continue
        seen_last = key
        states.append(m)
        ts.append(float(torch.as_tensor(t).flatten()[0].float()))
    return states, ts, as_map(trace["final"])


def build_pairs(traces):
    """One pair per student step boundary: (state_k, t_k, v*, dt, cond)."""
    pairs = []
    grid = None
    for tr in traces:
        states, ts, final = trajectory(tr)
        if len(states) < 2:
            note(f"  skip {tr['image']}: only {len(states)} distinct states")
            continue
        cond = split_call(tr["calls"][0])[2]
        n = len(states)
        # dt between consecutive recorded states. One update per call, so there
        # are n intervals for n recorded states - the last one lands on `final`
        # and has no recorded t of its own. Reuse the previous interval for it
        # rather than inventing a terminal t.
        dts = [ts[i + 1] - ts[i] for i in range(n - 1)]
        if dts:
            dts.append(dts[-1])
        all_states = states + [final]
        n_int = len(dts)
        if n_int < S_STEPS:
            note(f"  skip {tr['image']}: {n_int} intervals < {S_STEPS} steps")
            continue
        step = n_int // S_STEPS
        if n_int % S_STEPS:
            note(f"  {tr['image']}: {n_int} intervals over {S_STEPS} steps "
                 f"is not even - segments of {step} and one longer")
        grid = [min(i * step, n_int) for i in range(S_STEPS)] + [n_int]
        for k in range(S_STEPS):
            i, j = grid[k], grid[k + 1]
            x0, x1 = all_states[i], all_states[j]
            span = sum(dts[i:j])
            keys = [k2 for k2 in x0 if k2 in x1]
            target = {k2: (x1[k2].float() - x0[k2].float()) / span for k2 in keys}
            move = sum(float(v.norm()) for v in target.values())
            if abs(span) < 1e-9 or move < 1e-9:
                note(f"  {tr['image']} k={k}: empty segment "
                     f"(dt {span:.2e}, move {move:.2e}) - skipped")
                continue
            pairs.append({
                "image": tr["image"],
                "k": k,
                "state": {k2: v for k2, v in x0.items()},
                "t": ts[min(i, len(ts) - 1)],
                "dt": span,
                "target": target,
                "cond": cond,
            })
    return pairs, grid


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

def build_pipeline():
    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    files = sorted(CKPT.rglob("*.safetensors"))

    def pick(*keys):
        for f in files:
            nm = f.name.lower()
            if all(k in nm for k in keys):
                return f
        return None

    dino_p, vae_p = pick("dino"), (pick("vae") or pick("flux"))
    rmbg_p = pick("biref") or pick("rmbg")
    dec_p = pick("decoder") or pick("octree")
    taken = {p for p in (dino_p, vae_p, rmbg_p, dec_p) if p}
    ck = [f for f in files if f not in taken][0]

    pipe = T.TripoSplatPipeline(
        ckpt_path=str(ck), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))
    mods = {type(v).__name__: v for v in vars(pipe).values() if isinstance(v, nn.Module)}
    flow = mods.get("LatentSeqMMFlowModel")
    if flow is None:
        raise RuntimeError(f"no LatentSeqMMFlowModel; saw {list(mods)}")
    return pipe, flow


def forward_velocity(model, state_map, t, cond):
    """Call the flow module the way the sampler does.

    `t` is passed through untouched: the trace records what the module actually
    received, which is already the scaled convention (k0b's export adapter
    multiplies by 1000, so the trace's t values sit in the hundreds-to-1000
    range). Scaling it again would hand the model t=1e6 and teach it nothing.
    """
    t_in = torch.as_tensor([t], device=DEV, dtype=torch.float32)
    out = model(state_map, t_in, cond)
    return as_map(out)


def integrate(student, x0, t_values, dts, cond):
    """Euler with the student's own predictions - the exact loop the browser
    will run, written in Python first so a bug shows up here and not in WGSL."""
    x = {k: v.to(DEV).float().clone() for k, v in x0.items()}
    for k in range(len(t_values)):
        v = forward_velocity(student, x, t_values[k], cond)
        for k2 in list(x):
            if k2 in v:
                x[k2] = x[k2] + v[k2].float() * dts[k]
    return x


def to_dev(m):
    """Trace states are stored as half to keep the files small; the student's
    weights are fp32, and outside autocast an fp16 input against fp32 weights
    fails on the elementwise ops."""
    return {k: v.to(DEV).float() for k, v in m.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    note("=" * 72)
    note("K2e student distillation")
    note("=" * 72)
    note(f"device {DEV}  student steps {S_STEPS}  epochs {EPOCHS}  lr {LR}  "
         f"accum {ACCUM}  dry {DRY}")

    files = sorted(TRACE_DIR.glob("trace_*.pt")) if TRACE_DIR.is_dir() else []
    if not files:
        # k2d ran in an earlier session; its output is attached as an input
        # dataset, so the traces live under /kaggle/input, not here.
        files = sorted(Path("/kaggle/input").glob("*/**/trace_*.pt"))
        note(f"no traces in {TRACE_DIR}, found {len(files)} under /kaggle/input")
    if not files:
        raise RuntimeError(f"no traces in {TRACE_DIR} or under /kaggle/input")
    note(f"{len(files)} traces")

    traces = []
    for f in files:
        try:
            traces.append(load_trace(f))
        except Exception:  # noqa: BLE001
            note(f"  {f.name} unreadable")
            note(traceback.format_exc()[-600:])
    note(f"{len(traces)} loaded")

    first = traces[0]
    states, ts, final = trajectory(first)
    note(f"states per image {len(states)}  distinct t {len(set(ts))}")
    note(f"t range {min(ts):.1f} .. {max(ts):.1f}")
    for k, v in states[0].items():
        note(f"  {k} {tuple(v.shape)} {v.dtype}")
    note(f"n_calls {len(first['calls'])} -> {len(states)} unique states "
         f"({'two forwards per step: CFG' if len(first['calls']) == 2 * len(states) else 'flat'})")
    drift = max(float(states[i]["camera"].float().sub(
        states[0]["camera"].float()).norm()) for i in range(len(states)))
    note(f"camera drift over the trajectory {drift:.4f} "
         f"(|camera| {float(states[0]['camera'].float().norm()):.4f}) "
         f"- if this is not zero both channels have to be integrated")

    pairs, grid = build_pairs(traces)
    note(f"{len(pairs)} training pairs from {len(traces)} images")
    if pairs:
        mags = [float(p["target"]["latent"].abs().mean()) for p in pairs]
        note(f"|v*| mean {sum(mags) / len(mags):.5f}  "
             f"min {min(mags):.5f}  max {max(mags):.5f}")

    dt_by_k, t_by_k = {}, {}
    for p in pairs:
        dt_by_k.setdefault(p["k"], []).append(p["dt"])
        t_by_k.setdefault(p["k"], []).append(p["t"])
    for k, v in sorted(dt_by_k.items()):
        if max(v) - min(v) > 1e-6:
            note(f"  warning: dt varies across images at k={k}: "
                 f"{min(v):.6f}..{max(v):.6f}")
    grid_json = {
        "student_steps": S_STEPS,
        "t": [round(sum(t_by_k[k]) / len(t_by_k[k]), 4) for k in sorted(t_by_k)],
        "dt": [round(sum(v) / len(v), 6) for k, v in sorted(dt_by_k.items())],
        "note": "frozen: the student only saw this grid, the browser must use it too",
    }
    (OUT / "grid.json").write_text(json.dumps(grid_json, indent=2))
    note(f"grid t {grid_json['t']}")
    note(f"grid dt {grid_json['dt']}")

    if DRY:
        note("dry run - stopping before any training")
        save()
        return

    pipe, teacher = build_pipeline()
    note(f"teacher params {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f} M  "
         f"dtype {next(teacher.parameters()).dtype}")
    student = copy.deepcopy(teacher).float()
    student.train()
    n_par = sum(p.numel() for p in student.parameters())
    note(f"student params {n_par / 1e6:.1f} M  fp32 Adam state ~{n_par * 12 / 1e9:.1f} GB")

    # fp32 master weights matter more than they look. The pipeline hands over
    # fp16 params, and fp16 carries about three decimal digits: at lr 5e-5 an
    # update is ~1e-5 against weights of ~1e-2, which rounds to nothing.
    # Autocast runs the matmuls in fp16 either way, so this costs memory only.
    opt = torch.optim.AdamW(student.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.01)
    # T4 is sm75: no bf16 cores, so fp16 with a scaler. LayerNorm stays fp32
    # under autocast, which is the same reason dinov3 needed the fp32-stats fix.
    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEV.type == "cuda"))

    t0 = time.time()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(pairs)).tolist()
        run_loss, run_n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, idx in enumerate(order):
            p = pairs[idx]
            state = to_dev(p["state"])
            tgt = {k: v.to(DEV) for k, v in p["target"].items()}
            cond = {k: (v.to(DEV).float() if torch.is_tensor(v) else v)
                    for k, v in p["cond"].items()}
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=(DEV.type == "cuda")):
                v = forward_velocity(student, state, p["t"], cond)
                loss = sum(torch.nn.functional.mse_loss(
                    v[k].float(), tgt[k]) for k in tgt if k in v) / ACCUM
            scaler.scale(loss).backward()
            run_loss += loss.item() * ACCUM
            run_n += 1
            if (bi + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        note(f"epoch {epoch:3d}  loss {run_loss / max(run_n, 1):.6f}  "
             f"{(time.time() - t0) / 60:.1f} min")
        R["training"].append({"epoch": epoch, "loss": run_loss / max(run_n, 1)})
        save()

    student.eval()
    note("=" * 72)
    note("eval: student 4 steps vs teacher 20 steps, same start state")
    for tr in traces[:4]:
        states, ts, final = trajectory(tr)
        if len(states) < 2:
            continue
        cond = {k: (v.to(DEV).float() if torch.is_tensor(v) else v)
                for k, v in split_call(tr["calls"][0])[2].items()}
        x0 = states[0]
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                 enabled=(DEV.type == "cuda")):
                xs = integrate(student, x0, grid_json["t"], grid_json["dt"], cond)
            a = xs["latent"].float().flatten()
            b = final["latent"].to(DEV).float().flatten()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            rel = ((a - b).norm() / b.norm()).item()
            note(f"  {tr['image']}  cos {cos:.4f}  rel-l2 {rel:.3f}")
            R["eval"].append({"image": tr["image"], "cos": round(cos, 4),
                              "rel_l2": round(rel, 4)})
        except Exception:  # noqa: BLE001
            note(f"  {tr['image']} eval failed")
            note(traceback.format_exc()[-800:])

    torch.save({"model": student.state_dict(), "grid": grid_json,
                "s_steps": S_STEPS}, OUT / "student.pt")
    note(f"wrote {OUT / 'student.pt'} "
         f"({(OUT / 'student.pt').stat().st_size / 1e9:.2f} GB)")
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        note(traceback.format_exc())
        save()
        raise
