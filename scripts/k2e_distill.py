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

Writes k2e/student.pt, grid.json, eval.json, and eval PLYs.

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
import math
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
N_GAUSS = 65536
SEED = 7

R = {"notes": [], "pairs": [], "training": [], "eval": []}


def note(msg):
    line = str(msg)
    print(line, flush=True)
    R["notes"].append(line)


def save(name="report.json"):
    (OUT / name).write_text(json.dumps(
        {k: v for k, v in R.items() if k != "pairs"}, indent=2, default=str))


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


def state_latent(state):
    if torch.is_tensor(state):
        return state
    if isinstance(state, dict):
        for k in ("latent", "x", "sample"):
            if k in state:
                return state[k]
        raise KeyError(f"no latent in state dict: {list(state)}")
    raise TypeError(type(state).__name__)


def split_call(call):
    """Recover (state, t, cond) from whatever the sampler actually passed.

    The trace stores the raw args, so this has to tolerate positional and
    keyword calls. k2d printed the shapes on its first run precisely so this
    function could be written against the real thing.
    """
    args, kwargs = call["args"], call["kwargs"]
    state = t = cond = None
    positional = [a for a in args if not isinstance(a, (int, float))]
    for a in positional:
        if isinstance(a, dict) and any(k in a for k in ("latent", "x", "sample")):
            state = a
        elif torch.is_tensor(a) and a.numel() == 1:
            t = a
        elif isinstance(a, dict):
            cond = a
        elif torch.is_tensor(a) and a.dim() >= 1 and a.numel() > 1:
            # a bare latent tensor, no camera
            state = {"latent": a}
    for k, v in kwargs.items():
        lk = k.lower()
        if "state" in lk or lk in ("x", "inputs"):
            state = v
        elif lk in ("t", "timestep", "time"):
            t = v
        elif "cond" in lk or lk == "c":
            cond = v
    for a in args:
        if isinstance(a, (int, float)) and t is None:
            t = torch.tensor(float(a))
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
        x = state_latent(state)
        key = (float(torch.as_tensor(t).flatten()[0].float()),
               round(float(x.float().sum()), 4))
        if key == seen_last:
            continue
        seen_last = key
        states.append(x)
        ts.append(float(torch.as_tensor(t).flatten()[0].float()))
    final = trace["final"]
    if isinstance(final, dict):
        final = state_latent(final)
    return states, ts, final


def build_pairs(traces):
    """One pair per student step boundary: (x_k, t_k, v*, dt, cond) per image."""
    pairs = []
    grid = None
    for tr in traces:
        states, ts, final = trajectory(tr)
        if len(states) < 2:
            note(f"  skip {tr['image']}: only {len(states)} distinct states")
            continue
        cond = split_call(tr["calls"][0])[2]
        n = len(states)
        # dt between consecutive recorded states. The sampler makes one update
        # per call, so there are n intervals for n recorded states - the last
        # one lands on `final` and has no recorded t of its own. Reuse the
        # previous interval for it rather than inventing a terminal t.
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
            x0 = all_states[i]
            x1 = all_states[j]
            span = sum(dts[i:j])
            move = (x1.float() - x0.float()).norm().item()
            if abs(span) < 1e-9 or move < 1e-9:
                note(f"  {tr['image']} k={k}: empty segment "
                     f"(dt {span:.2e}, move {move:.2e}) - skipped")
                continue
            pairs.append({
                "image": tr["image"],
                "k": k,
                "state": x0,
                "t": ts[min(i, len(ts) - 1)],
                "dt": span,
                "target": (x1.float() - x0.float()) / span,
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


def forward_velocity(model, state, t, cond):
    """Call the flow module the way the sampler does.

    `t` is passed through untouched: the trace records what the module actually
    received, which is already the scaled convention (k0b's export adapter
    multiplies by 1000, so the trace's t values sit in the thousands). Scaling
    it again here would hand the model t=1e6 and teach it nothing.
    """
    t_in = torch.as_tensor([t], device=DEV, dtype=torch.float32)
    out = model(state, t_in, cond)
    v = state_latent(out)
    return v if torch.is_tensor(v) else torch.as_tensor(v)


def integrate(student, x0, t_values, dts, cond):
    """Euler with the student's own predictions - the exact inference loop the
    browser will run, in Python."""
    x = x0
    for k in range(len(t_values)):
        v = forward_velocity(student, x, t_values[k], cond)
        x = x + v.float() * dts[k]
    return x


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    note("=" * 72)
    note("K2e student distillation")
    note("=" * 72)
    note(f"device {DEV}  student steps {S_STEPS}  epochs {EPOCHS}  lr {LR}  "
         f"accum {ACCUM}  dry {DRY}")

    files = sorted(TRACE_DIR.glob("trace_*.pt"))
    if not files:
        raise RuntimeError(f"no traces in {TRACE_DIR}")
    note(f"{len(files)} traces in {TRACE_DIR}")
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
    note(f"latent shape {tuple(state_latent(states[0]).shape)}  "
         f"dtype {state_latent(states[0]).dtype}")
    note(f"n_calls {len(first['calls'])}  -> {len(states)} unique states "
         f"({'CFG does 2 forwards/step' if len(first['calls']) > 2 * len(states) else 'flat'})")

    pairs, grid = build_pairs(traces)
    note(f"{len(pairs)} training pairs from {len(traces)} images")
    R["pairs"] = [{"image": p["image"], "k": p["k"], "t": round(p["t"], 2),
                   "dt": round(p["dt"], 6)} for p in pairs]
    if pairs:
        mags = [p["target"].abs().mean().item() for p in pairs]
        note(f"|v*| mean {sum(mags) / len(mags):.4f}  "
             f"min {min(mags):.4f}  max {max(mags):.4f}")

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
    note(f"grid {grid_json['t']}  dt {grid_json['dt']}")

    if DRY:
        note("dry run - stopping before any training")
        save()
        return

    pipe, teacher = build_pipeline()
    note(f"teacher params {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f} M")
    student = copy.deepcopy(teacher)
    student.train()
    n_par = sum(p.numel() for p in student.parameters())
    note(f"student params {n_par / 1e6:.1f} M  fp32 Adam state ~{n_par * 12 / 1e9:.1f} GB")

    opt = torch.optim.AdamW(student.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.01)
    # T4 is sm75: no bf16, so fp16 with a scaler. LayerNorm stays fp32 under
    # autocast, which is the same reason dinov3 needed the fp32-stats fix.
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV.type == "cuda"))

    def transfer(p):
        return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in p["cond"].items()} \
            if isinstance(p["cond"], dict) else p["cond"]

    t0 = time.time()
    step = 0
    for epoch in range(EPOCHS):
        order = torch.randperm(len(pairs)).tolist()
        run_loss, run_n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, idx in enumerate(order):
            p = pairs[idx]
            state = {k: v.to(DEV) for k, v in p["state"].items()} \
                if isinstance(p["state"], dict) else p["state"].to(DEV)
            tgt = p["target"].to(DEV)
            cond = transfer(p)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV.type == "cuda")):
                v = forward_velocity(student, state, p["t"], cond)
                loss = torch.nn.functional.mse_loss(v.float(), tgt) / ACCUM
            scaler.scale(loss).backward()
            run_loss += loss.item() * ACCUM
            run_n += 1
            if (bi + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
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
        cond = transfer({"cond": split_call(tr["calls"][0])[2]})
        x0 = states[0]
        x0 = {k: v.to(DEV) for k, v in x0.items()} if isinstance(x0, dict) else x0.to(DEV)
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                 enabled=(DEV.type == "cuda")):
                xs = integrate(student, x0, grid_json["t"], grid_json["dt"], cond)
            fin = final.to(DEV) if torch.is_tensor(final) else final
            a = xs.float().flatten()
            b = fin.float().flatten()
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
