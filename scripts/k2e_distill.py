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
  FLIP_LR         AdamW lr (default 2e-5)
  FLIP_ACCUM      samples per optimiser step (default 1)
  FLIP_W_CAM      weight on the camera channel's MSE (default 0.05)
  FLIP_GRID       "sigma" (default) or "index" - how the steps are spaced
  FLIP_T_SCALE    dt per unit of recorded t (default: measured from the trace)
  FLIP_CKPT       0 to disable gradient checkpointing
  FLIP_DRY        1 = print the pair table and exit, no training
"""

import copy
import gc
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
LR = float(os.environ.get("FLIP_LR", "2e-5"))
ACCUM = int(os.environ.get("FLIP_ACCUM", "1"))
SAVE_EVERY = int(os.environ.get("FLIP_SAVE_EVERY", "4"))
DRY = os.environ.get("FLIP_DRY", "0") == "1"
# Weight on the camera channel's MSE. Measured, not guessed: unweighted, the
# camera is 5 elements against the latent's 131072 and vanishes - but at 1.0 it
# is worse than vanishing, because its per-segment error is ~2.6 against the
# latent's ~0.48, so it took 97% of epoch 0's gradient and the eval went
# backwards while the loss fell 8x. The camera needs almost no training: it
# starts at 0.996 cosine. This puts the two terms on the same scale.
W_CAM = float(os.environ.get("FLIP_W_CAM", "0.05"))
# "sigma" (default) spaces the student's steps evenly in the schedule;
# "index" spaces them evenly in teacher step count.
GRID_MODE = os.environ.get("FLIP_GRID", "sigma").strip().lower()
# Images never trained on, so the eval measures generalisation, not fit.
HOLDOUT = int(os.environ.get("FLIP_HOLDOUT", "2"))
# Not CKPT - that name is already the weights directory above, and this flag
# shadowed it, which is how a run died on 'bool' object has no attribute
# 'rglob' after five minutes of tracing.
GRAD_CKPT = os.environ.get("FLIP_CKPT", "1") == "1"

# The factor that turns a difference of recorded t values into the dt the
# sampler multiplies the velocity by. Measured before any pair is built unless
# FLIP_T_SCALE says otherwise - guessing it costs a factor of a thousand.
DT_SCALE = 1.0
FIT_T = not os.environ.get("FLIP_T_SCALE", "").strip()
if not FIT_T:
    DT_SCALE = float(os.environ["FLIP_T_SCALE"])

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


def fit_dt_scale(tr, verbose=True):
    """Measure the dt the sampler actually multiplies the velocity by.

    This is the one number I do not want to guess. The trace records t in the
    0..1000 range, and the sampler's step could be the difference of those
    values or the difference of the underlying 0..1 schedule - a factor of a
    thousand either way. Get it wrong and the student is trained to output a
    velocity a thousand times too small, which is not a correction it should
    have to learn.

    So measure it. Every recorded state has two model outputs on it (CFG runs
    the module twice per step), and the next state is known, so

        dx_i = v_eff_i * dt_i          dt_i = <dx_i, v_eff_i> / <v_eff_i, v_eff_i>

    and the only unknown is which of the two calls is the conditional branch.
    Try every plausible combination and keep the one whose implied dt is most
    consistently proportional to the recorded one - the wrong branch assignment
    gives a scatter, the right one gives a straight line.
    """
    states, ts, final = trajectory(tr)
    outs = [as_map(c["out"]) for c in tr["calls"]]
    if len(outs) != 2 * len(states):
        return None
    n = len(states)
    dts = [(ts[i + 1] - ts[i]) if i < n - 1 else (0.0 - ts[i]) for i in range(n)]

    combos = [
        ("cfg 3a-2b", lambda a, b: 3.0 * a - 2.0 * b),
        ("cfg 3b-2a", lambda a, b: 3.0 * b - 2.0 * a),
        ("first only", lambda a, b: a),
        ("second only", lambda a, b: b),
        ("mean", lambda a, b: 0.5 * (a + b)),
    ]
    best = None
    for name, f in combos:
        ratios = []
        for i in range(n):
            v = f(outs[2 * i]["latent"], outs[2 * i + 1]["latent"])
            v = v.float().flatten()
            denom = float(v.dot(v))
            if denom < 1e-12 or abs(dts[i]) < 1e-12:
                continue
            x1 = final["latent"] if i == n - 1 else states[i + 1]["latent"]
            dx = (x1.float().flatten() - states[i]["latent"].float().flatten())
            ratios.append(float(dx.dot(v)) / denom / dts[i])
        if len(ratios) < n // 2:
            continue
        med = sorted(ratios)[len(ratios) // 2]
        spread = (max(ratios) - min(ratios)) / max(abs(med), 1e-12)
        if verbose:
            note(f"  dt fit {name:12s} median {med:.6g}  "
                 f"spread {spread:.3f}  n={len(ratios)}")
        if best is None or spread < best[2]:
            best = (name, med, spread)
    return best


def enable_ckpt(model, min_blocks=4):
    """Checkpoint every repeated stack of blocks so activations are not all
    resident at once.

    One checkpoint around the whole model saves nothing - the recompute
    rebuilds the same graph. The saving comes from many small segments, each
    freed before the next one recomputes.

    Every homogeneous stack of >= min_blocks is wrapped, not just the longest.
    Picking the longest was a guess about which stack is the transformer, and
    a 14 GB allocation says the guess did not pay off: the blocks that
    actually hold the activations may not have been the ones wrapped. Wrapping
    all of them costs a recompute on stacks that did not need it, which is
    minutes, against a run that does not train at all, which is hours.
    """
    import torch.nn as tn
    import torch.utils.checkpoint as cp

    found = []
    for name, mod in model.named_modules():
        kids = list(mod) if isinstance(mod, (tn.ModuleList, tn.Sequential)) else None
        if not kids or len(kids) < min_blocks:
            continue
        if len({type(k).__name__ for k in kids}) != 1:
            continue
        for blk in kids:
            orig = blk.forward

            def wrapped(*a, _o=orig, **kw):
                return cp.checkpoint(_o, *a, use_reentrant=False, **kw)

            blk.forward = wrapped
        n_par = sum(p.numel() for p in mod.parameters())
        found.append(f"{name}: {len(kids)} x {type(kids[0]).__name__} "
                     f"({n_par / 1e6:.0f} M)")
    return found or None


def probe_sdpa(head_dim, n_lat=8192, n_cond=4101):
    """Which attention backend actually works here, and does it materialise
    the N^2 matrix.

    On sm75 flash is out, so the choice is mem-efficient against math. If it
    falls back to math, one layer's backward holds 12293^2 * 2 bytes * heads,
    which is several GB - the difference between fitting and not. Rather than
    assume, run the real shapes and see.
    """
    import torch.nn.attention as attn

    try:
        backends = list(attn.SDPBackend.__members__.values())
    except Exception:  # noqa: BLE001
        return "unknown"
    report, usable = [], []
    for hd in (head_dim, 64, 128):
        q = torch.randn(1, 16, n_lat + n_cond, hd, device=DEV, dtype=torch.float16)
        row = []
        for b in backends:
            try:
                with attn.sdpa_kernel([b]):
                    torch.nn.functional.scaled_dot_product_attention(q, q, q)
                row.append(f"{b.name} ok")
                if hd == head_dim:
                    usable.append(b)
            except Exception as exc:  # noqa: BLE001
                row.append(f"{b.name} no ({type(exc).__name__})")
        report.append(f"head {hd}: " + ", ".join(row))
    note("sdpa: " + " | ".join(report))
    # Prefer anything that is not the math backend. If only math works, say so
    # loudly - the run will be fighting for memory the whole way.
    for b in usable:
        if "FLASH" in b.name:
            return b.name
    for b in usable:
        if "EFFICIENT" in b.name or "CUDNN" in b.name:
            return b.name
    note("MATH ONLY: every attention layer materialises its N^2 matrix, and "
         "backward holds it. Expect to fight for memory the whole run.")
    return "MATH_ONLY"


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


def pick_grid(ts, n_int, s_steps):
    """Which recorded states become the student's step boundaries.

    Equal spacing in sigma, not in step index. The schedule is not linear in
    the step index - the recorded t values are sigma*1000 for sigma =
    3u/(1+2u), u uniform - so splitting 20 teacher steps four ways gives
    dt = (0.1, 0.15, 0.25, 0.5): the last step covers half the distance on its
    own. Euler truncation goes as dt^2, so the split that looks even actually
    puts 0.25 of 0.345 of the total error into one step. Spacing in sigma
    makes all four steps 0.25.

    The indices are the nearest recorded state to each target sigma, forced
    strictly increasing, and the last boundary is the endpoint itself.
    """
    if GRID_MODE == "index":
        step = n_int // s_steps
        if n_int % s_steps:
            note(f"  {n_int} intervals over {s_steps} steps is not even")
        return [min(i * step, n_int) for i in range(s_steps)] + [n_int]
    start = ts[0]
    idx, last = [], -1
    for i in range(s_steps):
        want = start - (start - 0.0) * i / s_steps
        best = min(range(len(ts)), key=lambda j: abs(ts[j] - want))
        if best <= last:
            best = min(last + 1, len(ts) - 1)
        idx.append(best)
        last = best
    return idx + [n_int]


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
        # dt between consecutive recorded states. The last interval is not
        # recorded: the sampler steps from ts[-1] to the endpoint. The recorded
        # t values fit sigma = 3u/(1+2u)*1000 for u uniform on [0.05, 1] to
        # four decimal places, so the endpoint is 0 and the interval is
        # -ts[-1]. Reusing the previous interval instead, as this did, counts
        # it twice and makes the final student step a third too long.
        dts = [ts[i + 1] - ts[i] for i in range(n - 1)]
        if dts:
            dts.append(0.0 - ts[-1])
        dts = [d * DT_SCALE for d in dts]
        all_states = states + [final]
        n_int = len(dts)
        if n_int < S_STEPS:
            note(f"  skip {tr['image']}: {n_int} intervals < {S_STEPS} steps")
            continue
        grid = pick_grid(ts, n_int, S_STEPS)
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
def autocast():
    """fp16 matmuls, fp32 reductions - and no weight cache.

    cache_enabled=False because the student's weights are fp32 master copies:
    autocast would otherwise keep an fp16 copy of all 370 M of them alive for
    the whole run, 741 MB, to save re-casting them per use. That cast is
    milliseconds against 6 seconds a sample.
    """
    return torch.autocast("cuda", dtype=torch.float16,
                          enabled=(DEV.type == "cuda"), cache_enabled=False)


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


def evaluate(student, traces, grid_json, tag):
    """Four student steps against the teacher's twenty, from the same start.

    Run once before training too. That first number is not a formality: it is
    how far plain Euler on the frozen grid lands from the teacher's endpoint
    with the teacher's own weights, i.e. the integration error the student has
    to absorb. Everything the training achieves is measured against it.
    """
    note("=" * 72)
    note(f"eval [{tag}]: student {len(grid_json['t'])} steps vs teacher, "
         f"same start state")
    rows = []
    for tr in traces[:4]:
        states, ts, final = trajectory(tr)
        if len(states) < 2:
            continue
        cond = {k: (v.to(DEV).float() if torch.is_tensor(v) else v)
                for k, v in split_call(tr["calls"][0])[2].items()}
        x0 = states[0]
        try:
            with torch.no_grad(), autocast():
                xs = integrate(student, x0, grid_json["t"],
                               grid_json["dt"], cond)
            out = {}
            for ch in ("latent", "camera"):
                if ch not in xs or ch not in final:
                    continue
                a = xs[ch].float().flatten()
                b = final[ch].to(DEV).float().flatten()
                out[ch] = {
                    "cos": round(torch.nn.functional.cosine_similarity(
                        a, b, dim=0).item(), 4),
                    "rel_l2": round(((a - b).norm() / b.norm()).item(), 4),
                }
            note(f"  {tr['image']}  " + "  ".join(
                f"{ch} cos {m['cos']:.4f} rel-l2 {m['rel_l2']:.3f}"
                for ch, m in out.items()))
            rows.append({"image": tr["image"], "tag": tag, "channels": out})
        except Exception:  # noqa: BLE001
            note(f"  {tr['image']} eval failed")
            note(traceback.format_exc()[-800:])
    R["eval"].extend(rows)
    return rows


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

    global DT_SCALE
    if FIT_T:
        note("fitting the sampler's dt against the recorded states")
        fit = fit_dt_scale(first) or fit_dt_scale(traces[-1])
        if fit is None:
            note("  could not fit - assuming recorded t differences are the dt")
            DT_SCALE = 1.0
        else:
            name, med, spread = fit
            DT_SCALE = med
            note(f"  {name}: dt = {med:.6g} x the recorded t difference "
                 f"(spread {spread:.3f})")
            if spread > 0.2:
                note(f"  WARNING spread {spread:.2f} - the branch assignment may "
                     f"be wrong; set FLIP_T_SCALE to override")
    else:
        note(f"dt scale from FLIP_T_SCALE: {DT_SCALE}")
    note(f"dt scale {DT_SCALE:.6g}  "
         f"(1 = the velocity is per unit of recorded t, "
         f"~1e-3 = per unit of the underlying 0..1 schedule)")

    pairs, grid = build_pairs(traces)
    if HOLDOUT:
        # Evaluating on images the student trained on only measures fit. With
        # this few objects, "fit" and "memorised" are hard to tell apart, and
        # the whole point is whether it generalises to the photo someone drops
        # in. So the first few images are never trained on.
        held = {tr["image"] for tr in traces[:HOLDOUT]}
        before_n = len(pairs)
        pairs = [p for p in pairs if p["image"] not in held]
        note(f"held out {sorted(held)} - {len(pairs)} of {before_n} pairs left")
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
    # The pipeline was only a way to construct the DiT with the right weights.
    # DINOv3, the VAE, RMBG and the decoder are several GB of fp16 on the card
    # and are dead weight for the next three hours, so give the memory back
    # before allocating fp32 master weights and Adam state.
    del pipe
    gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
        note(f"freed the pipeline: {torch.cuda.memory_allocated() / 1e9:.2f} GB still live, "
             f"{torch.cuda.mem_get_info()[0] / 1e9:.2f} GB free")
    student = copy.deepcopy(teacher).float()
    student.train()
    n_par = sum(p.numel() for p in student.parameters())
    note(f"student params {n_par / 1e6:.1f} M  fp32 Adam state ~{n_par * 12 / 1e9:.1f} GB")
    if DEV.type == "cuda":
        note(f"sdpa backend in use: {probe_sdpa(80)}")
    with torch.no_grad():
        evaluate(student, traces, grid_json, "before")
    # The teacher is 741 MB of fp16 sitting on the same card for no reason
    # now that the student is a deepcopy of it.
    teacher.to("cpu")
    del teacher
    gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
        note(f"dropped the teacher: {torch.cuda.memory_allocated() / 1e9:.2f} GB live")
    if GRAD_CKPT:
        # The smoke run skipped every single sample on OOM: inference fits
        # easily because nothing is retained, but a backward through 24 layers
        # over 8192 latent tokens plus 4101 conditioning tokens does not.
        where = enable_ckpt(student)
        note(f"gradient checkpointing on {where}")
        if where is None:
            note("  NO repeated block list found - nothing is checkpointed and "
                 "the activations will all be resident")
            note("  no repeated block list found - dumping the structure so the "
                 "next run can target it by name")
            for name, mod in student.named_children():
                kids = [type(k).__name__ for k in mod.children()]
                note(f"    {name}: {type(mod).__name__} "
                     f"{kids[:6]}{' ...' if len(kids) > 6 else ''}")

    # fp32 master weights matter more than they look. The pipeline hands over
    # fp16 params, and fp16 carries about three decimal digits: at lr 5e-5 an
    # update is ~1e-5 against weights of ~1e-2, which rounds to nothing.
    # Autocast runs the matmuls in fp16 either way, so this costs memory only.
    # weight_decay 0: the student starts from the teacher and is meant to stay
    # near it. Pulling 370 M weights toward zero while asking them to reproduce
    # the teacher is two objectives, and only one of them is the task.
    opt = torch.optim.AdamW(student.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.0)
    # T4 is sm75: no bf16 cores, so fp16 with a scaler. LayerNorm stays fp32
    # under autocast, which is the same reason dinov3 needed the fp32-stats fix.
    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEV.type == "cuda"))

    def dump(tag="student.pt"):
        torch.save({"model": student.state_dict(), "grid": grid_json,
                    "s_steps": S_STEPS}, OUT / tag)
        note(f"  saved {tag} {(OUT / tag).stat().st_size / 1e9:.2f} GB")

    best = float("inf")
    t0 = time.time()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(pairs)).tolist()
        run_loss, run_n, skipped, run_terms = 0.0, 0, 0, {}
        opt.zero_grad(set_to_none=True)
        for bi, idx in enumerate(order):
            p = pairs[idx]
            try:
                state = to_dev(p["state"])
                tgt = {k: v.to(DEV) for k, v in p["target"].items()}
                cond = {k: (v.to(DEV).float() if torch.is_tensor(v) else v)
                        for k, v in p["cond"].items()}
                with autocast():
                    v = forward_velocity(student, state, p["t"], cond)
                    terms = {k: torch.nn.functional.mse_loss(v[k].float(), tgt[k])
                             for k in tgt if k in v}
                    # Latent is 8192x16 and camera is 1x5. Summing the two MSEs
                    # unweighted gives camera a 4e-5 share of the gradient and
                    # it never gets learned, even though it drifts by more than
                    # its own norm over the trajectory.
                    loss = (sum(terms[k] * (W_CAM if k == "camera" else 1.0)
                                for k in terms) / ACCUM)
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError as exc:
                # A three-hour run should not end because one sample was the
                # first to tip the allocator over. Drop the sample's gradients
                # and carry on - but say why, because "skipped 24 of 24" with
                # no reason is exactly how a swallowed bug looks.
                if skipped == 0:
                    note(f"  OOM on sample {bi}: {exc}")
                    if DEV.type == "cuda":
                        note(f"    peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, "
                             f"{torch.cuda.mem_get_info()[0] / 1e9:.2f} GB free now")
                skipped += 1
                opt.zero_grad(set_to_none=True)
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            run_loss += loss.item() * ACCUM
            run_n += 1
            for k, val in terms.items():
                run_terms[k] = run_terms.get(k, 0.0) + float(val)
            if run_n == 1 and DEV.type == "cuda":
                # How much room is left. The next sample is the same size, so
                # this is the number that decides whether the run survives.
                note(f"  first sample peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                     f"of {torch.cuda.mem_get_info()[1] / 1e9:.2f} GB")
            if (bi + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        per = "  ".join(f"{k} {v / max(run_n, 1):.5f}" for k, v in sorted(run_terms.items()))
        note(f"epoch {epoch:3d}  loss {run_loss / max(run_n, 1):.6f}  {per}  "
             f"skipped {skipped}  {(time.time() - t0) / 60:.1f} min")
        R["training"].append({"epoch": epoch, "loss": run_loss / max(run_n, 1),
                              "skipped": skipped,
                              "terms": {k: v / max(run_n, 1)
                                        for k, v in run_terms.items()}})
        save()
        if SAVE_EVERY and (epoch + 1) % SAVE_EVERY == 0:
            dump(f"student_e{epoch + 1:03d}.pt")
        # Evaluate every epoch and keep the best. The loss falling is not the
        # same as the reconstruction improving - it did not, last time - and a
        # three-hour run that ends on a worse model than it started with has
        # nothing to show for itself. Two images is ~16 s.
        student.eval()
        with torch.no_grad():
            rows = evaluate(student, traces[:2], grid_json, f"e{epoch + 1}")
        student.train()
        scores = [r["channels"]["latent"]["rel_l2"] for r in rows
                  if "latent" in r.get("channels", {})]
        if scores:
            score = sum(scores) / len(scores)
            if score < best:
                best = score
                dump("student_best.pt")
                note(f"  best so far: rel-l2 {score:.3f}")
        if skipped > len(pairs) // 2:
            # Do not spend three hours proving the same thing 1536 times.
            note(f"STOPPING: {skipped} of {len(pairs)} samples did not fit. "
                 f"Nothing was learned; fix the memory budget first.")
            break

    student.eval()
    evaluate(student, traces, grid_json, "final")

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
