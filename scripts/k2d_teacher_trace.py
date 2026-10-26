"""K2d - record what the flow DiT actually does, step by step.

Why: the 4-step problem is not solvable by choosing a nicer number of steps.
The base model at 4 steps loses material (K2b, K2c): skin goes pale, black
shorts go pink, hair turns to flying threads. A 4-step *student* is a different
model - it is trained to land the same place in 4 steps that the teacher takes
20 for. That needs the teacher's velocity field as a function of (state, t),
which is what this script records, once, on Kaggle.

The trick that keeps this short: do not reimplement the sampler. Run the real
`pipe.sample_latent` and intercept the flow module's forward. Every call gives
(state, t, cond) -> velocity, for free, including both branches of CFG.

Writes k2d/trace_XXXX.pt per image plus trace.json summary.

Env:
  FLIP_TRACE_N     images to trace (default 8; the smoke run)
  FLIP_TRACE_STEPS teacher steps (default 20)
  FLIP_TRACE_GAIN  guidance scale (default 3.0, the pipeline default)
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k2d"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_IMG = int(os.environ.get("FLIP_TRACE_N", "8"))
STEPS = int(os.environ.get("FLIP_TRACE_STEPS", "20"))
GAIN = float(os.environ.get("FLIP_TRACE_GAIN", "3.0"))
SHIFT = float(os.environ.get("FLIP_TRACE_SHIFT", "3.0"))
SEED = 42

R = {"calls": [], "images": [], "notes": []}


def note(msg):
    line = str(msg)
    print(line, flush=True)
    R["notes"].append(line)


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

def _hash(t):
    """Cheap content id for a tensor, so identical cond features are stored once."""
    if t is None:
        return "none"
    b = t.detach().float().cpu().contiguous().numpy().tobytes()
    import hashlib
    return hashlib.sha1(b).hexdigest()[:16]


class Recorder:
    """Wraps the flow module's forward and keeps every call.

    Patching the class, not the instance: `__call__` resolves on the type, so an
    instance attribute would be silently bypassed. K0e already burned me once on
    exactly this - a monkeypatch that never entered the graph.

    The patch is a plain function, so it binds like any other method and `self`
    is the flow module. Assigning a bound method instead would make the module
    itself land in `args` and get saved into the trace.
    """

    def __init__(self, flow):
        self.flow = flow
        self.calls = []
        self._cls = flow.__class__
        self._orig = self._cls.forward
        self._cls.forward = self._make()
        self.t0 = time.time()

    def _make(self):
        rec = self

        def patched(self, *a, **kw):
            out = rec._orig(self, *a, **kw)
            if self is rec.flow:
                try:
                    rec._keep(a, kw, out)
                except Exception:  # noqa: BLE001 - never break the sampler
                    note("record failed: " + traceback.format_exc()[-500:])
            return out

        return patched

    def restore(self):
        self._cls.forward = self._orig

    def _keep(self, a, kw, out):
        """Store the call verbatim rather than guessing its signature.

        I know the module takes ({latent, camera}, t*1000, {feature1, feature2})
        from the export adapter, but I do not know how `sample_latent` actually
        calls it - positionally, by keyword, or through a wrapper. Guessing wrong
        produces a clean-looking trace with no states in it, so keep the raw
        shape and sort it out in the summary.
        """
        def small(x):
            if torch.is_tensor(x):
                return x.detach().float().cpu().half()
            if isinstance(x, dict):
                return {k: small(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [small(v) for v in x]
            if isinstance(x, nn.Module):
                return f"<module {type(x).__name__}>"
            return x

        self.calls.append({
            "args": [small(x) for x in a],
            "kwargs": {k: small(v) for k, v in kw.items()},
            "out": small(out),
        })


def _shape(x):
    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_shape(v) for v in x]
    if isinstance(x, (int, float)):
        return f"scalar:{x}"
    return type(x).__name__


def unwrap(rec):
    """Summarise the recorded calls so the next script can be written against
    the real signature: shapes, scalar t values, which features repeat."""
    rows = []
    for i, c in enumerate(rec.calls):
        flat = []
        for v in list(c["args"]) + list(c["kwargs"].values()):
            if torch.is_tensor(v) and v.numel() == 1:
                flat.append(float(v.flatten()[0]))
        feats = []
        for v in list(c["args"]) + list(c["kwargs"].values()):
            if isinstance(v, dict):
                for k, t in v.items():
                    feats.append(f"{k}:{_hash(t)}")
        rows.append({
            "i": i,
            "t": flat[0] if flat else None,
            "args": [_shape(v) for v in c["args"]],
            "kwargs": {k: _shape(v) for k, v in c["kwargs"].items()},
            "out": _shape(c["out"]),
            "feat": feats,
        })
    return rows


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def main():
    note("=" * 72)
    note("K2d teacher trace")
    note("=" * 72)
    note(f"device={DEV} n_images={N_IMG} steps={STEPS} gain={GAIN} shift={SHIFT}")

    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415

    files = sorted(CKPT.rglob("*.safetensors"))

    def pick(*keys):
        for f in files:
            n = f.name.lower()
            if all(k in n for k in keys):
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
        raise RuntimeError(f"no LatentSeqMMFlowModel in pipeline; saw {list(mods)}")
    note(f"flow params {sum(p.numel() for p in flow.parameters()) / 1e6:.1f} M")

    ex = REPO / "static" / "example_inputs"
    cands = sorted(p for p in ex.glob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not cands:
        cands = [p for p in sorted((REPO / "static").rglob("*"))
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    note(f"{len(cands)} candidate inputs, taking {min(N_IMG, len(cands))}")

    for idx, img_path in enumerate(cands[:N_IMG]):
        rec = Recorder(flow)
        try:
            image = Image.open(img_path).convert("RGB")
            prepared = pipe.preprocess_image(image)
            gen = torch.Generator(device=DEV).manual_seed(SEED + idx)
            cond = pipe.encode_image(prepared, generator=gen)

            torch.cuda.synchronize()
            t0 = time.time()
            gen_i = torch.Generator(device=DEV).manual_seed(SEED + idx)
            lat = pipe.sample_latent(cond, steps=STEPS, guidance_scale=GAIN,
                                     shift=SHIFT, generator=gen_i)
            torch.cuda.synchronize()
            dt = time.time() - t0

            rows = unwrap(rec)
            payload = {
                "image": img_path.name,
                "steps": STEPS,
                "guidance": GAIN,
                "shift": SHIFT,
                "seed": SEED + idx,
                "sample_s": round(dt, 2),
                "calls": rec.calls,
                "final": {k: (v.detach().float().cpu().half() if torch.is_tensor(v) else v)
                          for k, v in lat.items()},
            }
            path = OUT / f"trace_{idx:04d}.pt"
            torch.save(payload, path)
            R["images"].append({
                "index": idx, "image": img_path.name, "file": path.name,
                "mb": round(path.stat().st_size / 1e6, 1), "sample_s": round(dt, 2),
                "n_calls": len(rows),
                "t_values": sorted({round(r["t"], 4) for r in rows if r["t"] is not None}),
                "distinct_feat": len({tuple(r["feat"]) for r in rows}),
                "args_shape": rows[0]["args"] if rows else None,
                "kwargs_shape": rows[0]["kwargs"] if rows else None,
                "out_shape": rows[0]["out"] if rows else None,
            })
            note(f"[{idx}] {img_path.name}  {dt:.1f}s  calls={len(rows)}  "
                 f"branches={len({tuple(r['feat']) for r in rows})}  "
                 f"{path.stat().st_size/1e6:.1f} MB")
            note(f"    args {rows[0]['args'] if rows else None}")
            note(f"    out  {rows[0]['out'] if rows else None}")
        except Exception:  # noqa: BLE001
            note(f"[{idx}] FAILED {img_path.name}")
            note(traceback.format_exc()[-1200:])
        finally:
            rec.restore()

    # the shape of the trace decides how the student is written, so print it
    if R["images"]:
        first = R["images"][0]
        note("=" * 72)
        note(f"calls per image   {first['n_calls']}")
        note(f"distinct t         {len(first['t_values'])}  {first['t_values'][:8]}")
        note(f"cond branches      {first['distinct_feat']}")
        note("if calls == 2 * (distinct t) the sampler runs CFG as two forwards,")
        note("and the student can fold guidance into a single velocity target.")

    (OUT / "trace.json").write_text(json.dumps({k: v for k, v in R.items() if k != "calls"}, indent=2))
    note(f"wrote {OUT / 'trace.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        note(traceback.format_exc())
        raise
