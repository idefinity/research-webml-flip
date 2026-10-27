"""K2i - does the DiT survive fp16, and does the fp16 graph export?

fp16 is the second of the two levers that can get us inside thirty seconds
(the first is the step count). Both of the numbers behind it were measured in
isolation and then extrapolated:

  - 2.16x came from bench_gemm_f16.html, a 2048x2048 GEMM. It says the *ALU*
    is twice as fast in fp16. It says nothing about whether this network is
    still correct in fp16.
  - K0e measured a residual peak of 33603 on the DiT and called it "does not
    overflow". True, and not the question: fp16 maxes at 65504, so 33603 fits,
    but at that magnitude the spacing between representable numbers is 32.
    Fitting is not surviving.

So this measures the thing that decides: run the DiT in fp32 and in fp16 on
the *real* sampler state and compare. Not randn - the error depends entirely
on where the activations actually sit, and randn at t=0.5 is a place the
sampler never visits.

Real states are obtained by running the actual sampler and recording what it
feeds the flow model, then bailing out after four calls. Two snapshots are
kept: the first (pure noise, t near 1) and a late one (near data, t near 0),
because a model can be fine at one and wrong at the other.

The model under test is whatever is loaded. Point FLIP_STUDENT at the distilled
checkpoint and this gates the model that would actually ship; leave it unset
and it gates the teacher.

If fp16 fails the gate, the fallback is the K0f fix - normalisation statistics
computed in fp32 - which is what took dinov3 from cosine 0.075 to 1.0003.
FLIP_PATCH_NORM=1 applies it and re-measures, and the report records how many
norm layers were swapped and which norm classes exist, because a fix that
swapped zero layers would look like a fix that did nothing.

Env:
  FLIP_STUDENT   checkpoint to load into flow_model (default: none, teacher)
  FLIP_PATCH_NORM  1 = swap nn.LayerNorm for the fp32-stats version first
  FLIP_STEPS / FLIP_SHIFT / FLIP_GUIDANCE   schedule used to capture states
  FLIP_SNAP     how many DiT calls to record before bailing out (default 4)
  FLIP_EXPORT   0 = skip the ONNX step, just measure the cosine
"""

import json
import os
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k2i"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STUDENT = Path(os.environ.get("FLIP_STUDENT", "").strip()) if os.environ.get(
    "FLIP_STUDENT", "").strip() else None
PATCH_NORM = os.environ.get("FLIP_PATCH_NORM", "0") == "1"
N_SNAP = int(os.environ.get("FLIP_SNAP", "4"))
DO_EXPORT = os.environ.get("FLIP_EXPORT", "1") == "1"
STEPS = int(os.environ.get("FLIP_STEPS", "4"))
SHIFT = float(os.environ.get("FLIP_SHIFT", "1.0"))
GUIDANCE = float(os.environ.get("FLIP_GUIDANCE", "1.0"))
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")

R = {"notes": [], "model": {}, "norms": {}, "snapshots": [], "export": {}}


def note(msg=""):
    print(msg, flush=True)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=2))


# ---------------------------------------------------------------------------
# the K0f fix, reused
# ---------------------------------------------------------------------------

class LayerNormFP32Stats(nn.Module):
    """nn.LayerNorm with the reduction in fp32 and the storage untouched.

    (x - mu) ** 2 rather than x*x - mu*mu: the second form cancels badly once
    the mean dwarfs the variance, which is exactly the regime K0f measured -
    per-token mean(x^2) up to 1.4e6 while the residual sits at 3e4.
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        mu = xf.mean(-1, keepdim=True)
        var = (xf - mu).pow(2).mean(-1, keepdim=True)
        y = (xf - mu) * torch.rsqrt(var + self.eps)
        y = y * self.weight.float() + self.bias.float()
        return y.to(x.dtype)


def swap_layernorms(model):
    n = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.LayerNorm):
                dim = child.normalized_shape[0]
                new = LayerNormFP32Stats(dim, child.eps).to(
                    device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    new.weight.copy_(child.weight)
                    new.bias.copy_(child.bias)
                setattr(parent, name, new)
                n += 1
    return n


def norm_census(model):
    """Which norm classes are in here. A swap that finds zero is a no-op, and
    saying so out loud is cheaper than debugging it later."""
    c = Counter()
    for name, m in model.named_modules():
        if "norm" in name.lower() or "norm" in type(m).__name__.lower():
            c[type(m).__name__] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

class FlowAdapter(nn.Module):
    """Same wrapper K0b exported from, so the graph is comparable."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, latent, camera, t, feat1, feat2):
        out = self.m({"latent": latent, "camera": camera},
                     t * 1000.0,
                     {"feature1": feat1, "feature2": feat2})
        return out["latent"], out["camera"]


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


class _Bail(Exception):
    pass


def capture_states(pipe, cond):
    """Run the real sampler and record what it actually feeds the DiT.

    Bails out after N_SNAP calls by raising through the sampler: capturing is
    only worth doing for a couple of steps, and a 20-step CFG run would spend
    8 minutes producing forty states we do not need.
    """
    cls = type(pipe.flow_model)
    orig = cls.forward
    seen = []

    def spy(self, state, t, condf, *a, **k):
        snap = {"latent": state["latent"].detach().clone(),
                "camera": state["camera"].detach().clone()}
        snap["t"] = t.detach().clone() if torch.is_tensor(t) else torch.tensor(
            float(t), device=DEV)
        for key in condf:
            snap[f"cond_{key}"] = condf[key].detach().clone()
        seen.append(snap)
        if len(seen) >= N_SNAP:
            raise _Bail()
        return orig(self, state, t, condf, *a, **k)

    cls.forward = spy
    try:
        pipe.sample_latent(cond, steps=STEPS, guidance_scale=GUIDANCE,
                           shift=SHIFT)
    except _Bail:
        pass
    finally:
        cls.forward = orig
    return seen


def pick_image():
    root = os.environ.get("FLIP_IMAGE_DIR", "").strip()
    if root:
        p = Path(root)
        found = ([p] if p.is_file() else
                 sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT))
        if found:
            return found[0]
    ex = REPO / "static" / "example_inputs"
    cands = sorted(q for q in ex.glob("*") if q.suffix.lower() in IMG_EXT)
    return cands[0] if cands else None


def tensors_for(snap, dt):
    lat = snap["latent"].to(dt)
    cam = snap["camera"].to(dt)
    t = snap["t"].to(dt).reshape(-1)
    return [lat, cam, t,
            snap["cond_feature1"].to(dt), snap["cond_feature2"].to(dt)]


def compare(a, b):
    """cosine and rel-L2 per output channel, flattened."""
    out = []
    for x, y in zip(a, b):
        # cpu() because one side is often a numpy round-trip and the other is
        # still on cuda; subtracting across devices is a crash, not a number
        x = x.detach().float().cpu().flatten()
        y = y.detach().float().cpu().flatten()
        out.append({
            "cos": round(torch.nn.functional.cosine_similarity(
                x, y, dim=0).item(), 6),
            "rel_l2": round(((x - y).norm() / y.norm()).item(), 6),
            "max_abs": round((x - y).abs().max().item(), 4),
        })
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    note("=" * 72)
    note("K2i fp16 gate")
    note("=" * 72)

    pipe = build_pipeline()
    flow = pipe.flow_model
    R["model"]["params"] = sum(p.numel() for p in flow.parameters())
    note(f"flow_model {type(flow).__name__}  {R['model']['params']/1e6:.1f} M")

    if STUDENT is not None and STUDENT.exists():
        blob = torch.load(str(STUDENT), map_location="cpu")
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        flow.load_state_dict(sd)
        R["model"]["source"] = f"student {STUDENT.name}"
        note(f"loaded student {STUDENT.name} ({len(sd)} tensors)")
    else:
        R["model"]["source"] = "teacher"
        note("no student checkpoint - gating the teacher")

    R["norms"]["census_before"] = norm_census(flow)
    note(f"norm census {R['norms']['census_before']}")
    if PATCH_NORM:
        n = swap_layernorms(flow)
        R["norms"]["layer_norm_swapped"] = n
        R["norms"]["census_after"] = norm_census(flow)
        note(f"patched {n} nn.LayerNorm -> fp32 stats; "
             f"census now {R['norms']['census_after']}")
        if n == 0:
            note("  WARNING: zero layers swapped. Whatever normalises this "
                 "model is not nn.LayerNorm, so this patch did nothing.")

    # ---- real states ------------------------------------------------------
    img = pick_image()
    if img is None:
        raise RuntimeError("no input image found")
    note(f"input {Path(img).name}")
    prepared = pipe.preprocess_image(Image.open(img).convert("RGB"))
    t0 = time.time()
    cond = pipe.encode_image(prepared)
    note(f"encode {time.time() - t0:.1f}s")

    seen = capture_states(pipe, cond)
    note(f"captured {len(seen)} DiT calls")
    if not seen:
        raise RuntimeError("sampler never called the flow model")
    # first and last: pure noise vs nearly denoised. A model can pass one and
    # fail the other, and only the late one predicts the picture.
    picks = [seen[0], seen[-1]] if len(seen) > 1 else [seen[0]]

    # ---- fp32 vs fp16 -----------------------------------------------------
    adapter = FlowAdapter(flow).eval()
    for i, snap in enumerate(picks):
        rec = {"which": "first" if i == 0 else "last",
               "t": round(float(snap["t"].flatten()[0]), 4)}
        with torch.no_grad():
            a32 = [x.float() for x in tensors_for(snap, torch.float32)]
            adapter.float()
            torch.cuda.synchronize()
            t0 = time.time()
            r32 = adapter(*a32)
            torch.cuda.synchronize()
            rec["fp32_ms"] = round((time.time() - t0) * 1000, 1)

            a16 = [x.half() for x in tensors_for(snap, torch.float16)]
            adapter.half()
            torch.cuda.synchronize()
            t0 = time.time()
            r16 = adapter(*a16)
            torch.cuda.synchronize()
            rec["fp16_ms"] = round((time.time() - t0) * 1000, 1)

        rec["speedup"] = round(rec["fp32_ms"] / max(rec["fp16_ms"], 1e-6), 3)
        rec["channels"] = compare(r16, r32)
        worst = min(c["cos"] for c in rec["channels"])
        rec["worst_cos"] = worst
        rec["passes"] = worst >= 0.999
        R["snapshots"].append(rec)
        note(f"  {rec['which']:>5} t={rec['t']:<8} "
             f"fp32 {rec['fp32_ms']}ms  fp16 {rec['fp16_ms']}ms  "
             f"{rec['speedup']}x  worst cos {worst:.6f}  "
             f"{'PASS' if rec['passes'] else 'FAIL'}")
        for c in rec["channels"]:
            note(f"          cos {c['cos']:.6f}  rel-l2 {c['rel_l2']:.5f}  "
                 f"max|d| {c['max_abs']}")
        save()
        del r32, r16
        torch.cuda.empty_cache()

    R["fp16_ok"] = all(s["passes"] for s in R["snapshots"])
    note(f"fp16 gate: {'PASS' if R['fp16_ok'] else 'FAIL'}")
    save()

    if not DO_EXPORT:
        return

    # ---- export -----------------------------------------------------------
    adapter.half()
    args16 = tensors_for(picks[0], torch.float16)
    with torch.no_grad():
        ref16 = adapter(*[a.clone() for a in args16])
    path = OUT / "flow_dit.fp16.onnx"
    note(f"exporting fp16 -> {path.name}")
    t0 = time.time()
    try:
        with torch.no_grad():
            torch.onnx.export(adapter, tuple(args16), str(path),
                              do_constant_folding=True, dynamo=True,
                              opset_version=18)
    except Exception:  # noqa: BLE001
        R["export"]["ok"] = False
        R["export"]["err"] = traceback.format_exc()[-1800:]
        note("export FAILED")
        note(traceback.format_exc()[-2000:])
        save()
        return
    secs = time.time() - t0
    mb = path.stat().st_size / 1e6
    ext_mb = sum(p.stat().st_size for p in
                 OUT.glob("flow_dit.fp16.onnx.data")) / 1e6
    R["export"] = {"ok": True, "seconds": round(secs, 1),
                   "mb": round(mb, 1), "data_mb": round(ext_mb, 1)}
    note(f"  {mb:.1f} MB (+{ext_mb:.0f} MB external) in {secs:.1f}s")

    try:
        import onnx  # noqa: PLC0415
        m = onnx.load(str(path), load_external_data=False)
        R["export"]["opset"] = m.opset_import[0].version
        casts = sum(1 for n in m.graph.node if n.op_type == "Cast")
        to_f32 = sum(1 for n in m.graph.node if n.op_type == "Cast" and any(
            a.i == 1 for a in n.attribute if a.name == "to"))
        R["export"]["cast_nodes"] = casts
        R["export"]["cast_to_float"] = to_f32
        note(f"  opset {R['export']['opset']}  {casts} Cast "
             f"({to_f32} -> float)")
    except Exception as exc:  # noqa: BLE001
        note(f"  onnx inspect failed: {exc}")

    # ---- ORT --------------------------------------------------------------
    try:
        import onnxruntime as ort  # noqa: PLC0415
        avail = ort.get_available_providers()
        provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in avail]
        if not provs or provs[0] == "CPUExecutionProvider":
            note(f"  skip ORT verify: {avail}")
            R["export"]["ort"] = {"ok": False, "skipped": True, "providers": avail}
            save()
            return
        sess = ort.InferenceSession(str(path), providers=provs)
        feeds = {s.name: a.detach().cpu().numpy()
                 for s, a in zip(sess.get_inputs(), args16)}
        outs = sess.run(None, feeds)
        for _ in range(2):
            sess.run(None, feeds)
        torch.cuda.synchronize()
        t0 = time.time()
        n = 5
        for _ in range(n):
            sess.run(None, feeds)
        torch.cuda.synchronize()
        lat = (time.time() - t0) / n * 1000
        rows = compare([torch.from_numpy(o) for o in outs], ref16)
        R["export"]["ort"] = {"ok": True, "provider": provs[0],
                              "latency_ms": round(lat, 1), "rows": rows}
        note(f"  ORT {provs[0].replace('ExecutionProvider','')} "
             f"{lat:.0f} ms/run  " +
             "  ".join(f"cos {r['cos']:.6f}" for r in rows))
    except Exception as exc:  # noqa: BLE001
        R["export"]["ort"] = {"ok": False,
                              "err": f"{type(exc).__name__}: {str(exc)[:600]}"}
        note(f"  ORT FAILED {type(exc).__name__}: {str(exc)[:300]}")
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        note(traceback.format_exc())
        save()
        raise
