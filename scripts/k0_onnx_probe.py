"""K0 - TripoSplat ONNX feasibility probe.

Canonical version. The Kaggle kernel does not contain this code; it fetches
this file from a pinned commit so that every run is reproducible and every
change has a diff.

Answers, in this order of importance:

  1. How many parameters are really in each submodule? K0 measured 1753.8 M
     total, which at INT4 is ~904 MB - far above the 600 MB the plan assumed.
     Kept here because the number should be re-derived, not remembered.
  2. Does each submodule export to ONNX, and by which path? Two paths are
     tried per module: the TorchScript exporter (dynamo=False) and the
     torch.export exporter (dynamo=True). They fail for different reasons,
     and the browser side only cares that one of them works.
  3. Is the exported graph *numerically correct*? A graph that exports but
     computes something else is worse than one that refuses to export.
  4. What does one generation cost, split by stage, on this GPU?

K0 ran the parameter and latency probes successfully but failed on three of
five exports for reasons that were my bugs, not the model's:

  - dinov3 / vae_encoder: the pipeline loads them in bf16, the probe fed fp16.
  - flow_dit: the adapter passes a dict, and torch.fx.symbolic_trace cannot
    iterate a Proxy, so the trace dies before ONNX is ever reached.

This version derives the dtype from each module's own parameters and lets the
dynamo path handle pytree inputs. Both are fixed, not worked around.
"""

import glob
import json
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k0"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shapes straight out of the source. q_token_length / in_channels /
# cam_channels come from FLOW_MODEL_ARGS; cond lengths from DINOv3 (64x64
# patches + 5 prefix tokens) and the Flux2 VAE (4096 + 5 zero tokens).
Q_TOKENS = 8192
LATENT_CH = 16
COND_TOKENS = 4101
COND_CH = 1280
COND2_CH = 128
CAM_CH = 5
GS_PER_POINT = 32
N_GAUSS = 65536
N_POINTS = N_GAUSS // GS_PER_POINT

# Ops that are known to fight ONNX, keyed by substring of the op name.
FLAGS = {
    "view_as_complex": "complex64 -> ONNX has no complex type",
    "view_as_real": "complex64",
    "polar": "complex64",
    "searchsorted": "dynamic output shape",
    "nonzero": "dynamic output shape",
    "repeat_interleave": "tensor-valued repeats",
    "unique": "dynamic output shape (and host sync)",
    "sort": "needs a shape-stable kernel",
    "topk": "ok only with static k",
    "scatter_add": "needs static shape",
    "item": "host sync -> graph break",
}

R = {"env": {}, "paths": {}, "modules": {}, "stages": {}, "notes": []}


def note(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def probe_env():
    note("=" * 72)
    note("ENV")
    note("=" * 72)
    cap = torch.cuda.get_device_capability(0) if DEV.type == "cuda" else None
    e = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "device": torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu",
        "capability": cap,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if DEV.type == "cuda" else 0,
    }
    for pkg in ("onnx", "onnxscript", "onnxruntime", "safetensors",
                "transformers", "torchvision"):
        try:
            e[pkg] = __import__(pkg).__version__
        except Exception as exc:  # noqa: BLE001
            e[pkg] = f"MISSING ({type(exc).__name__})"
    e["bf16_native"] = bool(cap and cap[0] >= 8)
    R["env"] = e
    for k, v in e.items():
        note(f"  {k:12s} {v}")


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------

def find_ckpts():
    files = sorted(CKPT.rglob("*.safetensors"))
    note(f"  found {len(files)} safetensors under {CKPT}")
    for f in files:
        note(f"    {f.relative_to(CKPT)}  {f.stat().st_size / 1e6:.1f} MB")

    def pick(*keys, exclude=()):
        for f in files:
            n = f.name.lower()
            if all(k in n for k in keys) and not any(x in n for x in exclude):
                return f
        return None

    m = {
        "dinov3_path": pick("dino"),
        "flux2_vae_encoder_path": pick("vae") or pick("flux"),
        "rmbg_path": pick("biref") or pick("rmbg") or pick("matting"),
        "decoder_path": pick("decoder") or pick("octree"),
    }
    taken = {v for v in m.values() if v}
    rest = [f for f in files if f not in taken]
    m["ckpt_path"] = rest[0] if rest else None
    R["paths"] = {k: (str(v.relative_to(CKPT)) if v else None) for k, v in m.items()}
    for k, v in R["paths"].items():
        note(f"  {k:26s} {v}")
    return {k: (str(v) if v else None) for k, v in m.items()}


# --------------------------------------------------------------------------
# params
# --------------------------------------------------------------------------

def dtype_of(m):
    """The dtype the module is actually loaded in. K0 hardcoded fp16 and
    broke on the two bf16 modules; there is no reason to guess."""
    for p in m.parameters():
        return p.dtype
    return torch.float32


def pstats(m):
    total = sum(p.numel() for p in m.parameters())
    return {
        "params": total,
        "fp16_mb": round(total * 2 / 1e6, 1),
        "int4_mb": round(total * 0.5 / 1e6, 1),
        "int4_g128_mb": round((total * 0.5 + total / 128 * 2) / 1e6, 1),
        "dtype": str(dtype_of(m)),
    }


# --------------------------------------------------------------------------
# graph capture diagnostics
# --------------------------------------------------------------------------

def fx_scan(name, mod):
    """Static trace. Purely informational - it explains *why* an export
    would fail, and it is where the complex64 question got answered."""
    rec = {"trace_ok": False}
    try:
        gm = torch.fx.symbolic_trace(mod)
    except Exception as exc:  # noqa: BLE001
        rec["trace_error"] = f"{type(exc).__name__}: {exc}"[:400]
        note(f"  [{name}] fx trace FAILED -> {rec['trace_error'][:160]}")
        return rec
    rec["trace_ok"] = True
    c = Counter()
    for n in gm.graph.nodes:
        if n.op == "call_function":
            c[str(n.target)] += 1
        elif n.op == "call_method":
            c[f"method:{n.target}"] += 1
    hits = {}
    for op, cnt in c.items():
        low = op.lower()
        for flag, why in FLAGS.items():
            if flag in low:
                hits[op] = {"count": cnt, "why": why}
    rec["nodes"] = sum(c.values())
    rec["flags"] = hits
    note(f"  [{name}] fx ok, {rec['nodes']} nodes, {len(hits)} flagged op kinds")
    for op, info in sorted(hits.items()):
        note(f"      {info['count']:5d}x {op}   ({info['why']})")
    return rec


def export_attempt(name, mod, args, tag, **kw):
    import onnx

    path = OUT / f"{name}.{tag}.onnx"
    t0 = time.time()
    try:
        torch.onnx.export(mod, tuple(args), str(path), opset_version=17,
                          do_constant_folding=True, **kw)
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        note(f"  [{name}/{tag}] FAILED in {time.time() - t0:.1f}s")
        note("      " + msg[:900].replace("\n", "\n      "))
        return {"ok": False, "err": msg[:1600]}
    secs = time.time() - t0
    try:
        mdl = onnx.load(str(path), load_external_data=False)
        ops = Counter(n.op_type for n in mdl.graph.node)
        nodes = sum(ops.values())
        top = dict(ops.most_common(10))
    except Exception as exc:  # noqa: BLE001
        ops, nodes, top = Counter(), -1, {"error": str(exc)[:200]}
    mb = path.stat().st_size / 1e6
    note(f"  [{name}/{tag}] ok  {mb:.1f} MB  {nodes} nodes  {secs:.1f}s")
    note(f"      top ops: {top}")
    rec = {"ok": True, "mb": round(mb, 1), "nodes": nodes, "top_ops": top,
           "seconds": round(secs, 1), "path": str(path),
           "complex_ops": {k: v for k, v in ops.items()
                           if "complex" in k.lower() or k in ("FFT",)}}
    if rec["complex_ops"]:
        note(f"      !! complex ops survived: {rec['complex_ops']}")
    return rec


def ort_verify(name, path, args, ref):
    """Run the exported graph and compare against eager torch.

    A silent numerical break here would poison everything downstream, so this
    is reported as a real verdict rather than a warning.
    """
    import onnxruntime as ort

    try:
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in avail]
        sess = ort.InferenceSession(path, providers=providers)
        feeds = {}
        for spec, a in zip(sess.get_inputs(), args):
            feeds[spec.name] = a.detach().cpu().numpy()
        outs = sess.run(None, feeds)
        flat_ref = [t.detach().float().cpu() for t in
                    (ref if isinstance(ref, (list, tuple)) else [ref])]
        rows = []
        for i, (o, r) in enumerate(zip(outs, flat_ref)):
            o = torch.from_numpy(o).float()
            if o.shape != r.shape:
                rows.append({"idx": i, "shape_mismatch": [list(o.shape), list(r.shape)]})
                continue
            diff = (o - r).abs()
            denom = r.abs().clamp_min(1e-6)
            cos = torch.nn.functional.cosine_similarity(
                o.flatten(), r.flatten(), dim=0).item()
            rows.append({
                "idx": i,
                "max_abs": round(diff.max().item(), 6),
                "mean_abs": round(diff.mean().item(), 6),
                "max_rel": round((diff / denom).max().item(), 4),
                "cosine": round(cos, 6),
            })
        note(f"  [{name}] ORT({providers[0].replace('ExecutionProvider','')}) "
             f"verify -> {rows}")
        return {"ok": True, "provider": providers[0], "rows": rows}
    except Exception as exc:  # noqa: BLE001
        note(f"  [{name}] ORT verify FAILED -> {type(exc).__name__}: {str(exc)[:300]}")
        return {"ok": False, "err": f"{type(exc).__name__}: {str(exc)[:500]}"}


def check_module(name, mod, args):
    """Trace, export by both paths, then verify the first one that works."""
    note(f"\n[{name}]  args={[tuple(a.shape) for a in args]}")
    st = R["modules"].setdefault(name, {})

    with torch.no_grad():
        st["fx"] = fx_scan(name, mod)

    ref = None
    try:
        with torch.no_grad():
            ref = mod(*args)
        st["eager_ok"] = True
    except Exception as exc:  # noqa: BLE001
        st["eager_ok"] = False
        st["eager_err"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        note(f"  [{name}] eager forward FAILED -> {st['eager_err'][:200]}")
        return st

    exports = {}
    for tag, kw in (("ts", {"dynamo": False}), ("dynamo", {"dynamo": True})):
        try:
            with torch.no_grad():
                exports[tag] = export_attempt(name, mod, args, tag, **kw)
        except Exception as exc:  # noqa: BLE001
            exports[tag] = {"ok": False, "err": f"{type(exc).__name__}: {str(exc)[:400]}"}
            note(f"  [{name}/{tag}] crashed -> {exports[tag]['err'][:200]}")
        finally:
            torch.cuda.empty_cache()
    st["exports"] = exports

    winner = next(((t, r) for t, r in exports.items() if r.get("ok")), None)
    if winner:
        tag, rec = winner
        st["onnx"] = rec
        st["onnx_path"] = rec["path"]
        st["onnx_via"] = tag
        with torch.no_grad():
            st["ort"] = ort_verify(name, rec["path"], args, ref)
    else:
        st["onnx"] = {"ok": False,
                      "err": "; ".join(f"{t}: {r.get('err', '')[:200]}"
                                       for t, r in exports.items())}
    torch.cuda.empty_cache()
    return st


# --------------------------------------------------------------------------
# adapters: flat tensor signatures, which is what the browser side wants
# --------------------------------------------------------------------------

class DinoAdapter(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        out = self.m(pixel_values=pixel_values)
        if isinstance(out, dict):
            return out.get("pooler_output", out.get("last_hidden_state"))
        return out


class VaeAdapter(nn.Module):
    """Deterministic encode: mean only. Noise is supplied by the caller."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        out = self.m.encode(image, deterministic=True)
        if isinstance(out, dict):
            for k in ("latent", "z", "mean", "moments"):
                if k in out:
                    return out[k]
            return list(out.values())[0]
        return out


class FlowAdapter(nn.Module):
    """The dict arguments here are the K0 failure. torch.fx cannot iterate a
    Proxy, so the TorchScript path will still die; the dynamo path treats the
    two dicts as pytree leaves and survives. That asymmetry is the finding."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, latent, camera, t, feat1, feat2):
        out = self.m({"latent": latent, "camera": camera},
                     t * 1000.0,
                     {"feature1": feat1, "feature2": feat2})
        return out["latent"], out["camera"]


class OctreeLevelAdapter(nn.Module):
    """One octree level. `sample()` loops over 8 of these - see notes."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, coords, level, cond, npoints):
        return self.m(coords, level, cond, npoints)["logits"]


class GsAdapter(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, points, cond):
        return self.m(x={"points": points}, cond=cond)["features"]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    probe_env()

    note("\n" + "=" * 72)
    note("WEIGHTS")
    note("=" * 72)
    p = find_ckpts()

    sys.path.insert(0, str(REPO))
    import model as M  # noqa: F401,PLC0415
    import triposplat as T  # noqa: PLC0415

    note("\n" + "=" * 72)
    note("BUILD")
    note("=" * 72)
    pipe = None
    try:
        pipe = T.TripoSplatPipeline(
            ckpt_path=p["ckpt_path"],
            decoder_path=p["decoder_path"],
            dinov3_path=p["dinov3_path"],
            flux2_vae_encoder_path=p["flux2_vae_encoder_path"],
            rmbg_path=p["rmbg_path"],
            device=str(DEV),
        )
        note("  pipeline constructed")
    except Exception:  # noqa: BLE001
        note("  pipeline construction FAILED:")
        note(traceback.format_exc()[-3000:])

    mods = {}
    if pipe is not None:
        for _, v in vars(pipe).items():
            if isinstance(v, nn.Module):
                mods[type(v).__name__] = v
    note(f"  modules found: {list(mods)}")
    R["module_names"] = list(mods)

    dino = mods.get("DinoV3ViT")
    vae = mods.get("Flux2VAEEncoder")
    flow = mods.get("LatentSeqMMFlowModel")
    dec = mods.get("OctreeGaussianDecoder")
    rmbg = mods.get("BiRefNet")

    note("\n" + "=" * 72)
    note("PARAMS")
    note("=" * 72)
    total = 0
    named = [("dinov3", dino), ("vae_encoder", vae), ("flow_dit", flow),
             ("octree_decoder", dec.octree if dec else None),
             ("gs_decoder", dec.gs if dec else None), ("birefnet", rmbg)]
    for nm, m in named:
        if m is None:
            continue
        st = pstats(m)
        total += st["params"]
        R["modules"].setdefault(nm, {}).update(st)
        note(f"  {nm:16s} {st['params'] / 1e6:8.1f} M   "
             f"fp16 {st['fp16_mb']:8.0f} MB   int4 {st['int4_g128_mb']:8.0f} MB   "
             f"{st['dtype']}")
    R["modules"]["TOTAL"] = {
        "params": total,
        "fp16_mb": round(total * 2 / 1e6, 1),
        "int4_g128_mb": round((total * 0.5 + total / 128 * 2) / 1e6, 1),
        "note": "birefnet included; browser side replaces it with a lighter matting model",
    }
    note(f"  {'TOTAL':16s} {total / 1e6:8.1f} M   fp16 {total * 2 / 1e6:8.0f} MB   "
         f"int4 {R['modules']['TOTAL']['int4_g128_mb']:8.0f} MB")

    # ---- one real generation, timed per stage ----------------------------
    note("\n" + "=" * 72)
    note("BASELINE RUN")
    note("=" * 72)
    stamp = {}

    def tick(name, fn):
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = fn()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        stamp[name] = round(ms, 1)
        note(f"  {name:22s} {ms:9.1f} ms")
        return out

    img_path = next(iter(sorted(glob.glob(str(REPO / "static/**/*.*"), recursive=True))), None)
    note(f"  input image: {img_path}")
    if pipe is not None and img_path:
        try:
            from PIL import Image  # noqa: PLC0415
            img = Image.open(img_path).convert("RGB")
            note(f"  image size  : {img.size}")
            prep = tick("preprocess", lambda: pipe.preprocess_image(img))
            cond = tick("encode", lambda: pipe.encode_image(prep))
            note(f"  feat1 {tuple(cond['feature1'].shape)}  "
                 f"feat2 {tuple(cond['feature2'].shape)}")
            lat = tick("sample_20step", lambda: pipe.sample_latent(
                cond, steps=20, guidance_scale=3.0, shift=3.0))
            latent = lat["latent"]
            note(f"  latent {tuple(latent.shape)}")
            tick("decode_65k", lambda: pipe.decode_latent(latent, num_gaussians=N_GAUSS))
        except Exception:  # noqa: BLE001
            note("  baseline FAILED:")
            note(traceback.format_exc()[-3000:])
    R["stages"] = stamp
    if "sample_20step" in stamp:
        per_step = stamp["sample_20step"] / 20 / 1000
        R["seconds_per_dit_step"] = round(per_step, 3)
        note(f"  -> {per_step:.3f} s per DiT step on this GPU")

    # ---- FLOPs of one denoising step -------------------------------------
    if flow is not None:
        note("\n  measuring DiT FLOPs for a single step ...")
        try:
            from torch.utils.flop_counter import FlopCounterMode  # noqa: PLC0415
            fa = FlowAdapter(flow).eval()
            dt = dtype_of(flow)
            z = torch.randn(1, Q_TOKENS, LATENT_CH, device=DEV, dtype=dt)
            cam = torch.randn(1, 1, CAM_CH, device=DEV, dtype=dt)
            f1 = torch.randn(1, COND_TOKENS, COND_CH, device=DEV, dtype=dt)
            f2 = torch.randn(1, COND_TOKENS, COND2_CH, device=DEV, dtype=dt)
            t = torch.tensor([0.5], device=DEV, dtype=dt)
            with torch.no_grad(), FlopCounterMode(display=False) as fc:
                fa(z, cam, t, f1, f2)
            gflop = fc.get_total_flops() / 1e9
            R["dit_gflop_per_step"] = round(gflop, 1)
            R["dit_tflop_20step"] = round(gflop * 20 / 1000, 2)
            note(f"  DiT {gflop:.0f} GFLOP/step -> {gflop * 20 / 1000:.2f} TFLOP for 20 steps")
            if "seconds_per_dit_step" in R and R["seconds_per_dit_step"] > 0:
                tf = gflop / 1e3 / R["seconds_per_dit_step"]
                R["tflops_achieved"] = round(tf, 2)
                note(f"  achieved {tf:.2f} TFLOPS on this card")
        except Exception:  # noqa: BLE001
            note("  FLOP count failed: " + traceback.format_exc()[-800:])

    # ---- per-module ONNX --------------------------------------------------
    note("\n" + "=" * 72)
    note("ONNX")
    note("=" * 72)

    cases = []
    if dino is not None:
        dt = dtype_of(dino)
        cases.append(("dinov3", DinoAdapter(dino).eval(),
                      [torch.randn(1, 3, 1024, 1024, device=DEV, dtype=dt)]))
    if vae is not None:
        dt = dtype_of(vae)
        cases.append(("vae_encoder", VaeAdapter(vae).eval(),
                      [torch.randn(1, 3, 1024, 1024, device=DEV, dtype=dt)]))
    if flow is not None:
        dt = dtype_of(flow)
        cases.append(("flow_dit", FlowAdapter(flow).eval(), [
            torch.randn(1, Q_TOKENS, LATENT_CH, device=DEV, dtype=dt),
            torch.randn(1, 1, CAM_CH, device=DEV, dtype=dt),
            torch.tensor([0.5], device=DEV, dtype=dt),
            torch.randn(1, COND_TOKENS, COND_CH, device=DEV, dtype=dt),
            torch.randn(1, COND_TOKENS, COND2_CH, device=DEV, dtype=dt),
        ]))
    if dec is not None:
        dt = dtype_of(dec.octree)
        cases.append(("octree_level", OctreeLevelAdapter(dec.octree).eval(), [
            torch.rand(1, N_POINTS, 3, device=DEV, dtype=dt) * 0.5,
            torch.full((1,), 8.0, device=DEV, dtype=dt),
            torch.randn(1, Q_TOKENS, LATENT_CH, device=DEV, dtype=dt),
            torch.full((1,), float(N_POINTS), device=DEV, dtype=dt),
        ]))
        dt = dtype_of(dec.gs)
        cases.append(("gs_decoder", GsAdapter(dec.gs).eval(), [
            torch.rand(1, N_POINTS, 3, device=DEV, dtype=dt) * 0.5,
            torch.randn(1, Q_TOKENS, LATENT_CH, device=DEV, dtype=dt),
        ]))

    for name, mod, args in cases:
        try:
            check_module(name, mod, args)
        except Exception:  # noqa: BLE001
            note(f"  [{name}] probe crashed: " + traceback.format_exc()[-600:])
        finally:
            torch.cuda.empty_cache()

    # ---- the octree sampler, decomposed ----------------------------------
    note("\n" + "=" * 72)
    note("OCTREE SAMPLE DECOMPOSITION")
    note("=" * 72)
    src = (REPO / "model.py").read_text()
    for token in ("mask.sum(dim=1).max().item()", "torch.searchsorted", "nonzero(",
                  "repeat_interleave", "sample_probs", "torch.rand_like",
                  ".item()"):
        R.setdefault("octree_src_hits", {})[token] = src.count(token)
        note(f"  {token:34s} x{src.count(token)}")
    R["notes"].append(
        "sample() is a python loop over 8 levels; each level calls the same "
        "fixed-shape transformer, so the graph is exportable per level and the "
        "loop plus systematic sampling belong on the host / in WGSL. "
        "max_valid = mask.sum().max().item() is the only true sync point.")

    # ---- download budget from real file sizes -----------------------------
    total_mb = sum(m.get("onnx", {}).get("mb", 0) for m in R["modules"].values()
                   if isinstance(m, dict))
    R["onnx_total_mb"] = round(total_mb, 1)
    note(f"\n  total ONNX (fp16/bf16, unquantized): {total_mb:.0f} MB")

    (OUT / "report.json").write_text(json.dumps(R, indent=2))
    note(f"wrote {OUT / 'report.json'}")

    # ---- compact summary --------------------------------------------------
    note("\n" + "=" * 72)
    note("SUMMARY")
    note("=" * 72)
    for nm, st in R["modules"].items():
        if nm == "TOTAL" or not isinstance(st, dict):
            continue
        o = st.get("onnx") or {}
        ort_r = (st.get("ort") or {})
        verdict = "OK" if o.get("ok") else ("FAIL" if st.get("eager_ok") else "EAGER-FAIL")
        via = st.get("onnx_via", "-")
        numeric = "-"
        if ort_r.get("rows"):
            r0 = ort_r["rows"][0]
            numeric = f"cos={r0.get('cosine')}"
        note(f"  {nm:14s} {verdict:10s} via={via:6s} int4={st.get('int4_g128_mb', 0):6.0f}MB "
             f"{numeric:16s} {(o.get('err') or '')[:70]}")
    note(f"  stages: {R['stages']}")
    note(f"  per DiT step: {R.get('seconds_per_dit_step')} s")


if __name__ == "__main__":
    main()
