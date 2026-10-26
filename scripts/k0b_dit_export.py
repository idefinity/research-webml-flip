"""K0b - the two questions K0 could not answer, in isolation.

K0 left exactly two things open, and both were instrumentation bugs rather
than model properties:

  1. flow_dit reported eager_ok=False with a *Proxy* error. That is not a
     fact about the DiT - the pipeline runs that same module twenty times
     during sample_latent and it works. Calling symbolic_trace on the wrapper
     first, and letting it throw, leaves the module in a state that breaks
     the next eager call. The probe poisoned its own reference. Here we never
     touch symbolic_trace.

  2. dinov3 and vae_encoder export, but onnxruntime refuses to load the
     graphs: `Type 'tensor(bfloat16)' of input parameter ... is an invalid
     model`. bf16 has no ONNX tensor type that ORT accepts, and it has no
     WebGPU storage format that a browser can read either. So the encoder
     has to be *cast to fp16 before export*. This measures whether that is
     cheap (it should be) and whether the result still matches.

Both answers land in k0b/report.json.
"""

import glob
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k0b"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q_TOKENS = 8192
LATENT_CH = 16
COND_TOKENS = 4101
COND_CH = 1280
COND2_CH = 128
CAM_CH = 5

R = {"env": {}, "dit": {}, "dinov3": {}, "notes": []}


def note(msg=""):
    print(msg, flush=True)


def dtype_of(m):
    for p in m.parameters():
        return p.dtype
    return torch.float32


class FlowAdapter(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, latent, camera, t, feat1, feat2):
        out = self.m({"latent": latent, "camera": camera},
                     t * 1000.0,
                     {"feature1": feat1, "feature2": feat2})
        return out["latent"], out["camera"]


class DinoAdapter(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        out = self.m(pixel_values=pixel_values)
        if isinstance(out, dict):
            return out.get("pooler_output", out.get("last_hidden_state"))
        return out


def ort_check(name, path, args, ref, time_it=False):
    import onnxruntime as ort

    try:
        avail = ort.get_available_providers()
        provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        t0 = time.time()
        sess = ort.InferenceSession(path, providers=provs)
        load_s = time.time() - t0
        feeds = {}
        for spec, a in zip(sess.get_inputs(), args):
            feeds[spec.name] = a.detach().cpu().numpy()
        outs = sess.run(None, feeds)
        rows = []
        refs = ref if isinstance(ref, (list, tuple)) else [ref]
        for i, (o, r) in enumerate(zip(outs, refs)):
            o = torch.from_numpy(o).float()
            r = r.detach().float().cpu()
            if o.shape != r.shape:
                rows.append({"idx": i, "shape": [list(o.shape), list(r.shape)]})
                continue
            rows.append({
                "idx": i,
                "max_abs": round((o - r).abs().max().item(), 6),
                "mean_abs": round((o - r).abs().mean().item(), 6),
                "cosine": round(torch.nn.functional.cosine_similarity(
                    o.flatten(), r.flatten(), dim=0).item(), 6),
            })
        lat = None
        if time_it:
            for _ in range(2):
                sess.run(None, feeds)
            torch.cuda.synchronize()
            t0 = time.time()
            n = 5
            for _ in range(n):
                sess.run(None, feeds)
            torch.cuda.synchronize()
            lat = round((time.time() - t0) / n * 1000, 1)
            note(f"  [{name}] ORT latency {lat} ms/run")
        note(f"  [{name}] ORT {provs[0].replace('ExecutionProvider','')} ok "
             f"load={load_s:.1f}s rows={rows}")
        return {"ok": True, "provider": provs[0], "load_s": round(load_s, 2),
                "rows": rows, "latency_ms": lat}
    except Exception as exc:  # noqa: BLE001
        note(f"  [{name}] ORT FAILED -> {type(exc).__name__}: {str(exc)[:300]}")
        return {"ok": False, "err": f"{type(exc).__name__}: {str(exc)[:600]}"}


# ---------------------------------------------------------------------------
# 1. the DiT
# ---------------------------------------------------------------------------

def probe_dit(flow):
    note("=" * 72)
    note("FLOW DIT")
    note("=" * 72)
    dt = dtype_of(flow)
    note(f"  dtype {dt}")

    fa = FlowAdapter(flow).eval()
    args = [
        torch.randn(1, Q_TOKENS, LATENT_CH, device=DEV, dtype=dt),
        torch.randn(1, 1, CAM_CH, device=DEV, dtype=dt),
        torch.tensor([0.5], device=DEV, dtype=dt),
        torch.randn(1, COND_TOKENS, COND_CH, device=DEV, dtype=dt),
        torch.randn(1, COND_TOKENS, COND2_CH, device=DEV, dtype=dt),
    ]

    # 1a. eager, with nothing traced before it
    ref = None
    try:
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            ref = fa(*args)
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        R["dit"]["eager_ok"] = True
        R["dit"]["eager_ms"] = round(ms, 1)
        note(f"  eager ok in {ms:.0f} ms -> "
             f"{[tuple(r.shape) for r in ref]}")
    except Exception:  # noqa: BLE001
        R["dit"]["eager_ok"] = False
        R["dit"]["eager_err"] = traceback.format_exc()[-1500:]
        note("  eager FAILED")
        note(traceback.format_exc()[-2000:])
        return

    # 1b. torch.export capture - the honest question: can this graph be captured
    try:
        import torch.export  # noqa: F401,PLC0415
        t0 = time.time()
        ep = torch.export.export(fa, tuple(args))
        secs = time.time() - t0
        nodes = len(list(ep.graph.nodes))
        R["dit"]["export_capture_ok"] = True
        R["dit"]["export_nodes"] = nodes
        R["dit"]["export_seconds"] = round(secs, 1)
        note(f"  torch.export capture ok  {nodes} nodes  {secs:.1f}s")
        del ep
    except Exception:  # noqa: BLE001
        R["dit"]["export_capture_ok"] = False
        R["dit"]["export_capture_err"] = traceback.format_exc()[-2000:]
        note("  torch.export capture FAILED")
        note(traceback.format_exc()[-2500:])
    torch.cuda.empty_cache()

    # 1c. ONNX, dynamo first (no symbolic_trace involved)
    for tag, kw in (("dynamo", {"dynamo": True}), ("ts", {"dynamo": False})):
        path = OUT / f"flow_dit.{tag}.onnx"
        note(f"  exporting via {tag} ...")
        t0 = time.time()
        try:
            with torch.no_grad():
                torch.onnx.export(fa, tuple(args), str(path), opset_version=17,
                                  do_constant_folding=True, **kw)
        except Exception:  # noqa: BLE001
            R["dit"][f"onnx_{tag}_ok"] = False
            R["dit"][f"onnx_{tag}_err"] = traceback.format_exc()[-1800:]
            note(f"  {tag} FAILED in {time.time() - t0:.1f}s")
            note(traceback.format_exc()[-2000:])
            torch.cuda.empty_cache()
            continue
        secs = time.time() - t0
        mb = path.stat().st_size / 1e6
        ext = list(OUT.glob(f"flow_dit.{tag}.onnx.data"))
        ext_mb = sum(p.stat().st_size for p in ext) / 1e6
        R["dit"][f"onnx_{tag}_ok"] = True
        R["dit"][f"onnx_{tag}_mb"] = round(mb, 1)
        R["dit"][f"onnx_{tag}_data_mb"] = round(ext_mb, 1)
        R["dit"][f"onnx_{tag}_seconds"] = round(secs, 1)
        note(f"  {tag} ok  {mb:.1f} MB (+{ext_mb:.0f} MB external data)  {secs:.1f}s")
        with torch.no_grad():
            R["dit"][f"ort_{tag}"] = ort_check(f"flow_dit/{tag}", str(path),
                                               args, ref, time_it=True)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 2. dinov3 in fp16
# ---------------------------------------------------------------------------

def probe_dino_fp16(dino):
    note("\n" + "=" * 72)
    note("DINOV3 -> fp16")
    note("=" * 72)
    note(f"  loaded as {dtype_of(dino)}")

    args_bf16 = [torch.randn(1, 3, 1024, 1024, device=DEV, dtype=dtype_of(dino))]
    ref = None
    try:
        with torch.no_grad():
            ref = DinoAdapter(dino).eval()(*args_bf16)
        note(f"  eager ok -> {tuple(ref.shape)}")
    except Exception:  # noqa: BLE001
        note("  eager FAILED")
        note(traceback.format_exc()[-1500:])
        R["dinov3"]["eager_err"] = traceback.format_exc()[-1200:]
        return

    # dynamic cast on a copy would double 1.7 GB; cast in place and keep a
    # small sample of outputs to compare against
    R["dinov3"]["orig_dtype"] = str(dtype_of(dino))
    try:
        dino.half()
        torch.cuda.empty_cache()
        note(f"  cast to {dtype_of(dino)}")
    except Exception:  # noqa: BLE001
        note("  cast FAILED")
        note(traceback.format_exc()[-800:])
        return

    args_fp16 = [a.half() for a in args_bf16]
    ref16 = None
    try:
        with torch.no_grad():
            ref16 = DinoAdapter(dino).eval()(*args_fp16)
        diff = (ref16.float() - ref.float()).abs()
        cos = torch.nn.functional.cosine_similarity(
            ref16.float().flatten(), ref.float().flatten(), dim=0).item()
        R["dinov3"]["fp16_vs_bf16"] = {
            "max_abs": round(diff.max().item(), 5),
            "mean_abs": round(diff.mean().item(), 6),
            "cosine": round(cos, 6),
        }
        note(f"  fp16 vs bf16: max_abs={diff.max().item():.5f} "
             f"mean_abs={diff.mean().item():.6f} cos={cos:.6f}")
    except Exception:  # noqa: BLE001
        note("  fp16 eager FAILED")
        note(traceback.format_exc()[-1200:])
        return

    for tag, kw in (("dynamo", {"dynamo": True}), ("ts", {"dynamo": False})):
        path = OUT / f"dinov3_fp16.{tag}.onnx"
        note(f"  exporting {tag} ...")
        t0 = time.time()
        try:
            with torch.no_grad():
                torch.onnx.export(DinoAdapter(dino).eval(), tuple(args_fp16),
                                  str(path), opset_version=17,
                                  do_constant_folding=True, **kw)
        except Exception:  # noqa: BLE001
            R["dinov3"][f"onnx_{tag}_ok"] = False
            R["dinov3"][f"onnx_{tag}_err"] = traceback.format_exc()[-1500:]
            note(f"  {tag} FAILED {time.time() - t0:.1f}s")
            note(traceback.format_exc()[-1500:])
            continue
        mb = path.stat().st_size / 1e6
        ext = sum(p.stat().st_size for p in OUT.glob(f"dinov3_fp16.{tag}.onnx.data")) / 1e6
        R["dinov3"][f"onnx_{tag}_ok"] = True
        R["dinov3"][f"onnx_{tag}_mb"] = round(mb + ext, 1)
        note(f"  {tag} ok  {mb:.1f} MB (+{ext:.0f} MB ext)  {time.time() - t0:.1f}s")
        with torch.no_grad():
            R["dinov3"][f"ort_{tag}"] = ort_check(
                f"dinov3/{tag}", str(path), args_fp16, ref16)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    note("=" * 72)
    note("ENV")
    note("=" * 72)
    R["env"] = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu",
        "capability": torch.cuda.get_device_capability(0) if DEV.type == "cuda" else None,
    }
    for k, v in R["env"].items():
        note(f"  {k:12s} {v}")

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
    vae_p = pick("vae") or pick("flux")
    rmbg_p = pick("biref") or pick("rmbg")
    dec_p = pick("decoder") or pick("octree")
    taken = {p for p in (dino_p, vae_p, rmbg_p, dec_p) if p}
    ck = [f for f in files if f not in taken][0]

    note("\nbuilding pipeline ...")
    pipe = T.TripoSplatPipeline(
        ckpt_path=str(ck), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p),
        device=str(DEV))
    mods = {type(v).__name__: v for v in vars(pipe).values() if isinstance(v, nn.Module)}
    note(f"  modules: {list(mods)}")

    flow = mods.get("LatentSeqMMFlowModel")
    dino = mods.get("DinoV3ViT")

    if flow is not None:
        probe_dit(flow)
    if dino is not None:
        probe_dino_fp16(dino)

    (OUT / "report.json").write_text(json.dumps(R, indent=2))
    note(f"\nwrote {OUT / 'report.json'}")

    note("\n" + "=" * 72)
    note("SUMMARY")
    note("=" * 72)
    d = R["dit"]
    note(f"  dit eager      {d.get('eager_ok')} {d.get('eager_ms')} ms")
    note(f"  dit capture    {d.get('export_capture_ok')} nodes={d.get('export_nodes')}")
    for tag in ("dynamo", "ts"):
        note(f"  dit onnx/{tag:6s} {d.get(f'onnx_{tag}_ok')} "
             f"{d.get(f'onnx_{tag}_mb')} MB "
             f"ort={(d.get(f'ort_{tag}') or {}).get('ok')} "
             f"lat={(d.get(f'ort_{tag}') or {}).get('latency_ms')} ms")
    v = R["dinov3"]
    note(f"  dino fp16 vs bf16 {v.get('fp16_vs_bf16')}")
    for tag in ("dynamo", "ts"):
        note(f"  dino onnx/{tag:6s} {v.get(f'onnx_{tag}_ok')} {v.get(f'onnx_{tag}_mb')} MB "
             f"ort={(v.get(f'ort_{tag}') or {}).get('ok')}")


if __name__ == "__main__":
    main()
