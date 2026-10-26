"""K0d - is the dinov3 mismatch precision or structure?

K0c narrowed it down and killed two hypotheses on the way:

  * the architecture exports faithfully. A 2-layer random-init DinoV3ViT, same
    code path including the prefix/patch RoPE split, matches eager to cosine
    1.0000 at every rung of the ladder, through both export paths, in fp32.
  * it is not a permutation. The exported graph's output is not the same
    multiset of values in a different order (sorted_values_match = false).

What is left is a very specific fingerprint on the real 32-layer fp16 model:
prefix tokens 0, 1, 3, 4 match to 0.99999, register token 2 is -0.04, and all
4096 patch tokens sit at 0.067. Two export paths, one number to five decimals.

Note what 0.067 might be. ViT patch tokens are close to mutually orthogonal,
so ~0.07 is what an unrelated token looks like. That is consistent with two
very different stories - the patches are being computed from the wrong thing,
or the comparison is misaligned - and the tiny model cannot distinguish them
because it is neither fp16 nor 32 layers deep.

So this run separates the two variables the tiny model could not:

  1. real weights, real size, **fp32**. If fp32 matches, the defect is fp16
     arithmetic and the fix is a mixed-precision policy, not an export hack.
     If fp32 also fails, it is structural at scale and fp16 is innocent.
  2. a nearest-neighbour test. For each patch token in the exported output,
     which eager token is it actually closest to? Identity everywhere means
     the values are wrong; a shifted or scrambled index map means the layout
     is wrong. The two have completely different fixes.

Also measures the pairwise cosine between distinct eager patch tokens, which
calibrates what "unrelated" scores in this model - otherwise 0.067 is a
number without a scale.

Writes k0d/report.json plus the raw tensors as .npy for offline analysis.
"""

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k0d"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = {}


def note(msg=""):
    print(msg, flush=True)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=2))


def analyse(tag, out, ref, n_pref=5):
    """Cosine overall, per token group, plus where each token actually lands."""
    a = torch.from_numpy(out).float() if isinstance(out, np.ndarray) else out.float()
    b = ref.float()
    a, b = a.cpu(), b.cpu()
    rec = {
        "flat_cosine": round(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(), 6),
        "max_abs": round((a - b).abs().max().item(), 6),
        "mean_abs": round((a - b).abs().mean().item(), 6),
    }
    if a.dim() != 3:
        note(f"  [{tag}] {rec}")
        return rec

    per_tok = F.cosine_similarity(a[0], b[0], dim=-1)
    rec["cos_prefix"] = [round(v, 6) for v in per_tok[:n_pref].tolist()]
    rec["cos_patch_mean"] = round(per_tok[n_pref:].mean().item(), 6)

    # calibration: what do two unrelated patch tokens score against each other?
    bp = F.normalize(b[0, n_pref:], dim=-1)
    sim = bp @ bp.T
    n = sim.shape[0]
    off = (sim.sum() - sim.diagonal().sum()) / (n * n - n)
    rec["eager_patch_pairwise_cos_mean"] = round(off.item(), 6)

    # where does each exported patch token actually come from?
    ap = F.normalize(a[0, n_pref:], dim=-1)
    nn = (ap @ bp.T).argmax(dim=1)
    ident = (nn == torch.arange(n)).float().mean().item()
    rec["nn_identity_fraction"] = round(ident, 6)
    if ident < 0.99:
        off_by = (nn - torch.arange(n)).float()
        vals, counts = off_by.unique(return_counts=True)
        top = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:5]
        rec["nn_offset_top"] = [(int(v), int(c)) for c, v in top]
        rec["nn_sample"] = [int(x) for x in nn[:16].tolist()]
        note(f"  [{tag}] NN identity {ident:.3f}; common offsets (offset,count) {rec['nn_offset_top']}")
    note(f"  [{tag}] cos={rec['flat_cosine']:.4f} prefix={rec['cos_prefix']} "
         f"patch={rec['cos_patch_mean']} (unrelated≈{rec['eager_patch_pairwise_cos_mean']})")
    return rec


def export_and_run(name, mod, args):
    import onnxruntime as ort

    path = OUT / f"{name}.onnx"
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(mod, tuple(args), str(path), do_constant_folding=True,
                          dynamo=True, opset_version=18)
    note(f"  [{name}] exported in {time.time() - t0:.1f}s "
         f"({path.stat().st_size / 1e6:.0f} MB + "
         f"{sum(p.stat().st_size for p in OUT.glob(name + '.onnx.data')) / 1e6:.0f} MB)")
    torch.cuda.empty_cache()
    provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
             if p in ort.get_available_providers()]
    t0 = time.time()
    sess = ort.InferenceSession(str(path), providers=provs)
    note(f"  [{name}] ORT {provs[0].replace('ExecutionProvider', '')} "
         f"loaded in {time.time() - t0:.1f}s")
    feeds = {s.name: a.detach().cpu().numpy() for s, a in zip(sess.get_inputs(), args)}
    t0 = time.time()
    out = sess.run(None, feeds)[0]
    note(f"  [{name}] ran in {time.time() - t0:.1f}s")
    np.save(OUT / f"{name}.out.npy", out)
    del sess
    torch.cuda.empty_cache()
    return out


class Full(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def main():
    note(f"torch {torch.__version__}  device "
         f"{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}")
    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415

    dino_p = [f for f in sorted(CKPT.rglob("*.safetensors")) if "dino" in f.name.lower()][0]
    note(f"weights {dino_p.name}")

    # ---- fp32: real weights, real size, no half precision anywhere ----------
    note("=" * 72)
    note("FP32")
    note("=" * 72)
    try:
        dino = T.load_dinov3(str(dino_p), device=DEV, dtype=torch.float32)
        note(f"  loaded fp32, {sum(p.numel() for p in dino.parameters()) / 1e6:.0f} M params")
        px = torch.randn(1, 3, 1024, 1024, device=DEV, dtype=torch.float32)
        with torch.no_grad():
            t0 = time.time()
            ref = dino(px)
        note(f"  eager {tuple(ref.shape)} in {time.time() - t0:.1f}s")
        np.save(OUT / "fp32.ref.npy", ref.float().cpu().numpy())
        R["fp32"] = {"eager_stats": {"mean": round(ref.mean().item(), 5),
                                    "std": round(ref.std().item(), 5),
                                    "min": round(ref.min().item(), 4),
                                    "max": round(ref.max().item(), 4)}}
        out = export_and_run("dinov3_fp32", Full(dino).eval(), [px])
        R["fp32"]["accuracy"] = analyse("fp32", out, ref)
        save()
        del dino, ref
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        R["fp32"] = {"error": traceback.format_exc()[-2500:]}
        note("  FP32 FAILED")
        note(traceback.format_exc()[-2500:])
        save()

    # ---- fp16: the configuration that fails --------------------------------
    note("=" * 72)
    note("FP16")
    note("=" * 72)
    try:
        dino16 = T.load_dinov3(str(dino_p), device=DEV, dtype=torch.float16)
        px16 = torch.randn(1, 3, 1024, 1024, device=DEV, dtype=torch.float16)
        with torch.no_grad():
            t0 = time.time()
            ref16 = dino16(px16)
        note(f"  eager {tuple(ref16.shape)} in {time.time() - t0:.1f}s")
        np.save(OUT / "fp16.ref.npy", ref16.float().cpu().numpy())
        out16 = export_and_run("dinov3_fp16", Full(dino16).eval(), [px16])
        R["fp16"] = {"accuracy": analyse("fp16", out16, ref16)}
        save()
    except Exception:  # noqa: BLE001
        R["fp16"] = {"error": traceback.format_exc()[-2500:]}
        note("  FP16 FAILED")
        note(traceback.format_exc()[-2500:])
        save()

    # ---- is the defect dtype-dependent or not -----------------------------
    try:
        if "accuracy" in R.get("fp32", {}) and "accuracy" in R.get("fp16", {}):
            ok32 = R["fp32"]["accuracy"]["flat_cosine"]
            ok16 = R["fp16"]["accuracy"]["flat_cosine"]
            note("\n" + "=" * 72)
            note("VERDICT")
            note("=" * 72)
            if ok32 > 0.999 and ok16 < 0.9:
                note(f"  fp32 {ok32:.4f} ok, fp16 {ok16:.4f} broken -> precision")
                note("  the graph is right; fp16 arithmetic or fp16 kernels differ.")
                note("  next: keep the encoder fp32, or find the op that overflows.")
                R["verdict"] = "precision"
            elif ok32 < 0.9:
                note(f"  fp32 {ok32:.4f} also broken -> structural, not precision")
                note("  next: bisect depth on the real model (blocks 4/8/16).")
                R["verdict"] = "structural"
            else:
                note(f"  inconclusive: fp32 {ok32:.4f} fp16 {ok16:.4f}")
                R["verdict"] = "inconclusive"
    except Exception:  # noqa: BLE001
        note(traceback.format_exc()[-800:])

    save()
    note(f"\nwrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
