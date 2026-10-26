"""K0e - fp16 is the fault. Which op?

K0d settled the framing with two numbers:

    fp32 graph vs eager   cosine 1.0003
    fp16 graph vs eager   cosine 0.0750

Same weights, same exporter, same input. So the graph is correct and fp16
arithmetic is not. It also measured what "wrong" means here: unrelated patch
tokens in this model score 0.42 against each other, and the exported patch
tokens score 0.067 - below unrelated. They are not merely mismatched, they
have collapsed.

The likely cause is the one the model's own loader hints at: TripoSplat loads
dinov3 in bf16, not fp16. bf16 has fp32's exponent range; fp16 tops out at
65504. Any intermediate that outgrows that in fp16 becomes inf, and everything
downstream is garbage. K0b's earlier measurement is consistent with this -
fp16 *eager* matched bf16 eager at cosine 0.99994, because torch's
scaled_dot_product_attention accumulates in fp32 and never lets an activation
sit in fp16 long enough to overflow. An exported graph has no such courtesy:
ORT decomposes attention into fp16 matmul, fp16 softmax, fp16 matmul, and each
of the 32 layers multiplies the damage.

This probe does two things.

1. Watch the activations. Hooks on every block record max |x| in fp32 and in
   fp16 for the same input. If anything approaches 65504 the overflow story is
   confirmed directly instead of inferred.

2. Ablate, one surface at a time, by monkeypatching what the model calls and
   re-exporting: attention in fp32, then the MLP, then LayerNorm. Whichever
   ablation restores the cosine is the one the browser build has to honour -
   and all three are things a WebGPU kernel can do, since fp32 math is legal
   there even when fp16 storage is not.

Writes k0e/report.json.
"""

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k0e"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = {}
FP16_MAX = 65504.0


def note(msg=""):
    print(msg, flush=True)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=2))


def cos(a, b):
    return round(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item(), 6)


def per_token(a, b):
    c = F.cosine_similarity(a.float()[0], b.float()[0], dim=-1)
    return {"prefix": [round(v, 6) for v in c[:5].tolist()],
            "patch_mean": round(c[5:].mean().item(), 6)}


# ---------------------------------------------------------------------------
# fp32 substitutes for the three candidate surfaces
# ---------------------------------------------------------------------------

_orig_sdpa = F.scaled_dot_product_attention


def sdpa_fp32(query, key, value, *args, **kwargs):
    """Attention with fp32 arithmetic, fp16 storage. Torch already does this
    internally; an exported graph does not, which is the whole hypothesis."""
    out = _orig_sdpa(query.float(), key.float(), value.float(), *args, **kwargs)
    return out.to(query.dtype)


_orig_layer_norm = F.layer_norm


def layer_norm_fp32(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    out = _orig_layer_norm(input.float(), normalized_shape,
                           None if weight is None else weight.float(),
                           None if bias is None else bias.float(), eps)
    return out.to(input.dtype)


_orig_mlp_forward = None


def mlp_fp32_forward(self, x):
    out = self.down_proj(F.silu(self.gate_proj(x.float())) * self.up_proj(x.float()))
    return out.to(x.dtype)


def patch(attn=False, mlp=False, norm=False):
    F.scaled_dot_product_attention = sdpa_fp32 if attn else _orig_sdpa
    if mlp:
        import model as M  # noqa: PLC0415
        M.DinoV3MLP.forward = mlp_fp32_forward
    else:
        import model as M  # noqa: PLC0415
        M.DinoV3MLP.forward = _orig_mlp_forward
    if norm:
        F.layer_norm = layer_norm_fp32
        nn.functional.layer_norm = layer_norm_fp32
    else:
        F.layer_norm = _orig_layer_norm
        nn.functional.layer_norm = _orig_layer_norm


def export_onnx(name, mod, args):
    path = OUT / f"{name}.onnx"
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(mod, tuple(args), str(path), do_constant_folding=True,
                          dynamo=True, opset_version=18)
    note(f"  [{name}] exported {time.time() - t0:.0f}s")
    torch.cuda.empty_cache()
    return path


def ort_run(path, args):
    import onnxruntime as ort
    provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
             if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(path), providers=provs)
    feeds = {s.name: a.detach().cpu().numpy() for s, a in zip(sess.get_inputs(), args)}
    out = sess.run(None, feeds)[0]
    del sess
    torch.cuda.empty_cache()
    return torch.from_numpy(out)


class Full(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def main():
    global _orig_mlp_forward
    note(f"torch {torch.__version__}  device "
         f"{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}")
    sys.path.insert(0, str(REPO))
    import model as M  # noqa: PLC0415
    import triposplat as T  # noqa: PLC0415

    _orig_mlp_forward = M.DinoV3MLP.forward

    dino_p = [f for f in sorted(CKPT.rglob("*.safetensors")) if "dino" in f.name.lower()][0]

    torch.manual_seed(0)
    x32 = torch.randn(1, 3, 1024, 1024, device=DEV, dtype=torch.float32)

    # ---- 1. fp32 reference, and the activation map ------------------------
    note("=" * 72)
    note("FP32 REFERENCE + ACTIVATION MAP")
    note("=" * 72)
    dino32 = T.load_dinov3(str(dino_p), device=DEV, dtype=torch.float32)
    with torch.no_grad():
        ref32 = dino32(x32)
    note(f"  fp32 out {tuple(ref32.shape)}  max|x| {ref32.abs().max().item():.1f}")

    def block_stats(model):
        rec = []
        hs = [b.register_forward_hook(
            lambda m, i, o, k=k: rec.append((k, float(o.abs().max().item()),
                                             float(o.abs().mean().item()))))
            for k, b in enumerate(model.blocks)]
        return hs, rec

    hs, rec32 = block_stats(dino32)
    with torch.no_grad():
        dino32(x32)
    for h in hs:
        h.remove()
    note(f"  fp32 block max|x|: first {rec32[0][1]:.0f}  peak "
         f"{max(r[1] for r in rec32):.0f}  last {rec32[-1][1]:.0f}")
    R["fp32_blocks"] = {"peak": round(max(r[1] for r in rec32), 1),
                        "first": round(rec32[0][1], 1),
                        "last": round(rec32[-1][1], 1)}
    np32 = ref32.cpu()
    del dino32, ref32
    torch.cuda.empty_cache()

    # ---- 2. fp16 eager, same input ---------------------------------------
    note("=" * 72)
    note("FP16 EAGER (same input, cast)")
    note("=" * 72)
    dino16 = T.load_dinov3(str(dino_p), device=DEV, dtype=torch.float16)
    x16 = x32.half()
    with torch.no_grad():
        ref16 = dino16(x16)
    note(f"  fp16 eager vs fp32 eager: cos {cos(ref16, np32)}")
    R["fp16_eager_vs_fp32"] = {"cos": cos(ref16, np32), **per_token(ref16.cpu(), np32)}

    hs, rec16 = block_stats(dino16)
    with torch.no_grad():
        dino16(x16)
    for h in hs:
        h.remove()
    over = [(k, round(v, 1)) for k, v, _ in rec16 if v > 1000]
    note(f"  fp16 block max|x|: first {rec16[0][1]:.0f}  peak "
         f"{max(r[1] for r in rec16):.0f}  last {rec16[-1][1]:.0f}")
    note(f"  blocks above 1000: {over[:12]}{' ...' if len(over) > 12 else ''}")
    R["fp16_blocks"] = {"peak": round(max(r[1] for r in rec16), 1),
                        "over_1000": over,
                        "near_overflow": [k for k, v, _ in rec16 if v > FP16_MAX * 0.5]}
    R["fp32_vs_fp16_block_ratio"] = [
        {"block": k, "fp32": round(rec32[k][1], 1), "fp16": round(rec16[k][1], 1)}
        for k in range(min(len(rec32), len(rec16)))][:4]
    save()

    # ---- 3. baseline: the fp16 graph that fails --------------------------
    ref_cpu = ref16.cpu()
    note("=" * 72)
    note("ABLATIONS (fp16 storage, fp32 math on one surface at a time)")
    note("=" * 72)

    cases = [
        ("baseline", dict(attn=False, mlp=False, norm=False)),
        ("attn_fp32", dict(attn=True, mlp=False, norm=False)),
        ("attn_mlp_fp32", dict(attn=True, mlp=True, norm=False)),
        ("attn_mlp_norm_fp32", dict(attn=True, mlp=True, norm=True)),
    ]
    for name, flags in cases:
        try:
            patch(**flags)
            with torch.no_grad():
                eager_patched = dino16(x16)
            path = export_onnx(name, Full(dino16).eval(), [x16])
            out = ort_run(path, [x16])
            c = cos(out, eager_patched.cpu())
            R.setdefault("ablations", {})[name] = {
                "flags": flags,
                # against the graph's own eager reference: isolates export loss
                "ort_vs_patched_eager": c,
                **per_token(out, eager_patched.cpu()),
                # against the true fp16 eager: did the ablation fix the model?
                "ort_vs_unpatched_eager": cos(out, ref_cpu),
            }
            note(f"  {name:20s} ort vs patched-eager {c:.4f}   "
                 f"ort vs unpatched {cos(out, ref_cpu):.4f}")
        except Exception:  # noqa: BLE001
            R.setdefault("ablations", {})[name] = {"error": traceback.format_exc()[-1200:]}
            note(f"  {name} FAILED")
            note(traceback.format_exc()[-1200:])
        save()
        torch.cuda.empty_cache()

    patch(attn=False, mlp=False, norm=False)
    save()

    note("=" * 72)
    note("SUMMARY")
    note("=" * 72)
    note(f"  fp16 eager vs fp32 eager  {R.get('fp16_eager_vs_fp32', {}).get('cos')}")
    note(f"  fp32 peak activation      {R.get('fp32_blocks', {}).get('peak')}")
    note(f"  fp16 peak activation      {R.get('fp16_blocks', {}).get('peak')}")
    for name, rec in R.get("ablations", {}).items():
        if "error" in rec:
            note(f"  {name:20s} FAILED")
        else:
            note(f"  {name:20s} ort/patched={rec['ort_vs_patched_eager']} "
                 f"ort/unpatched={rec['ort_vs_unpatched_eager']}")


if __name__ == "__main__":
    main()
