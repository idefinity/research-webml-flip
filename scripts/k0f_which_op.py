"""K0f - LayerNorm is the surface, and it is per token, not per tensor.

K0e gave a negative result and a bug. The negative result cannot be trusted:
the ablations compared ORT against a *patched* eager, and for `attn_fp32` the
two numbers came out 0.0746 and 0.074612 - i.e. the monkeypatch changed the
eager path by nothing, which means the dynamo exporter never traced it. Two of
the three ablations then died on a dtype error in the MLP patch, and the one
that mattered most, "norm alone", was never in the variant list at all.

So stop asking ONNX. The export is not what we need to diagnose; the *arithmetic*
is. This probe reproduces the collapse in pure PyTorch, with no exporter in the
loop, and names the op.

Why LayerNorm is the suspect, arithmetically:

    K0e measured the fp32 residual stream peaking at 33603.

nn.LayerNorm normalises over the last dim, so it is per token, and it needs the
variance, which is a mean of *squares*. A residual token sitting at magnitude
3e4 has squares near 1e9. fp16 tops out at 65504. The square overflows before
the variance is ever formed, and anything divided by sqrt(inf) is zero - the
token comes out constant across all 1280 channels, which is exactly the
"collapsed patch tokens" signature K0d measured (cos 0.067, below the 0.42 that
two unrelated tokens score).

And it explains the one detail that never fit: in K0c the prefix tokens scored
0.99999, 1.0, 0.99999, 0.99999 - except token 2, at -0.04, and every one of the
4096 patch tokens sat at 0.067. A per-tensor overflow would have taken all of
them. A per-token one takes only the tokens whose magnitude is large, and in
this model that is the patch tokens plus one unlucky register.

Torch's own fused LayerNorm promotes the reduction to fp32 (`acc_type`), which
is why fp16 *eager* matches fp32 at cosine 1.0003 while an exported fp16 graph
does not. An ONNX graph has no such promotion: it gets Mean, Mul, Sqrt in fp16.

Four things this writes down:

1. Per-token mean(x^2) at the input of every LayerNorm in the encoder, in fp32 -
   the quantity that has to fit in fp16, and the count of tokens that do not.
2. An fp16 run with LayerNorm statistics computed naively in fp16. If this
   lands near cosine 0.067 we have reproduced the export failure without
   exporting anything.
3. An fp16 run with LayerNorm statistics forced to fp32 but storage kept fp16.
   This is the fix, and it is one a WebGPU kernel can honour.
4. Attribution controls: attention and the MLP each run with fp16 arithmetic
   and no fp32 accumulation, to show they are not the surface.

Writes k0f/report.json.
"""

import json
import sys
import types
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k0f"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = {}
FP16_MAX = 65504.0
FP16_SQUARE_LIMIT = 256.0  # 256**2 already exceeds FP16_MAX


def note(msg=""):
    print(msg, flush=True)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=2))


def _cpu(t):
    return (t.detach() if torch.is_tensor(t) else torch.as_tensor(t)).float().cpu()


def cos(a, b):
    a, b = _cpu(a), _cpu(b)
    return round(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(), 6)


def per_token(a, b):
    a, b = _cpu(a), _cpu(b)
    c = F.cosine_similarity(a[0], b[0], dim=-1)
    n_pref = 5
    patches = c[n_pref:]
    return {
        "prefix": [round(v, 6) for v in c[:n_pref].tolist()],
        "patch_mean": round(patches.mean().item(), 6),
        "patch_below_09": int((patches < 0.9).sum().item()),
        "n_patch": int(patches.numel()),
    }


# ---------------------------------------------------------------------------
# The candidates. Each is a drop-in LayerNorm forward, or a drop-in attention
# or MLP forward, so the model itself is untouched.
# ---------------------------------------------------------------------------


def ln_torch(self, x):
    return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


def ln_naive_fp16(self, x):
    """What an fp16 ONNX graph gets: mean, square, mean, sqrt - all in fp16.

    The square overflows for any channel above 256, the variance becomes inf,
    rsqrt gives 0, and the token collapses to (bias) for every channel.
    """
    mu = x.mean(-1, keepdim=True)
    var = (x * x).mean(-1, keepdim=True) - mu * mu
    out = (x - mu) * torch.rsqrt(var + self.eps)
    if self.weight is not None:
        out = out * self.weight
    if self.bias is not None:
        out = out + self.bias
    return out


def ln_fp32_stats(self, x):
    """The fix: statistics in fp32, storage stays fp16. Legal in WebGPU."""
    dt = x.dtype
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = (xf * xf).mean(-1, keepdim=True) - mu * mu
    y = (xf - mu) * torch.rsqrt(var + self.eps)
    if self.weight is not None:
        y = y * self.weight.float()
    if self.bias is not None:
        y = y + self.bias.float()
    return y.to(dt)


_orig_sdpa = F.scaled_dot_product_attention


def sdpa_naive_fp16(query, key, value, *args, **kwargs):
    """Everything in the storage dtype: matmul, softmax, matmul. No fp32 path."""
    dt = query.dtype
    scale = query.shape[-1] ** -0.5
    logits = (query @ key.transpose(-1, -2)) * scale
    logits = logits - logits.amax(-1, keepdim=True)
    logits = logits.exp()
    probs = logits / logits.sum(-1, keepdim=True)
    return (probs @ value).to(dt)


def _linear(m, x):
    dt = x.dtype
    return (F.linear(x.float(), m.weight.float(),
                     None if m.bias is None else m.bias.float())).to(dt)


def mlp_naive_fp16(self, x):
    """SwiGLU with no fp32 accumulation anywhere."""
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def mlp_fp32_math(self, x):
    """Same SwiGLU, fp32 math on fp16 weights - the control for the MLP."""
    return (_linear(self.down_proj,
                    F.silu(_linear(self.gate_proj, x)) * _linear(self.up_proj, x)))


def apply_ln(model, fn):
    n = 0
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.forward = types.MethodType(fn, m)
            n += 1
    return n


def restore_ln(model):
    return apply_ln(model, ln_torch)


def main():
    note(f"torch {torch.__version__}  device "
         f"{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}")
    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415

    dino_p = [f for f in sorted(CKPT.rglob("*.safetensors")) if "dino" in f.name.lower()][0]

    torch.manual_seed(0)
    x32 = torch.randn(1, 3, 1024, 1024, device=DEV, dtype=torch.float32)

    # ---- 1. fp32 reference -----------------------------------------------
    note("=" * 74)
    note("FP32 REFERENCE")
    note("=" * 74)
    dino32 = T.load_dinov3(str(dino_p), device=DEV, dtype=torch.float32)
    with torch.no_grad():
        ref32 = dino32(x32)
    ref32_cpu = ref32.cpu()
    note(f"  out {tuple(ref32.shape)}  max|x| {ref32.abs().max().item():.2f}")

    # ---- 2. the quantity that has to fit in fp16 --------------------------
    # mean(x^2) per token, at every LayerNorm input, accumulated in float64 so
    # the measurement itself cannot overflow.
    note("=" * 74)
    note("PER-TOKEN mean(x^2) AT EVERY LAYERNORM INPUT (fp32 model)")
    note("=" * 74)

    records = []

    def make_hook(tag):
        def hook(module, inputs):
            x = inputs[0].detach().double()
            sq = (x * x).mean(-1)          # [B, N] - per token
            mx = x.abs().amax(-1)          # [B, N]
            sq = sq[0]
            mx = mx[0]
            records.append({
                "where": tag,
                "sq_max": float(sq.max().item()),
                "sq_median": float(sq.median().item()),
                "over_limit": int((sq > FP16_MAX).sum().item()),
                "n_tokens": int(sq.numel()),
                "absmax": float(mx.max().item()),
                "prefix_sq": [round(float(v), 1) for v in sq[:5].tolist()],
                "patch_sq_median": float(sq[5:].median().item()),
            })
        return hook

    hooks = []
    for i, blk in enumerate(dino32.blocks):
        hooks.append(blk.norm1.register_forward_pre_hook(make_hook(f"b{i}.norm1")))
        hooks.append(blk.norm2.register_forward_pre_hook(make_hook(f"b{i}.norm2")))
    hooks.append(dino32.norm.register_forward_pre_hook(make_hook("final.norm")))

    with torch.no_grad():
        dino32(x32)
    for h in hooks:
        h.remove()

    worst_sq = max(r["sq_max"] for r in records)
    worst_abs = max(r["absmax"] for r in records)
    fully_over = [r for r in records
                  if r["over_limit"] >= r["n_tokens"] - 5]
    lonely_over = [r for r in records if 0 < r["over_limit"] < r["n_tokens"] - 5]

    note(f"  norms inspected            {len(records)}")
    note(f"  worst per-token mean(x^2)  {worst_sq:.4g}")
    note(f"  fp16 ceiling               {FP16_MAX:.0f}")
    note(f"  ratio                      {worst_sq / FP16_MAX:.1f}x over")
    note(f"  worst |x| seen             {worst_abs:.1f}   "
         f"(squares overflow above {FP16_SQUARE_LIMIT:.0f})")
    note(f"  norms where EVERY patch token overflows   {len(fully_over)}")
    note(f"  norms where only SOME tokens overflow     {len(lonely_over)}")
    if lonely_over:
        r = lonely_over[0]
        note(f"    e.g. {r['where']}: {r['over_limit']} of {r['n_tokens']} over, "
             f"prefix mean(x^2) {r['prefix_sq']}")
    note(f"  patch tokens would overflow in "
         f"{sum(1 for r in records if r['over_limit'] > 0)} of {len(records)} norms")

    R["layernorm_input"] = {
        "n_norms": len(records),
        "fp16_ceiling": FP16_MAX,
        "worst_token_mean_square": round(worst_sq, 1),
        "over_ratio": round(worst_sq / FP16_MAX, 1),
        "worst_abs": round(worst_abs, 1),
        "norms_all_tokens_overflow": len(fully_over),
        "norms_some_tokens_overflow": len(lonely_over),
        "norms_any_overflow": sum(1 for r in records if r["over_limit"] > 0),
        "per_norm": records[::8] + records[-3:],   # sample, keeps the file small
    }
    save()

    del dino32
    torch.cuda.empty_cache()

    # ---- 3. fp16 model, five arithmetic variants --------------------------
    note("=" * 74)
    note("FP16 MODEL - WHICH ARITHMETIC REPRODUCES THE COLLAPSE")
    note("=" * 74)
    dino16 = T.load_dinov3(str(dino_p), device=DEV, dtype=torch.float16)
    x16 = x32.half()

    def run(tag):
        with torch.no_grad():
            out = dino16(x16)
        if not torch.isfinite(out).all():
            bad = int((~torch.isfinite(out)).sum().item())
            note(f"  {tag:22s} NON-FINITE output ({bad} values)")
            return {"cos_vs_fp32": None, "non_finite": bad}
        c = cos(out, ref32_cpu)
        rec = {"cos_vs_fp32": c, **per_token(out, ref32_cpu)}
        note(f"  {tag:22s} cos {c:.6f}   patch {rec['patch_mean']:.6f}   "
             f"patch<0.9 {rec['patch_below_09']}/{rec['n_patch']}")
        return rec

    import model as M  # noqa: PLC0415

    orig_mlp = M.DinoV3MLP.forward

    def reset_all():
        restore_ln(dino16)
        M.DinoV3MLP.forward = orig_mlp
        F.scaled_dot_product_attention = _orig_sdpa

    # tag -> (configure, one-line description of what it isolates)
    variants = [
        ("torch_default", reset_all, "nothing patched - the baseline fp16 run"),
        ("ln_naive_fp16",
         lambda: apply_ln(dino16, ln_naive_fp16),
         "LayerNorm variance formed from fp16 squares - the ONNX graph"),
        ("ln_fp32_stats",
         lambda: apply_ln(dino16, ln_fp32_stats),
         "LayerNorm statistics in fp32, storage fp16 - the candidate fix"),
        ("attn_naive_fp16",
         lambda: setattr(F, "scaled_dot_product_attention", sdpa_naive_fp16),
         "attention with fp16 matmul and fp16 softmax, no fp32 path"),
        ("mlp_naive_fp16",
         lambda: setattr(M.DinoV3MLP, "forward", mlp_naive_fp16),
         "SwiGLU in pure fp16, no fp32 accumulation"),
    ]

    R["variants"] = {}
    for tag, setup, why in variants:
        reset_all()
        try:
            setup()
            R["variants"][tag] = {"what": why, **run(tag)}
        except Exception as exc:  # noqa: BLE001
            R["variants"][tag] = {"what": why,
                                  "error": f"{type(exc).__name__}: {exc}"}
            note(f"  {tag} FAILED: {type(exc).__name__}: {exc}")
        save()
        torch.cuda.empty_cache()

    # ---- 4. combined: the two that survived, together ---------------------
    note("=" * 74)
    note("COMBINED")
    note("=" * 74)
    for tag, ln_fn, mlp_fn in [
        ("ln_fp32+mlp_fp32", ln_fp32_stats, mlp_fp32_math),
        ("ln_fp32+mlp_naive", ln_fp32_stats, mlp_naive_fp16),
    ]:
        reset_all()
        try:
            apply_ln(dino16, ln_fn)
            M.DinoV3MLP.forward = mlp_fn
            R["variants"][tag] = run(tag)
        except Exception as exc:  # noqa: BLE001
            R["variants"][tag] = {"error": f"{type(exc).__name__}: {exc}"}
            note(f"  {tag} FAILED: {type(exc).__name__}: {exc}")
        save()

    # ---- 5. verdict ------------------------------------------------------
    reset_all()
    save()

    note("=" * 74)
    note("SUMMARY")
    note("=" * 74)
    note(f"  worst per-token mean(x^2) {worst_sq:.4g} vs fp16 ceiling "
         f"{FP16_MAX:.0f}  -> {worst_sq / FP16_MAX:.0f}x over")
    note(f"  norms with any overflowing token: "
         f"{R['layernorm_input']['norms_any_overflow']}/{len(records)}")
    for tag, rec in R["variants"].items():
        if "error" in rec:
            note(f"  {tag:22s} FAILED")
        else:
            note(f"  {tag:22s} cos {rec['cos_vs_fp32']:.6f}  "
                 f"patch {rec['patch_mean']:.6f}")
    note("")
    note("  Reference points from earlier probes: fp32 graph cos 1.0003;")
    note("  fp16 graph cos 0.0750; unrelated patch tokens pairwise cos 0.42.")
    note("  If ln_naive_fp16 lands near 0.07 and ln_fp32_stats near 1.0,")
    note("  LayerNorm is the surface and fp32 statistics is the fix.")


if __name__ == "__main__":
    main()
