"""K0g - put the fix in the exported graph and prove it there.

K0f named the fault and the cure in pure PyTorch:

    fp16 model, torch's own LayerNorm     cosine 1.000332
    fp16 model, variance in fp16          0.069275   <- the exported graph
    fp16 model, variance in fp32          1.000334   <- the cure

The arithmetic is settled. What is not settled is whether the cure survives
export. K0e already showed that a monkeypatched `F.layer_norm` never reaches
the graph - dynamo lowers to aten and ignores the Python wrapper. So the fix
has to be a real `nn.Module` whose forward contains explicit casts, because
those casts become explicit `Cast` nodes.

This probe runs the A/B inside one session, same weights, same input:

    A  export as-is            expect cosine ~0.075   (reproduces K0b/K0c)
    B  swap 65 LayerNorms for
       LayerNormFP32Stats      expect cosine ~1.000   (the Gate 1 bar is 0.999)

and reports how many Cast-to-float nodes each graph contains, so "the cure
reached the graph" is a measurement rather than a hope.

Also reports the provider ORT actually selected. A silent fall back to CPU
would make the cosine meaningless as evidence about the browser.

Writes k0g/report.json.
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
OUT = WORK / "k0g"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = {}


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
    patch = c[5:]
    return {
        "prefix": [round(v, 6) for v in c[:5].tolist()],
        "patch_mean": round(patch.mean().item(), 6),
        "patch_below_09": int((patch < 0.9).sum().item()),
        "n_patch": int(patch.numel()),
    }


# ---------------------------------------------------------------------------
# The cure, as a module. Explicit casts in a real forward - not a monkeypatch.
# ---------------------------------------------------------------------------


class LayerNormFP32Stats(nn.Module):
    """nn.LayerNorm with the reduction in fp32 and the storage untouched.

    (x - mu) ** 2 rather than x*x - mu*mu: the second form cancels badly once
    the mean dwarfs the variance, which is exactly the regime here - K0f
    measured per-token mean(x^2) up to 1.4e6 while the residual sits at 3e4.
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

    def extra_repr(self):
        return f"eps={self.eps}"


def swap_layernorms(model):
    """Replace every nn.LayerNorm in place, keeping weight and bias."""
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


class Full(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


# ---------------------------------------------------------------------------


def count_casts(path):
    """Cast-to-float nodes in the graph, read from the .onnx next to its .data."""
    try:
        import onnx  # noqa: PLC0415

        m = onnx.load(str(path), load_external_data=False)
        casts = [n for n in m.graph.node if n.op_type == "Cast"]
        to_float = 0
        for n in casts:
            for a in n.attribute:
                if a.name == "to" and a.i == 1:  # TensorProto.FLOAT
                    to_float += 1
        return {"nodes": len(m.graph.node), "cast_total": len(casts),
                "cast_to_float": to_float}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def export_fp16(name, mod, args):
    path = OUT / f"{name}.onnx"
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(mod, tuple(args), str(path), do_constant_folding=True,
                          dynamo=True, opset_version=18)
    dt = time.time() - t0
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in OUT.glob(f"{name}.onnx*")}
    note(f"  [{name}] exported in {dt:.0f}s  {sizes}")
    return path, round(dt, 1), sizes


def ort_session(path):
    import onnxruntime as ort  # noqa: PLC0415
    want = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=want)
    return sess


def run_case(name, model, x16, ref, do_swap):
    rec = {"swapped_norms": 0}
    if do_swap:
        rec["swapped_norms"] = swap_layernorms(model)
        note(f"  swapped {rec['swapped_norms']} LayerNorm modules")

    with torch.no_grad():
        t0 = time.time()
        eager = model(x16)
        rec["eager_s"] = round(time.time() - t0, 2)
    rec["eager_vs_ref"] = cos(eager, ref)

    path, secs, sizes = export_fp16(name, Full(model).eval(), [x16])
    rec["export_s"] = secs
    rec["file_mb"] = sizes
    rec["graph"] = count_casts(path)

    sess = ort_session(path)
    rec["providers"] = sess.get_providers()
    feeds = {i.name: x16.detach().cpu().numpy() for i in sess.get_inputs()}
    t0 = time.time()
    out = torch.from_numpy(sess.run(None, feeds)[0])
    rec["ort_s"] = round(time.time() - t0, 2)
    del sess
    torch.cuda.empty_cache()

    rec["ort_vs_eager"] = cos(out, eager)
    rec.update({f"ort_{k}": v for k, v in per_token(out, eager).items()})
    rec["ort_vs_ref"] = cos(out, ref)
    note(f"  [{name}] ORT vs eager {rec['ort_vs_eager']:.6f}   "
         f"patch {rec['ort_patch_mean']:.6f}   "
         f"eager vs ref {rec['eager_vs_ref']:.6f}")
    note(f"  [{name}] providers {rec['providers']}   "
         f"cast->float {rec['graph'].get('cast_to_float')}")
    return rec, out


def main():
    note(f"torch {torch.__version__}  device "
         f"{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}")
    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415

    dino_p = [f for f in sorted(CKPT.rglob("*.safetensors")) if "dino" in f.name.lower()][0]

    torch.manual_seed(0)
    x16 = torch.randn(1, 3, 1024, 1024, device=DEV, dtype=torch.float16)

    note("=" * 74)
    note("LOAD + REFERENCE (fp32 model, how the pipeline actually runs it)")
    note("=" * 74)
    dino32 = T.load_dinov3(str(dino_p), device=DEV, dtype=torch.float32)
    with torch.no_grad():
        ref = dino32(x16.float()).cpu()
    del dino32
    torch.cuda.empty_cache()
    note(f"  reference out {tuple(ref.shape)}")

    dino16 = T.load_dinov3(str(dino_p), device=DEV, dtype=torch.float16)

    note("=" * 74)
    note("A - EXPORT AS-IS (the graph that fails)")
    note("=" * 74)
    try:
        rec_a, _ = run_case("dino_asis", dino16, x16, ref, do_swap=False)
        R["A_asis"] = rec_a
    except Exception:  # noqa: BLE001
        R["A_asis"] = {"error": traceback.format_exc()[-1500:]}
        note(traceback.format_exc()[-1500:])
    save()

    # fresh model so a failed export cannot poison the second case
    del dino16
    torch.cuda.empty_cache()
    dino16 = T.load_dinov3(str(dino_p), device=DEV, dtype=torch.float16)

    note("=" * 74)
    note("B - EXPORT WITH LayerNormFP32Stats")
    note("=" * 74)
    try:
        rec_b, _ = run_case("dino_lnfp32", dino16, x16, ref, do_swap=True)
        R["B_lnfp32"] = rec_b
    except Exception:  # noqa: BLE001
        R["B_lnfp32"] = {"error": traceback.format_exc()[-1500:]}
        note(traceback.format_exc()[-1500:])
    save()

    note("=" * 74)
    note("SUMMARY")
    note("=" * 74)
    for k in ("A_asis", "B_lnfp32"):
        r = R.get(k, {})
        if "error" in r:
            note(f"  {k:10s} FAILED")
        else:
            note(f"  {k:10s} ort/eager {r['ort_vs_eager']:.6f}  "
                 f"patch {r['ort_patch_mean']:.6f}  "
                 f"cast->float {r['graph'].get('cast_to_float')}  "
                 f"providers {r['providers'][0]}")
    a = R.get("A_asis", {}).get("ort_vs_eager")
    b = R.get("B_lnfp32", {}).get("ort_vs_eager")
    note("")
    note(f"  Gate 1 bar is cosine >= 0.999 on the exported graph.")
    if a is not None and b is not None:
        note(f"  A {a:.4f} -> B {b:.4f}.  "
             f"{'PASS' if b >= 0.999 else 'STILL FAILING'}")
    note("  A is expected near 0.075 - that is the fault, reproduced on purpose.")


if __name__ == "__main__":
    main()
