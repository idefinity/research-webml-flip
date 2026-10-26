"""K0c - why does the exported DINOv3 compute something else?

K0b measured the problem precisely and explained none of it: the dinov3 graph
loads, runs, and returns a tensor whose cosine against eager fp16 is 0.076.
Both export paths (dynamo and torchscript) produced that same number, on the
same weights, from the same input. Identical errors from independent code
paths mean a deterministic difference in *what the graph computes*, not
floating point noise.

The source gives one strong candidate. DinoV3Attention splits q and k into
prefix and patch halves and applies rotary embedding only to the patch half:

    q_pre, q_pat = q.split((num_prefix_tokens, N - num_prefix_tokens), dim=-2)
    q = cat((q_pre, q_pat * cos + _rotate_half(q_pat) * sin), dim=-2)

with num_prefix_tokens = 1 + 4 = 5 (cls + registers). If the exported graph
mishandles that split, the five prefix tokens come out exact and every patch
token comes out wrong. That predicts a very specific fingerprint: the overall
cosine collapses toward zero while the first five token vectors match to
~1.0. This script measures exactly that, per token, instead of arguing about it.

Two parts, cheapest first:

  A. a 2-layer random-init DinoV3ViT in fp32. No weights, no download, a few
     seconds. If this exports faithfully, the architecture is not the problem
     and the real-model failure is about scale. If it does not, the bug is
     reproducible without 1.7 GB of anything, and the ladder below bisects it
     down to patch_embed / token stacking / block 0 / final norm.

  B. the real encoder in fp16, with the cast done *before* any forward pass so
     that no bf16 cache can contaminate the reference. Same per-token
     diagnostic, plus a permutation check (is the output the same multiset of
     values, just reordered?) and a constant-output check.

Writes k0c/report.json.
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
OUT = WORK / "k0c"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = {"tiny": {}, "real": {}, "notes": []}


def note(msg=""):
    print(msg, flush=True)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=2))


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def compare(tag, got, ref):
    """Cosine plus a per-token breakdown, which is what actually localises it."""
    a = got.detach().float().cpu()
    b = ref.detach().float().cpu()
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    flat_cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rec = {
        "cosine": round(flat_cos, 6),
        "max_abs": round((a - b).abs().max().item(), 6),
        "mean_abs": round((a - b).abs().mean().item(), 6),
    }
    if a.dim() == 3:  # (B, N, C) - split prefix vs patch
        per_tok = F.cosine_similarity(a[0], b[0], dim=-1)  # (N,)
        n_pre = min(5, per_tok.numel())
        rec["n_tokens"] = int(per_tok.numel())
        rec["cos_prefix"] = [round(v, 6) for v in per_tok[:n_pre].tolist()]
        rec["cos_patch_mean"] = round(per_tok[n_pre:].mean().item(), 6) if per_tok.numel() > n_pre else None
        rec["cos_patch_min"] = round(per_tok[n_pre:].min().item(), 6) if per_tok.numel() > n_pre else None
        rec["n_patch_tokens_below_0_9"] = int((per_tok[n_pre:] < 0.9).sum().item())
        # same multiset of values, different order? then it is a layout bug
        rec["sorted_values_match"] = bool(torch.allclose(
            a.flatten().sort().values, b.flatten().sort().values,
            rtol=1e-3, atol=1e-3))
    note(f"  [{tag}] cos={rec['cosine']:.4f} mean_abs={rec['mean_abs']:.5f}"
         + (f" prefix={rec.get('cos_prefix')} patch_mean={rec.get('cos_patch_mean')}"
            if rec.get("cos_prefix") else ""))
    return rec


def stats(t):
    t = t.detach().float()
    return {"mean": round(t.mean().item(), 4), "std": round(t.std().item(), 4),
            "min": round(t.min().item(), 4), "max": round(t.max().item(), 4),
            "nan": bool(torch.isnan(t).any()), "inf": bool(torch.isinf(t).any())}


def export_and_check(name, mod, args, ref, allow_cpu=False,
                     opsets=(("dynamo", {"dynamo": True, "opset_version": 18}),
                             ("ts", {"dynamo": False, "opset_version": 17}))):
    import onnxruntime as ort
    rec = {}
    for tag, kw in opsets:
        path = OUT / f"{name}.{tag}.onnx"
        t0 = time.time()
        try:
            with torch.no_grad():
                torch.onnx.export(mod, tuple(args), str(path),
                                  do_constant_folding=True, **kw)
        except Exception:  # noqa: BLE001
            rec[tag] = {"ok": False, "err": traceback.format_exc()[-900:]}
            note(f"  [{name}/{tag}] EXPORT FAILED {time.time() - t0:.1f}s")
            note(traceback.format_exc()[-900:])
            torch.cuda.empty_cache()
            continue
        secs = time.time() - t0
        ext = sum(p.stat().st_size for p in OUT.glob(f"{name}.{tag}.onnx.data")) / 1e6
        mb = path.stat().st_size / 1e6 + ext
        try:
            import onnx  # noqa: PLC0415
            m = onnx.load(str(path), load_external_data=False)
            from collections import Counter
            ops = Counter(n.op_type for n in m.graph.node)
            rc = {"opset": m.opset_import[0].version, "nodes": sum(ops.values()),
                  "range_nodes": ops.get("Range", 0), "tile_nodes": ops.get("Tile", 0),
                  "expand_nodes": ops.get("Expand", 0)}
        except Exception:  # noqa: BLE001
            rc = {}
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        if not providers or (not allow_cpu and providers[0] == "CPUExecutionProvider"):
            rc.update({"ok": False, "skipped": True,
                       "err": f"no CUDA EP; providers={providers}"})
            note(f"  [{name}/{tag}] exported {mb:.1f} MB but no CUDA EP, "
                 f"not verifying a graph this size on CPU")
            rec[tag] = rc
            continue
        try:
            t0 = time.time()
            sess = ort.InferenceSession(str(path), providers=providers)
            load_s = time.time() - t0
            feeds = {s.name: a.detach().cpu().numpy()
                     for s, a in zip(sess.get_inputs(), args)}
            out = sess.run(None, feeds)
            o0 = torch.from_numpy(out[0])
            rc.update({"ok": True, "mb": round(mb, 1), "seconds": round(secs, 1),
                       "load_s": round(load_s, 1), "n_outputs": len(out),
                       "stats": stats(o0)})
            rc["accuracy"] = compare(f"{name}/{tag}", o0, ref)
            # a graph that ignores its input would return the same thing twice
            alt = {k: (v * 0 + 1 if v.dtype.kind == "f" else v) for k, v in feeds.items()}
            out2 = sess.run(None, alt)
            rc["changes_with_input"] = bool(
                abs(float(torch.from_numpy(out2[0]).float().mean())
                    - float(o0.float().mean())) > 1e-4)
            note(f"  [{name}/{tag}] {mb:.1f} MB {rc.get('opset')} opset "
                 f"{rc['nodes']} nodes  input-sensitive={rc['changes_with_input']}")
        except Exception:  # noqa: BLE001
            rc.update({"ok": False, "err": traceback.format_exc()[-900:]})
            note(f"  [{name}/{tag}] ORT FAILED")
            note(traceback.format_exc()[-700:])
        rec[tag] = rc
        torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------------------
# A. the architecture, in miniature
# ---------------------------------------------------------------------------

def part_a(M):
    note("=" * 72)
    note("A. TINY DinoV3ViT (fp32, random init, no weights)")
    note("=" * 72)
    torch.manual_seed(0)
    kw = dict(hidden_size=128, num_heads=4, num_layers=2, patch_size=16,
              num_register_tokens=4, intermediate_size=512, layerscale_init=1.0,
              query_bias=True, key_bias=False, value_bias=True, mlp_bias=True,
              rope_theta=100.0, layer_norm_eps=1e-5)
    d = M.DinoV3ViT(**kw).to(DEV, torch.float32).eval()
    note(f"  params {sum(p.numel() for p in d.parameters()) / 1e6:.2f} M")
    px = torch.randn(1, 3, 64, 64, device=DEV, dtype=torch.float32)
    with torch.no_grad():
        ref = d(px)
    note(f"  eager -> {tuple(ref.shape)}  {stats(ref)}")

    class Full(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)

    class PatchOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.patch_embed(x)

    class Stacked(nn.Module):
        """patch_embed + cls/register concat, no attention, no norm."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            B = x.shape[0]
            t = self.m.patch_embed(x)
            return torch.cat([self.m.cls_token.expand(B, -1, -1),
                              self.m.register_tokens.expand(B, -1, -1), t], dim=1)

    class FirstK(nn.Module):
        def __init__(self, m, k, final=False):
            super().__init__()
            self.m = m
            self.k = k
            self.final = final

        def forward(self, x):
            m = self.m
            B, _, H, W = x.shape
            t = m.patch_embed(x)
            hp, wp = H // m.patch_size, W // m.patch_size
            cos, sin = m.rope(hp, wp, t.device, t.dtype)
            t = torch.cat([m.cls_token.expand(B, -1, -1),
                           m.register_tokens.expand(B, -1, -1), t], dim=1)
            npre = 1 + m.num_register_tokens
            for blk in m.blocks[: self.k]:
                t = blk(t, cos, sin, num_prefix_tokens=npre)
            return m.norm(t) if self.final else t

    ladder = [
        ("patch_embed", PatchOnly(d).eval(), lambda: d.patch_embed(px)),
        ("tokens_stacked", Stacked(d).eval(), lambda: torch.cat(
            [d.cls_token.expand(1, -1, -1), d.register_tokens.expand(1, -1, -1),
             d.patch_embed(px)], dim=1)),
        ("block1", FirstK(d, 1).eval(), lambda: FirstK(d, 1).forward(px)),
        ("block2", FirstK(d, 2).eval(), lambda: FirstK(d, 2).forward(px)),
        ("full_plus_norm", FirstK(d, 2, final=True).eval(),
         lambda: FirstK(d, 2, final=True).forward(px)),
    ]

    for name, mod, eager in ladder:
        with torch.no_grad():
            r = eager()
        note(f"\n[{name}] eager {tuple(r.shape)} {stats(r)}")
        R["tiny"][name] = {"eager": stats(r),
                           "export": export_and_check(name, mod, [px], r, allow_cpu=True)}
        save()

    with torch.no_grad():
        again = Full(d).eval()(px)
    note(f"\n  full-vs-full sanity: {compare('tiny/full-self', again, ref)['cosine']}")
    return d


# ---------------------------------------------------------------------------
# B. the real encoder
# ---------------------------------------------------------------------------

def part_b(dino):
    note("\n" + "=" * 72)
    note("B. REAL DINOv3 (fp16, cast before any forward)")
    note("=" * 72)
    # cast first: a bf16 forward followed by an in-place .half() would leave the
    # reference produced by weights that no longer exist, which is a good way to
    # measure nothing at all
    dino.half()
    torch.cuda.empty_cache()
    note(f"  dtype now {next(dino.parameters()).dtype}")

    px = torch.randn(1, 3, 1024, 1024, device=DEV, dtype=torch.float16)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        ref = dino(px)
        torch.cuda.synchronize()
    note(f"  eager fp16 {tuple(ref.shape)} in {(time.time() - t0) * 1000:.0f} ms  {stats(ref)}")
    R["real"]["eager_ms"] = round((time.time() - t0) * 1000, 1)
    R["real"]["eager_stats"] = stats(ref)

    class Full(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)

    R["real"]["export"] = export_and_check("dinov3_fp16", Full(dino).eval(), [px], ref)
    save()


def main():
    note(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}")
    sys.path.insert(0, str(REPO))
    import model as M  # noqa: PLC0415

    try:
        part_a(M)
    except Exception:  # noqa: BLE001
        note("PART A CRASHED")
        note(traceback.format_exc()[-2500:])
        R["tiny"]["crash"] = traceback.format_exc()[-2000:]

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
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))
    dino = next(v for v in vars(pipe).values() if type(v).__name__ == "DinoV3ViT")
    del pipe
    torch.cuda.empty_cache()

    try:
        part_b(dino)
    except Exception:  # noqa: BLE001
        note("PART B CRASHED")
        note(traceback.format_exc()[-2500:])
        R["real"]["crash"] = traceback.format_exc()[-2000:]

    save()

    note("\n" + "=" * 72)
    note("SUMMARY")
    note("=" * 72)
    for name, rec in R["tiny"].items():
        if not isinstance(rec, dict) or "export" not in rec:
            continue
        for tag, r in rec["export"].items():
            acc = (r.get("accuracy") or {})
            note(f"  tiny/{name:16s} {tag:6s} ok={r.get('ok')} "
                 f"cos={acc.get('cosine')} prefix={acc.get('cos_prefix')}")

    note(f"  wrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
