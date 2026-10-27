"""K2m - pack the distilled DiT into 4-bit blocks, and measure what it costs.

The scheme is the one the WebFM Ming bundle already ships: 4 bits per weight,
symmetric, one scale per 32 values along the input dim, zero point fixed at 8.
That is exactly what ORT calls MatMulNBits(g=32, symmetric), so the layout is
not invented here - but the artifact is a flat .bin plus a manifest, because
the consumer is our own WebGPU kernel and a kernel wants weights, not a graph.

Why not export ONNX first and let ORT quantize it: the export path is known to
work (k2i wrote 741 MB of fp16 graph), but the browser will not run that graph.
Quantizing into a format we then have to parse back out adds a step whose only
product is a validation number we can compute directly here.

The validation that matters is the one measured here: run the same real
sampling states through the fp32 model and through the same module with the
packed bytes put back. Cosine and rel-l2 on the output, per channel. Not on
randn - a 4-bit model on noise is a different question than a 4-bit model on
the state the sampler actually reaches.

Writes k2m/int4.bin, k2m/manifest.json, k2m/report.json.

Env:
  FLIP_STUDENT      student checkpoint (default /kaggle/working/k2e/student_best.pt)
  FLIP_INT4_BLOCK   weights per scale (default 32)
  FLIP_SNAP         how many real states to validate on (default 2)
"""

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k2m"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STUDENT = Path(os.environ.get("FLIP_STUDENT", str(WORK / "k2e/student_best.pt")))
BLOCK = int(os.environ.get("FLIP_INT4_BLOCK", "32"))
N_SNAP = int(os.environ.get("FLIP_SNAP", "2"))
ALIGN = 256
R = {"notes": [], "tensors": [], "validate": [], "sizes": {}}


def note(msg=""):
    line = str(msg)
    print(line, flush=True)
    R["notes"].append(line)


def save(name="report.json"):
    (OUT / name).write_text(json.dumps(R, indent=2, default=float))


# ---------------------------------------------------------------------------
# the format. Pure numpy, so the same bytes can be checked on a laptop without
# a GPU: quantize -> pack -> unpack -> dequantize is the identity on the
# reconstructed values, and anything that is not is a bug in the reader.
# ---------------------------------------------------------------------------

def quantize(w):
    """[N, K] float32 -> packed nibbles [N, K/2] uint8, scales [N, K/BLOCK] f16.

    Symmetric inside each block: scale = max|w| / 7, so the grid runs from
    -8 * scale to 7 * scale and the stored byte is q + 8. The +8 is the zero
    point every MatMulNBits consumer expects; without it the nibble is signed
    and half the values decode as their own negation.
    """
    n, k = w.shape
    if k % BLOCK:
        raise ValueError(f"K={k} is not a multiple of {BLOCK}")
    blocks = w.reshape(n, k // BLOCK, BLOCK)
    scale = np.abs(blocks).max(axis=2, keepdims=True).astype(np.float32)
    # An all-zero block would divide by zero and then decode to NaN.
    scale = np.maximum(scale / 7.0, 1e-8)
    q = np.rint(blocks / scale).clip(-8, 7).astype(np.int8)
    u = (q + 8).astype(np.uint8)
    packed = (u[:, :, 0::2] | (u[:, :, 1::2] << 4)).astype(np.uint8)
    return (packed.reshape(n, k // 2),
            scale.reshape(n, k // BLOCK).astype(np.float16))


def dequantize(packed, scales, k):
    """Inverse of quantize(). Even k in the low nibble, odd k in the high one."""
    n = packed.shape[0]
    p = packed.reshape(n, k // BLOCK, BLOCK // 2)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    u = np.stack([lo, hi], axis=-1).reshape(n, k // BLOCK, BLOCK)
    q = u.astype(np.int16) - 8
    w = q.astype(np.float32) * scales.astype(np.float32)[:, :, None]
    return w.reshape(n, k)


# ---------------------------------------------------------------------------
# getting real states to validate on
# ---------------------------------------------------------------------------

class _Bail(Exception):
    pass


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

    dino_p, vae_p = pick("dino"), pick("flux")
    dec_p = pick("decoder") or pick("octree")
    rmbg_p = pick("biref") or pick("rmbg")
    taken = {p for p in (dino_p, vae_p, rmbg_p, dec_p) if p}
    ck = [f for f in files if f not in taken][0]
    return T.TripoSplatPipeline(
        ckpt_path=str(ck), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))


def pick_image():
    root = os.environ.get("FLIP_IMAGE_DIR", "").strip()
    exts = (".png", ".jpg", ".jpeg", ".webp")
    if root:
        p = Path(root)
        found = ([p] if p.is_file() else
                 sorted(q for q in p.rglob("*") if q.suffix.lower() in exts))
        if found:
            return found[0]
    ex = REPO / "static" / "example_inputs"
    cands = sorted(q for q in ex.glob("*") if q.suffix.lower() in exts)
    return cands[0]


def capture(pipe):
    """The states a 2-step student actually sees: t=1000 and t=500."""
    cls = type(pipe.flow_model)
    orig = cls.forward
    seen = []

    def spy(self, state, t, condf, *a, **k):
        if len(seen) < N_SNAP:
            seen.append(({kk: v.detach().clone() for kk, v in state.items()},
                         t.detach().clone() if torch.is_tensor(t) else t,
                         {kk: v.detach().clone() for kk, v in condf.items()}))
        if len(seen) >= N_SNAP:
            raise _Bail()
        return orig(self, state, t, condf, *a, **k)

    cls.forward = spy
    try:
        from PIL import Image  # noqa: PLC0415
        img = Image.open(pick_image()).convert("RGB")
        cond = pipe.encode_image(pipe.preprocess_image(img))
        pipe.sample_latent(cond, steps=2, guidance_scale=1.0, shift=1.0)
    except _Bail:
        pass
    finally:
        cls.forward = orig
    return seen


# ---------------------------------------------------------------------------

def main():
    note("=" * 72)
    note("K2m int4 pack")
    note("=" * 72)
    note(f"student {STUDENT}  exists {STUDENT.exists()}")
    if not STUDENT.exists():
        note("no student checkpoint - nothing to pack")
        save()
        return

    blob = torch.load(str(STUDENT), map_location="cpu")
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    note(f"student tensors {len(sd)}  "
         f"{sum(v.numel() for v in sd.values()) / 1e6:.1f} M params")

    pipe = build_pipeline()
    pipe.flow_model.load_state_dict(sd)
    pipe.flow_model.to(DEV).float()
    snaps = capture(pipe)
    note(f"captured {len(snaps)} real states")
    # keep the DiT, drop the encoders: the comparison needs the DiT twice and
    # nothing else, and T4 has 15.6 GB for both copies plus activations.
    model = pipe.flow_model
    pipe.flow_model = None
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    def run(m):
        outs = []
        for state, t, condf in snaps:
            state = {k: v.to(DEV) for k, v in state.items()}
            condf = {k: v.to(DEV) for k, v in condf.items()}
            with torch.inference_mode():
                outs.append({k: v.float().cpu()
                             for k, v in m(state, t, condf).items()})
            torch.cuda.empty_cache()
        return outs

    note("fp32 reference pass")
    outs32 = run(model)

    # ---- quantize every Linear, in place --------------------------------
    t0 = time.time()
    packed_by_name, n_int4, n_kept, kept_note = {}, 0, 0, []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        w = mod.weight.detach().float().cpu().numpy()
        if w.shape[1] % BLOCK:
            n_kept += w.size
            kept_note.append(f"{name} K={w.shape[1]}")
            continue
        p, s = quantize(w)
        back = dequantize(p, s, w.shape[1])
        rel = float(np.linalg.norm(back - w) / (np.linalg.norm(w) + 1e-12))
        packed_by_name[name] = (p, s)
        n_int4 += w.size
        R["tensors"].append({"name": name, "n": int(w.shape[0]),
                             "k": int(w.shape[1]), "q_bytes": int(p.nbytes),
                             "s_bytes": int(s.nbytes), "rel_l2": round(rel, 5)})
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(back).to(mod.weight.device,
                                                      mod.weight.dtype))
    note(f"quantized {len(packed_by_name)} Linear layers, {n_int4 / 1e6:.1f} M "
         f"weights, {time.time() - t0:.1f}s")
    if kept_note:
        note(f"kept fp16 (K not a multiple of {BLOCK}): {kept_note}")
    rels = sorted(t["rel_l2"] for t in R["tensors"])
    if rels:
        note(f"per-tensor rel-l2  min {rels[0]:.4f}  median "
             f"{rels[len(rels) // 2]:.4f}  max {rels[-1]:.4f}")

    note("int4 pass")
    outs4 = run(model)
    for i, (a, b) in enumerate(zip(outs4, outs32)):
        t_val = float(torch.as_tensor(snaps[i][1]).flatten()[0])
        row = {"state": i, "t": t_val, "ch": {}}
        for ch, ref in b.items():
            g = a[ch].flatten()
            r = ref.flatten()
            row["ch"][ch] = {
                "cos": round(torch.nn.functional.cosine_similarity(g, r, dim=0).item(), 6),
                "rel_l2": round(((g - r).norm() / (r.norm() + 1e-12)).item(), 5),
                "max_abs": round((g - r).abs().max().item(), 5),
                "ref_max": round(r.abs().max().item(), 3),
            }
        R["validate"].append(row)
        note(f"  state {i} t={row['t']:.0f}: " + "  ".join(
            f"{ch} cos {m['cos']:.6f} rel-l2 {m['rel_l2']:.5f} "
            f"max|d| {m['max_abs']}" for ch, m in row["ch"].items()))

    del model, outs32, outs4
    gc.collect()
    torch.cuda.empty_cache()

    # ---- write the bundle -------------------------------------------------
    note("-" * 72)
    bin_path = OUT / "int4.bin"
    entries = []
    offset = 0
    packed_bases = set(packed_by_name)
    packed_weights = {f"{b}.weight" for b in packed_bases}

    with open(bin_path, "wb") as fh:
        def write(arr):
            nonlocal offset
            fill = (-offset) % ALIGN
            if fill:
                fh.write(b"\0" * fill)
                offset += fill
            start = offset
            fh.write(arr.tobytes())
            offset += arr.nbytes
            return start

        for name, (p, s) in sorted(packed_by_name.items()):
            off_q = write(p)
            off_s = write(s)
            entries.append({"name": name, "kind": "int4",
                            "n": int(p.shape[0]), "k": int(p.shape[1] * 2),
                            "offset_q": off_q, "offset_s": off_s,
                            "bytes_q": int(p.nbytes), "bytes_s": int(s.nbytes)})

        # Everything that is not a packed Linear: norms, biases, the embedding
        # tables, the time embedding. 4-bit on a 5-element bias buys nothing.
        for name, buf in sorted(sd.items()):
            if name in packed_weights:
                continue
            arr = buf.detach().float().cpu().numpy().astype(np.float16)
            off = write(arr)
            entries.append({"name": name, "kind": "fp16",
                            "shape": list(arr.shape), "offset": off,
                            "bytes": int(arr.nbytes)})

    total = bin_path.stat().st_size
    R["sizes"] = {
        "bin_mb": round(total / 1e6, 1),
        "int4_weights_m": round(n_int4 / 1e6, 1),
        "bits_per_weight": round(total * 8 / max(n_int4 + n_kept, 1), 2),
        "entries": len(entries),
    }
    note(f"int4.bin {total / 1e6:.1f} MB  "
         f"{R['sizes']['bits_per_weight']} bits/weight over the packed part")

    manifest = {
        "model": "triposplat-flow-dit-student",
        "tier": "int4",
        "bits": 4,
        "block_size": BLOCK,
        "symmetric": True,
        "zero_point": 8,
        "layout": "nibbles packed along K, even k in the low nibble, "
                  "scales float16 per block",
        "file": "int4.bin",
        "bytes": total,
        "tensors": entries,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    save()
    note("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        note("FAILED")
        note(traceback.format_exc()[-2500:])
        save()
        raise
