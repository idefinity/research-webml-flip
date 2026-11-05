"""Rebuild the fused DiT graph around the published int4 pack (k2n).

The quantiser that produced dit.int4.onnx and the packer that produced
assets/k2n made two different models:

  - MatMulNBitsQuantizer quantises every MatMul it finds, including
    input_layer (K=16) and cam_refiner.mlp.0 (K=5). One 4-bit block spanning
    16 channels costs 6.7% of the signal at the very first layer.
  - nn.Linear on a 2-D input exports as Gemm, which the quantiser leaves
    alone - so t_embedder and adaLN_modulation stayed fp32 while the weight
    file quantised them. Feeding the same noise to the two artifacts gave
    cos 0.9992 on t_embedder's output, enough to bend every modulated block.

This script takes the fused graph (k2v) and the pack (k2m) and makes one model:

  1. MatMulNBits whose module is fp16 in the pack (K not a multiple of 32)
     goes back to a plain fp32 MatMul with the pack's values;
  2. every other MatMulNBits keeps its structure and takes the pack's nibbles
     and scales;
  3. every `f.<state_dict key>` initializer - norms, biases, the three Gemms -
     is overwritten with the pack's tensor.

After this the graph and model.safetensors are the same function, so the k2u
fixtures (generated from the pack) are a valid reference for the browser.

Usage:
    python3 scripts/k2w_onnx_pack.py [--src w/dit.mha.onnx] [--out w/dit.k2n.onnx]
                                     [--pack assets/k2n/model.safetensors]
"""
import argparse
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors import safe_open

BLOCK = 32


def dequantize(packed, scales):
    """[N, K/2] uint8 + [N, K/32] scales -> [N, K] fp32. Even k in the low nibble."""
    n, half = packed.shape
    k = half * 2
    p = packed.reshape(n, k // BLOCK, BLOCK // 2)
    u = np.stack([p & 0x0F, (p >> 4) & 0x0F], axis=-1).reshape(n, k // BLOCK, BLOCK)
    return ((u.astype(np.int16) - 8).astype(np.float32)
            * scales.astype(np.float32)[:, :, None]).reshape(n, k)


def load_pack(path):
    """Everything the graph may need, keyed by state_dict name."""
    int4, other = {}, {}
    with safe_open(path, framework="numpy") as f:
        keys = list(f.keys())
        for k in keys:
            if not k.endswith(".weight"):
                continue
            w = f.get_tensor(k)
            if w.dtype != np.uint8:
                continue
            wq = dequantize(w, f.get_tensor(k + "_scale"))
            int4[k[: -len(".weight")]] = (w, f.get_tensor(k + "_scale"), wq.shape)
            other[k] = wq
        for k in keys:
            if k.endswith("_scale") or k in other:
                continue
            t = f.get_tensor(k)
            if t.dtype == np.uint8:
                continue
            other[k] = t.astype(np.float32)
    return int4, other


def module_of(node_name):
    """`/f/blocks.0/attn/qkv/MatMul_Q4` -> `blocks.0.attn.qkv`."""
    assert node_name.startswith("/f/") and node_name.endswith("/MatMul_Q4"), node_name
    return node_name[len("/f/"):-len("/MatMul_Q4")].replace("/", ".")


def resolve(mod):
    """The exporter repeats a sequential's scope once more than the state dict
    does: /blocks.0/mlp/mlp/mlp.0 is blocks.0.mlp.mlp.0, and /cam_refiner/mlp/
    mlp.0 is cam_refiner.mlp.0. Try the literal path and every one-segment
    deletion; the shape checks decide which is real."""
    segs = mod.split(".")
    out = [mod]
    for i in range(len(segs)):
        out.append(".".join(segs[:i] + segs[i + 1:]))
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="w/dit.mha.onnx")
    ap.add_argument("--out", default="w/dit.k2n.onnx")
    ap.add_argument("--pack", default="assets/k2n/model.safetensors")
    args = ap.parse_args()

    int4, other = load_pack(args.pack)
    print(f"pack: {len(int4)} int4 modules, {len(other)} tensors")

    m = onnx.load(args.src, load_external_data=False)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    used, demoted, dead = set(), [], set()

    for n in g.node:
        if n.op_type != "MatMulNBits":
            continue
        names = resolve(module_of(n.name))
        mod = next((c for c in names if c in int4), None)
        if mod is not None:
            q, sc, shape = int4[mod]
            qi, si = inits[n.input[1]], inits[n.input[2]]
            n_blocks = shape[1] // BLOCK
            assert list(qi.dims) == [shape[0], n_blocks, BLOCK // 2], (mod, qi.dims)
            assert list(si.dims) == [shape[0], n_blocks], (mod, si.dims)
            qi.raw_data = q.reshape(shape[0], n_blocks, BLOCK // 2).tobytes()
            si.data_type = onnx.TensorProto.FLOAT
            si.ClearField("float_data")
            si.raw_data = sc.astype(np.float32).tobytes()
            used.add(mod)
        else:
            mod = next(c for c in names if f"{c}.weight" in other)
            w = other[f"{mod}.weight"]
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            assert w.shape == (attrs["N"], attrs["K"]), (mod, w.shape, attrs)
            assert w.shape[1] % BLOCK, (mod, w.shape)  # only sub-block K gets here
            name = f"f.{mod}.weight"
            assert name not in inits, name
            g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(w.T), name))
            dead.update(n.input[1:3])
            n.op_type, n.domain = "MatMul", ""
            n.input[1] = name
            del n.input[2:]
            del n.attribute[:]
            demoted.append(mod)

    gemm = {t.name[2:-len(".weight")] for t in g.initializer
            if t.name.startswith("f.") and t.name.endswith(".weight")}
    missing = sorted(set(int4) - used - gemm)
    assert not missing, f"int4 modules with no node: {missing}"

    overlaid = 0
    for t in g.initializer:
        if not t.name.startswith("f."):
            continue
        key = t.name[2:]
        if key not in other:
            continue
        if any(t.name == f"f.{d}.weight" for d in demoted):
            continue  # already carries the pack's values, transposed for MatMul
        w = np.ascontiguousarray(other[key])
        assert list(t.dims) == list(w.shape), (t.name, t.dims, w.shape)
        t.data_type = onnx.TensorProto.FLOAT
        t.ClearField("float_data")
        t.ClearField("int32_data")
        t.raw_data = w.tobytes()
        overlaid += 1

    keep = [t for t in g.initializer if t.name not in dead]
    del g.initializer[:]
    g.initializer.extend(keep)

    onnx.save(m, args.out)
    print(f"int4 swapped {len(used)}  demoted {demoted}  overlaid {overlaid}")
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
