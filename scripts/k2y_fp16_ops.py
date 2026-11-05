#!/usr/bin/env python3
"""Cast only across the heavy ops so they can run in fp16; everything between
them stays fp32.

The graph is fp32 and its 374 Cast nodes pin tensors to fp32 in places the
exporter chose on purpose (the RMSNorm path upcasts, rope is real arithmetic,
the timestep embedding multiplies 1000-scale angles). Flipping every
initializer to fp16 would fight those casts. Instead:

  MultiHeadAttention   q,k,v -> fp16, out -> fp32
  MatMulNBits          A -> fp16, scales -> fp16, out -> fp32

That is the q4f16 split PUBLISH.md describes: 4-bit weights, fp16 matmul,
norm / rope / softmax built from fp32 ops. The cast traffic is ~35 GB/step
on this graph, under 0.1 s at Metal bandwidth, against ~18 s of fp32 math.

usage:
  k2y_fp16_ops.py w/dit.k12.int4.mha.onnx --mma --out w/dit.k12.int4.mha16.onnx
  k2y_fp16_ops.py w/dit.k12.int4.mha.onnx --nbits --out w/dit.k12.int4.q4f16.onnx
  k2y_fp16_ops.py w/dit.k12.int4.mha.onnx --mma --nbits --out ...
"""

import argparse
import sys

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper


def sanitize(name):
    return name.strip('/').replace('/', '_').replace('.', '_')


def wrap(model, do_mma, do_nbits):
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    out_inits = dict(inits)
    converted_scales = set()

    taken = {n.name for n in g.node if n.name}
    taken |= {o for n in g.node for o in n.output}
    taken |= set(inits)
    taken |= {i.name for i in g.input}

    def fresh(base):
        i = 0
        while True:
            cand = f'f16/{base}' + (f'_{i}' if i else '')
            if cand not in taken:
                taken.add(cand)
                return cand
            i += 1

    nodes = []
    n_mma = n_nbits = n_scale = 0

    for node in g.node:
        if do_mma and node.op_type == 'MultiHeadAttention':
            new_in = []
            for x in node.input:
                c = fresh('cast/' + sanitize(x))
                nodes.append(helper.make_node(
                    'Cast', [x], [c], to=TensorProto.FLOAT16,
                    name=fresh('cast_in')))
                new_in.append(c)
            orig = node.output[0]
            mid = fresh('mha/' + sanitize(orig))
            node.input[:] = new_in
            node.output[0] = mid
            nodes.append(node)
            nodes.append(helper.make_node(
                'Cast', [mid], [orig], to=TensorProto.FLOAT,
                name=fresh('cast_out')))
            n_mma += 1
            continue

        if do_nbits and node.op_type == 'MatMulNBits':
            a = node.input[0]
            ca = fresh('cast/' + sanitize(a))
            nodes.append(helper.make_node(
                'Cast', [a], [ca], to=TensorProto.FLOAT16,
                name=fresh('cast_in')))
            sname = node.input[2]
            if sname not in inits:
                raise SystemExit(f'scales {sname} is not an initializer')
            if sname not in converted_scales:
                arr = numpy_helper.to_array(inits[sname])
                if arr.dtype != np.float32:
                    raise SystemExit(f'{sname}: unexpected {arr.dtype}')
                out_inits[sname] = numpy_helper.from_array(
                    arr.astype(np.float16), sname)
                converted_scales.add(sname)
                n_scale += 1
            orig = node.output[0]
            mid = fresh('mm/' + sanitize(orig))
            node.input[0] = ca
            node.output[0] = mid
            nodes.append(node)
            nodes.append(helper.make_node(
                'Cast', [mid], [orig], to=TensorProto.FLOAT,
                name=fresh('cast_out')))
            n_nbits += 1
            continue

        nodes.append(node)

    del g.node[:]
    g.node.extend(nodes)
    del g.initializer[:]
    g.initializer.extend(out_inits.values())
    return n_mma, n_nbits, n_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mma', action='store_true')
    ap.add_argument('--nbits', action='store_true')
    args = ap.parse_args()
    if not (args.mma or args.nbits):
        raise SystemExit('pick --mma and/or --nbits')

    model = onnx.load(args.src, load_external_data=False)
    checker.check_model(model)
    before = len(model.graph.node)

    n_mma, n_nbits, n_scale = wrap(model, args.mma, args.nbits)
    checker.check_model(model)
    onnx.save(model, args.out)

    dt = model.graph.output[0].type.tensor_type.elem_type
    print(f'{args.out}: {before} -> {len(model.graph.node)} nodes '
          f'(+{len(model.graph.node) - before})')
    print(f'  mha wrapped {n_mma} · nbits wrapped {n_nbits} · '
          f'scales fp16 {n_scale} · first output dtype {dt}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
