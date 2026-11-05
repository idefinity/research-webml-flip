#!/usr/bin/env python3
"""Rebuild a block_size=128 MatMulNBits graph at block_size=32 without
requantizing.

The 4-bit codes are packed linearly along K: a 128-wide block is exactly four
32-wide blocks in the same bytes, so B only changes shape. The scale that
covered 128 elements is repeated four times. Dequantized weights are
identical; only the scales tensor grows 4x.

Why: ORT-web 1.31.1 has a fast WebGPU path for b32 (789 GFLOP/s measured on
these shapes) and a much slower one for b128 - the k12 b128 graph runs
62 s/step where the b32 k2n graph ran 25 s/step at the same FLOPs.
"""

import argparse
import sys

import numpy as np
import onnx
from onnx import checker, numpy_helper

OLD = 128
NEW = 32
FACTOR = OLD // NEW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    model = onnx.load(args.src, load_external_data=False)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    out = dict(inits)
    done_b = set()
    done_s = set()
    n = 0

    for node in g.node:
        if node.op_type != 'MatMulNBits':
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get('block_size') != OLD:
            continue
        K = attrs['K']
        if K % OLD:
            raise SystemExit(f'{node.name}: K={K} not divisible by {OLD}')
        bname, sname = node.input[1], node.input[2]
        N = attrs['N']

        if bname not in done_b:
            b = inits[bname]
            if list(b.dims) != [N, K // OLD, OLD // 2]:
                raise SystemExit(f'{bname}: unexpected dims {list(b.dims)}')
            nb = onnx.TensorProto()
            nb.CopyFrom(b)
            del nb.dims[:]
            nb.dims.extend([N, K // NEW, NEW // 2])
            out[bname] = nb
            done_b.add(bname)

        if sname not in done_s:
            arr = numpy_helper.to_array(inits[sname])
            if arr.shape != (N, K // OLD):
                raise SystemExit(f'{sname}: unexpected shape {arr.shape}')
            out[sname] = numpy_helper.from_array(
                np.repeat(arr, FACTOR, axis=1), sname)
            done_s.add(sname)

        for a in node.attribute:
            if a.name == 'block_size':
                a.i = NEW
        n += 1

    del g.initializer[:]
    g.initializer.extend(out.values())
    checker.check_model(model)
    onnx.save(model, args.out)
    print(f'{args.out}: converted {n} nodes · B resized {len(done_b)} · '
          f'scales 4x {len(done_s)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
