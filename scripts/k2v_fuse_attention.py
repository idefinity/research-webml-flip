"""K2v - replace the 28 materialised attentions with com.microsoft::MultiHeadAttention.

The DiT graph exports attention as QK^T -> Softmax -> PV. That materialises
[1,16,L,L] fp32 - 9.66 GB at L=12294 - and ORT-web dies on the SafeInt overflow
before it can allocate. Measured in the browser (webgpu/onnx/mha-probe.html):
com.microsoft::MultiHeadAttention with Q/K/V already projected runs the same
attention at 867 GFLOP/s, and the wasm EP fails with the same overflow, so the
number is not a silent fallback.

The graph already has Q/K/V projected, RoPE'd and qk-RMS-normed where the
attention starts, so no weights move. For each of the 28 Softmax nodes:

    q_ready [1,L,16,64] ─ Reshape ─┐
    k_ready [1,L,16,64] ─ Reshape ─┼─ MultiHeadAttention(num_heads=16, scale=0.125) ─> [1,L,1024] -> out proj
    v_ready [1,L,16,64] ─ Reshape ─┘

The old per-head transposes, the sqrt-scale split across Q and K, QK^T, Softmax,
PV and the transpose back are all removed. The scale moves from two Muls into
the op's `scale` attribute - same arithmetic, `(q*sqrt(s)) . (k*sqrt(s))^T`.

Requires onnx (`pip install onnx`). Local graph surgery only; weights unchanged.

    python scripts/k2v_fuse_attention.py w/dit.int4.onnx w/dit.mha.onnx
"""

import sys
from collections import defaultdict

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

NUM_HEADS = 16


def consumers(graph):
    c = defaultdict(list)
    for n in graph.node:
        for x in n.input:
            if x:
                c[x].append(n)
    return c


def const_value(graph, name):
    state = {i.name: i for i in graph.initializer}
    if name in state:
        return numpy_helper.to_array(state[name])
    for n in graph.node:
        if n.op_type == "Constant" and n.output[0] == name:
            return numpy_helper.to_array(n.attribute[0].t)
    return None


def producer_map(graph):
    return {o: n for n in graph.node for o in n.output}


def nearest_reshape_shape(graph, name, depth=14):
    """Walk back to the closest Reshape whose shape is a constant."""
    prod = producer_map(graph)
    cur = name
    for _ in range(depth):
        n = prod.get(cur)
        if n is None:
            return None
        if n.op_type == "Reshape":
            v = const_value(graph, n.input[1])
            if v is not None:
                return [int(x) for x in v]
        if n.op_type in ("Cast", "Mul", "Div", "Squeeze", "Transpose", "Add", "Sub",
                         "Expand", "Clip", "ReduceL2", "Sqrt", "Neg"):
            cur = n.input[0]
            continue
        return None
    return None


def value_shape_ok(graph, name, L):
    """q/k are [1,L,16,64]; v is Squeeze([1,L,3,16,64]) -> [1,L,16,64]."""
    if nearest_reshape_shape(graph, name) == [1, L, NUM_HEADS, 64]:
        return True
    prod = producer_map(graph)
    n = prod.get(name)
    if n is None or n.op_type != "Squeeze":
        return False
    src = n.input[0]
    p = prod.get(src)
    if p is not None and p.op_type == "Split":
        src = p.input[0]
    return nearest_reshape_shape(graph, src) == [1, L, 3, NUM_HEADS, 64]


def transpose_perm(node):
    for a in node.attribute:
        if a.name == "perm":
            return list(a.ints)
    return None


def back_to_transpose(graph, name):
    """From a QK^T input, skip the sqrt-scale Mul, stop at the head Transpose."""
    prod = producer_map(graph)
    cur = name
    for _ in range(6):
        n = prod.get(cur)
        if n is None:
            return None
        if n.op_type == "Transpose":
            return n
        if n.op_type in ("Mul", "Div", "Cast"):
            cur = n.input[0]
            continue
        return None
    return None


def topo_sort(nodes, graph_inputs, initializers):
    available = set(graph_inputs) | set(initializers)
    pending = list(nodes)
    out = []
    while pending:
        progressed = False
        rest = []
        for n in pending:
            if all((x in available or not x) for x in n.input):
                out.append(n)
                available.update(n.output)
                progressed = True
            else:
                rest.append(n)
        if not progressed:
            detail = []
            for n in rest[:5]:
                missing = [x for x in n.input if x and x not in available]
                detail.append(f"{n.name} missing {missing}")
            raise RuntimeError("cycle or missing producer: " + " | ".join(detail))
        pending = rest
    return out


def fuse(src, dst):
    model = onnx.load(src)
    g = model.graph
    prod = producer_map(g)
    cons = consumers(g)

    softmaxes = [n for n in g.node if n.op_type == "Softmax"]
    print(f"{src}: {len(softmaxes)} softmax nodes")

    remove = set()
    new_nodes = []
    shapes = {}

    for sm in softmaxes:
        qk = prod[sm.input[0]]
        pv = next(n for n in cons[sm.output[0]] if n.op_type == "MatMul")
        if qk.op_type != "MatMul" or len(pv.input) != 2:
            raise RuntimeError(f"unexpected pattern at {sm.name}")

        q_t = back_to_transpose(g, qk.input[0])
        k_t = back_to_transpose(g, qk.input[1])
        v_t = prod[pv.input[1]]
        if not (q_t and k_t and v_t and v_t.op_type == "Transpose"):
            raise RuntimeError(f"cannot find head transposes at {sm.name}")
        if transpose_perm(q_t) != [0, 2, 1, 3] or transpose_perm(v_t) != [0, 2, 1, 3]:
            raise RuntimeError(f"q/v perm changed at {sm.name}: {transpose_perm(q_t)}")
        if transpose_perm(k_t) != [0, 2, 3, 1]:
            raise RuntimeError(f"k perm changed at {sm.name}: {transpose_perm(k_t)}")

        q_ready, k_ready, v_ready = q_t.input[0], k_t.input[0], v_t.input[0]

        out_t = next(n for n in cons[pv.output[0]] if n.op_type == "Transpose")
        out_reshape = next(n for n in cons[out_t.output[0]] if n.op_type == "Reshape")
        shape = [int(x) for x in const_value(g, out_reshape.input[1])]
        if len(shape) != 3 or shape[0] != 1 or shape[2] != NUM_HEADS * 64:
            raise RuntimeError(f"unexpected out reshape at {sm.name}: {shape}")
        L = shape[1]

        for name, ready in (("q", q_ready), ("k", k_ready), ("v", v_ready)):
            if not value_shape_ok(g, ready, L):
                raise RuntimeError(f"{name} layout at {sm.name}: {nearest_reshape_shape(g, ready)},"
                                   f" want [1, {L}, {NUM_HEADS}, 64]")
        shapes.setdefault(L, 0)
        shapes[L] += 1

        module = sm.name.rsplit("/", 1)[0]
        shape_name = f"mha_shape_{L}"
        new_nodes.append(helper.make_node(
            "Reshape", [q_ready, shape_name], [f"{module}/mha_q3d"], name=f"{module}/MHA_Q3D"))
        new_nodes.append(helper.make_node(
            "Reshape", [k_ready, shape_name], [f"{module}/mha_k3d"], name=f"{module}/MHA_K3D"))
        new_nodes.append(helper.make_node(
            "Reshape", [v_ready, shape_name], [f"{module}/mha_v3d"], name=f"{module}/MHA_V3D"))
        new_nodes.append(helper.make_node(
            "MultiHeadAttention",
            [f"{module}/mha_q3d", f"{module}/mha_k3d", f"{module}/mha_v3d"],
            [out_reshape.output[0]],
            name=f"{module}/MultiHeadAttention",
            domain="com.microsoft",
            num_heads=NUM_HEADS,
            scale=1.0 / np.sqrt(64),
        ))

        remove.update({id(qk), id(sm), id(pv), id(q_t), id(k_t), id(v_t), id(out_t), id(out_reshape)})

    kept = [n for n in g.node if id(n) not in remove]

    # Dead-code elimination: one reverse pass is enough because the graph is
    # topologically sorted and consumers always come after producers.
    # Seed it with the new nodes' outputs too, or the whole QKV/RoPE chain
    # that feeds them gets pruned.
    needed = {o.name for o in g.output}
    needed.update(o for n in new_nodes for o in n.output)
    needed.update(x for n in new_nodes for x in n.input if x)
    for n in reversed(kept):
        if any(o in needed for o in n.output):
            needed.update(x for x in n.input if x)

    kept = [n for n in kept if any(o in needed for o in n.output)]

    # One int64 [1,L,1024] shape initializer per distinct sequence length.
    existing = {i.name for i in g.initializer}
    for L in shapes:
        name = f"mha_shape_{L}"
        if name not in existing:
            g.initializer.append(numpy_helper.from_array(
                np.array([1, L, NUM_HEADS * 64], dtype=np.int64), name))

    mha_nodes = [n for n in new_nodes if n.op_type == "MultiHeadAttention"]
    if len(mha_nodes) != len(softmaxes):
        raise RuntimeError(f"built {len(mha_nodes)} MHA nodes for {len(softmaxes)} attentions")
    print(f"  new MHA nodes {len(mha_nodes)} · first out {mha_nodes[0].output[0]}")

    nodes = topo_sort(kept + new_nodes, [i.name for i in g.input], [i.name for i in g.initializer])

    del g.node[:]
    g.node.extend(nodes)

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst)

    # report
    kept_ops = defaultdict(int)
    for n in g.node:
        kept_ops[(n.domain, n.op_type)] += 1
    print(f"  shapes (L -> attention count): {shapes}")
    print(f"  nodes {len(nodes)} · MHA {kept_ops[('com.microsoft', 'MultiHeadAttention')]}"
          f" · Softmax {kept_ops[('', 'Softmax')]} · MatMul {kept_ops[('', 'MatMul')]}"
          f" · MatMulNBits {kept_ops[('com.microsoft', 'MatMulNBits')]}")
    print(f"  wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    fuse(sys.argv[1], sys.argv[2])
