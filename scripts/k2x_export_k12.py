"""K2x - the k12 4-step student as an int4 ONNX graph for ORT-web.

Same shape as k2p -> k2v -> k2w, with the quantiser written out because the
ORT MatMulNBitsQuantizer cannot do the three things this graph needs:

  - a block size other than 32. b128 measured 4/4 on the gate and saves 34 MB
    of fp32 scales over the shipped 294 MB graph;
  - the MSE-optimal per-block alpha the sweep (q1) measured against plain
    amax/qmax;
  - leaving the K=16 / K=5 matmuls alone. The quantizer pads them into a
    block and the padding costs 6.7% of the signal at the first layer, so
    here a MatMul is converted only when K >= 32, K is a multiple of the
    block, and the module is not one of the three 2-D Gemms (t_embedder x2,
    adaLN_modulation.1).

Optionally adaLN's Gemm is folded to MatMulNBits + Add: 6144x1024 is 25 MB
fp32, and q1's e4 job (4-bit on the three 2-D layers) still passed 4/4.

Then k2v fuses the 28 attentions into com.microsoft::MultiHeadAttention - at
L=16394 the materialised [1,16,L,L] is 17 GB and ORT-web dies on it.

Parity is measured against PyTorch, not against the fp32 graph: running the
unfused fp32 graph on CPU needs the same 17 GB attention. The reference is
the student itself on real trace states, and the numbers that matter are
cos/rel on the velocity at two points of the real trajectory.

Env:
  X_BITS=4 X_BLOCK=128 X_ADA=1 X_MHA=1 X_PARITY=1 X_WORK=/content
  X_OUT=/content/x1  X_STUDENT=/content/k12_assets/student_e006.pt
Writes x1/dit.int4.onnx (MHA fused, ada folded), x1/report.json, x1/log.txt.
"""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("X_WORK", "/content"))
REPO = WORK / "TripoSplat"
ASSETS = WORK / "k12_assets"
OUT = Path(os.environ.get("X_OUT", str(WORK / "x1")))
OUT.mkdir(exist_ok=True)
STUDENT = Path(os.environ.get("X_STUDENT", str(ASSETS / "student_e006.pt")))

BITS = int(os.environ.get("X_BITS", "4"))
BLOCK = int(os.environ.get("X_BLOCK", "128"))
ADA = os.environ.get("X_ADA", "1") == "1"
MHA = os.environ.get("X_MHA", "1") == "1"
PARITY = os.environ.get("X_PARITY", "1") == "1"

ALPHAS = [round(0.60 + 0.025 * i, 4) for i in range(17)]
SKIP_2D = ("t_embedder", "adaLN", "l_embedder")

LOG = []


def p(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)
    try:
        (OUT / "log.txt").write_text("\n".join(LOG) + "\n")
    except Exception:
        pass


def pack_tensor(w, block):
    """torch [N,K] fp32 on any device -> (nibbles [N,nb,block//2] uint8,
    scales [N,nb] fp32). Matches the fake-quant q1 measured: symmetric, no
    zero point stored (q+8), per-block MSE-optimal alpha."""
    import torch
    assert BITS == 4, "the ORT blob layout here is 4-bit only"
    n, k = w.shape
    nb = k // block
    b = w.reshape(n, nb, block)
    amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    qmax, qmin = 7, -8
    best = torch.full_like(amax, float("inf"))
    best_s = torch.zeros_like(amax)
    for a in ALPHAS:
        s = amax * a / qmax
        e = ((b - (b / s).round().clamp(qmin, qmax) * s) ** 2).sum(-1, keepdim=True)
        m = e < best
        best = torch.where(m, e, best)
        best_s = torch.where(m, s, best_s)
    q = (b / best_s).round().clamp(qmin, qmax).to(torch.int8)
    u = (q + 8).to(torch.uint8)
    packed = (u[:, :, 0::2] | (u[:, :, 1::2] << 4)).reshape(n, nb, block // 2)
    return packed.cpu().numpy(), best_s.squeeze(-1).cpu().numpy().astype(np.float32)


def rewrite_matmuls(g, dry=None):
    """Convert weight MatMuls to MatMulNBits. Returns (converted, skipped)."""
    import onnx
    import torch
    from onnx import helper, numpy_helper

    inits = {i.name: i for i in g.initializer}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    new_nodes, conv, skipped = [], [], []
    for n in g.node:
        # Weight MatMuls are the ones whose B is a rank-2 fp32 initializer.
        # The exporter names them f.<state_dict key> (k2w relies on it too),
        # but the name is only needed for the 2-D skip list, so match on shape.
        if n.op_type != "MatMul" or len(n.input) != 2 or n.input[1] not in inits:
            new_nodes.append(n)
            continue
        t = inits[n.input[1]]
        if t.data_type != onnx.TensorProto.FLOAT or len(t.dims) != 2:
            new_nodes.append(n)
            continue
        arr = numpy_helper.to_array(t)          # [K, N] for A[...,K] @ W
        K, N = int(arr.shape[0]), int(arr.shape[1])
        mod = n.input[1][len("f."):-len(".weight")] if n.input[1].startswith("f.") else n.input[1]
        if K < 32 or K % BLOCK or any(s in mod for s in SKIP_2D):
            skipped.append([mod, K])
            new_nodes.append(n)
            continue
        wt = torch.from_numpy(np.ascontiguousarray(arr.T)).to(dev, torch.float32)
        q, s = pack_tensor(wt, BLOCK)
        del wt
        qn, sn = n.input[1] + f"_q{BITS}", n.input[1] + f"_s{BITS}"
        g.initializer.append(numpy_helper.from_array(
            q.reshape(N, K // BLOCK, BLOCK // 2), qn))
        g.initializer.append(numpy_helper.from_array(s, sn))
        new_nodes.append(helper.make_node(
            "MatMulNBits", [n.input[0], qn, sn], list(n.output),
            name=n.name + f"_Q{BITS}", domain="com.microsoft",
            K=K, N=N, bits=BITS, block_size=BLOCK))
        conv.append([mod, K, N])
    del g.node[:]
    g.node.extend(new_nodes)
    return conv, skipped


def fold_ada(g):
    """Gemm(A,W[6144,1024],bias) -> MatMulNBits + Add, same output name."""
    import onnx
    import torch
    from onnx import helper, numpy_helper

    inits = {i.name: i for i in g.initializer}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nodes, done = [], []
    for n in g.node:
        if n.op_type != "Gemm":
            nodes.append(n)
            continue
        at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        ok = (at.get("transB", 0) == 1 and at.get("transA", 0) == 0
              and at.get("alpha", 1.0) == 1.0 and at.get("beta", 1.0) == 1.0
              and n.input[1] in inits)
        if not ok:
            nodes.append(n)
            continue
        arr = numpy_helper.to_array(inits[n.input[1]])       # [N, K]
        N, K = int(arr.shape[0]), int(arr.shape[1])
        if K < 32 or K % BLOCK:
            nodes.append(n)
            continue
        wt = torch.from_numpy(np.ascontiguousarray(arr)).to(dev, torch.float32)
        q, s = pack_tensor(wt, BLOCK)
        del wt
        qn, sn = n.input[1] + f"_q{BITS}", n.input[1] + f"_s{BITS}"
        g.initializer.append(numpy_helper.from_array(
            q.reshape(N, K // BLOCK, BLOCK // 2), qn))
        g.initializer.append(numpy_helper.from_array(s, sn))
        mm = n.name + "/mm"
        nodes.append(helper.make_node(
            "MatMulNBits", [n.input[0], qn, sn], [mm],
            name=n.name + f"_Q{BITS}", domain="com.microsoft",
            K=K, N=N, bits=BITS, block_size=BLOCK))
        if len(n.input) > 2 and n.input[2]:
            nodes.append(helper.make_node("Add", [mm, n.input[2]], list(n.output),
                                          name=n.name + "/add"))
        else:
            nodes.append(helper.make_node("Identity", [mm], list(n.output),
                                          name=n.name + "/id"))
        done.append([n.input[1], K, N])
    del g.node[:]
    g.node.extend(nodes)
    return done


def drop_dead(g):
    used = set()
    for n in g.node:
        used.update(x for x in n.input if x)
    used.update(o.name for o in g.output)
    keep = [t for t in g.initializer if t.name in used]
    dropped = len(g.initializer) - len(keep)
    del g.initializer[:]
    g.initializer.extend(keep)
    return dropped


# ---------------------------------------------------------------------------

def main():
    sys.path.insert(0, str(REPO))
    import torch
    import torch.nn as nn
    import onnx
    import triposplat as T
    import model as M

    p("=" * 78)
    p("k2x export", time.strftime("%Y-%m-%d %H:%M:%S"),
      f"bits {BITS} block {BLOCK} ada {ADA} mha {MHA} parity {PARITY}")

    # ---- rope: aten::polar / view_as_complex have no ONNX symbolic at 17.
    # Same substitution k2p proved against the complex implementation.
    def _rope_real(self, hidden_states):
        h = self.norm(hidden_states)
        feat = self.act(self.gate_map(h)) * self.content_map(h)
        out = self.final_map(feat)
        B, L, _ = out.shape
        delta_pos = out.reshape(B, L, self.num_heads, 3)
        ang_0 = M.clamp_mul(delta_pos[..., 0].unsqueeze(-1), self.freqs_0) * torch.pi
        ang_1 = M.clamp_mul(delta_pos[..., 1].unsqueeze(-1), self.freqs_1) * torch.pi
        ang_2 = M.clamp_mul(delta_pos[..., 2].unsqueeze(-1), self.freqs_2) * torch.pi
        ang = torch.cat([ang_0, ang_1, ang_2], dim=-1).float()
        return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

    def _apply_rope_real(hidden_states, freqs):
        cos, sin = freqs[..., 0], freqs[..., 1]
        x = hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2)
        a, b = x[..., 0], x[..., 1]
        out = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1)
        return out.reshape(*hidden_states.shape).type_as(hidden_states)

    _orig_rope = M.RePo3DRotaryEmbedding.forward
    _orig_apply = M.apply_rotary_emb
    M.RePo3DRotaryEmbedding.forward = _rope_real
    M.apply_rotary_emb = _apply_rope_real

    torch.manual_seed(0)
    _rl = M.RePo3DRotaryEmbedding(64, 4, 16).eval()
    _h = torch.randn(1, 7, 64)
    with torch.no_grad():
        _f = _rl(_h)
        _a = _apply_rope_real(_h.reshape(1, 7, 4, 16), _f)
        _f0 = _orig_rope(_rl, _h)
        _b = _orig_apply(_h.reshape(1, 7, 4, 16), _f0)
    p("rope substitution: max |real - complex| =", float((_a - _b).abs().max()))
    del _rl, _h, _f, _a, _f0, _b

    # ---- real shapes from a trace, so the static graph is what the sampler feeds
    import importlib.util
    spec = importlib.util.spec_from_file_location("k2e", str(WORK / "k2e_distill.py"))
    k2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k2e)
    k2e.WORK = WORK
    k2e.REPO = REPO
    k2e.CKPT = REPO / "ckpts"
    tr = k2e.load_trace(sorted(ASSETS.glob("trace_*.pt"))[0])
    states, ts, final = k2e.trajectory(tr)
    call0 = k2e.split_call(tr["calls"][0])
    cond0 = call0[2]
    x0 = states[0]
    f2_trace = cond0["feature2"]
    LQ = int(x0["latent"].shape[1])
    LC1 = int(cond0["feature1"].shape[1])
    padded = int(f2_trace.shape[1]) == LC1
    LC2 = int(f2_trace.shape[1]) - (5 if padded else 0)
    p(f"shapes latent {LQ} · feature1 {LC1} · feature2 {LC2} (+5 pad traced: {padded})")

    # ---- student
    flow = M.LatentSeqMMFlowModel(**T.FLOW_MODEL_ARGS)
    blob = torch.load(str(STUDENT), map_location="cpu",
                      weights_only=True)
    sd = blob["model"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = flow.load_state_dict(sd, strict=False)
    p("loaded:", len(sd), "tensors | missing", len(missing), "| unexpected", len(unexpected))
    if missing or unexpected:
        p("  missing sample:", list(missing)[:3])
        p("  unexpected sample:", list(unexpected)[:3])
    flow.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p("device", dev)

    class Wrap(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, latent, camera, feature1, feature2, t):
            b = feature2.shape[0]
            pad = torch.zeros(b, 5, feature2.shape[2], dtype=feature2.dtype,
                              device=feature2.device)
            out = self.f({"latent": latent, "camera": camera}, t,
                         {"feature1": feature1,
                          "feature2": torch.cat([pad, feature2], dim=1)})
            return out["latent"], out["camera"]

    wrap = Wrap(flow).to(dev).eval()
    g = torch.Generator().manual_seed(1)
    dummy = (torch.randn(1, LQ, 16, generator=g),
             torch.randn(1, 1, 5, generator=g),
             torch.randn(1, LC1, 1280, generator=g),
             torch.randn(1, LC2, 128, generator=g),
             torch.tensor([500.0]))
    dummy = tuple(x.to(dev) for x in dummy)
    names = ["latent", "camera", "feature1", "feature2", "t"]

    fp32 = str(OUT / "dit.fp32.onnx")
    p("exporting fp32 (static, opset 17) ...")
    t0 = time.time()
    try:
        with torch.no_grad():
            torch.onnx.export(wrap, dummy, fp32, input_names=names,
                              output_names=["out_latent", "out_camera"],
                              opset_version=17, dynamo=False, dynamic_axes=None)
    except Exception:
        p("tracing export failed:", traceback.format_exc()[-1500:])
        p("retrying with the dynamo exporter, opset 18")
        with torch.no_grad():
            torch.onnx.export(wrap, dummy, fp32, input_names=names,
                              output_names=["out_latent", "out_camera"],
                              opset_version=18, dynamo=True, dynamic_axes=None)
    p("fp32 graph", round(os.path.getsize(fp32) / 1e6, 1), "MB in",
      round(time.time() - t0, 1), "s")

    m = onnx.load(fp32, load_external_data=False)
    g_ = m.graph
    # the exporter only declares the default domain; every com.microsoft node
    # below needs this import or the checker rejects the model
    if not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    conv, skipped = rewrite_matmuls(g_)
    p(f"converted {len(conv)} MatMuls to MatMulNBits(b{BITS}, block {BLOCK})")
    p(f"left fp32: {len(skipped)} weight MatMuls -> {skipped}")
    # the census says 201; anything under 180 means the match found the wrong
    # nodes and the "int4" graph is still the 1.5 GB fp32 one
    assert len(conv) >= 180, f"only {len(conv)} weights converted"
    if ADA:
        ada = fold_ada(g_)
        p(f"adaLN fold: {ada}")
    dropped = drop_dead(g_)
    p("dead initializers dropped", dropped)
    int4 = str(OUT / "dit.int4.onnx")
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, int4)
    del m, g_
    gc.collect()
    p("int4 graph", round(os.path.getsize(int4) / 1e6, 1), "MB")

    final_path = int4
    if MHA:
        import k2v_fuse_attention as k2v
        fused = str(OUT / "dit.int4.mha.onnx")
        p("fusing attention ...")
        k2v.fuse(int4, fused)
        final_path = fused
        p("fused graph", round(os.path.getsize(fused) / 1e6, 1), "MB")

    # ---- parity against the student itself, on the states the sampler reaches
    report = {"bits": BITS, "block": BLOCK, "ada": ADA, "mha": MHA,
              "converted": len(conv), "skipped": skipped,
              "sizes": {Path(x).name: round(os.path.getsize(x) / 1e6, 1)
                        for x in (fp32, int4, final_path)}}
    if PARITY:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(final_path, so, providers=["CPUExecutionProvider"])
        p("onnx inputs", sess.get_inputs()[0].name, "->", [o.name for o in sess.get_outputs()])

        rows = []
        with torch.no_grad():
            for ci in (0, 2):
                st, tt, cn = k2e.split_call(tr["calls"][ci])
                lat = st["latent"].float()
                cam = st["camera"].float()
                tval = float(np.asarray(tt).reshape(-1)[0])
                f1 = cn["feature1"].float()
                f2 = cn["feature2"].float()
                f2_raw = f2[:, 5:] if f2.shape[1] == f1.shape[1] else f2
                ref = flow({"latent": lat.to(dev), "camera": cam.to(dev)},
                           torch.tensor([tval], device=dev),
                           {"feature1": f1.to(dev), "feature2": f2.to(dev)})
                feeds = {
                    "latent": lat.numpy(), "camera": cam.numpy(),
                    "feature1": f1.numpy(), "feature2": f2_raw.numpy(),
                    "t": np.array([tval], dtype=np.float32)}
                outs = sess.run(None, feeds)
                row = {"call": ci, "t": tval}
                for name, onnx_val, torch_val in (
                        ("latent", outs[0], ref["latent"]),
                        ("camera", outs[1], ref["camera"])):
                    a = torch.from_numpy(onnx_val.astype(np.float64)).flatten()
                    b = torch_val.double().cpu().flatten()
                    cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
                    rel = float((a - b).norm() / (b.norm() + 1e-12))
                    row[name] = {"cos": round(cos, 6), "rel_l2": round(rel, 5),
                                 "ref_absmax": round(float(b.abs().max()), 4)}
                rows.append(row)
                p(f"  call {ci} t={tval:.0f}: " + "  ".join(
                    f"{k} cos {row[k]['cos']:.5f} rel {row[k]['rel_l2']:.4f}"
                    for k in ("latent", "camera")))
        report["parity"] = rows
        (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))

    p("done", final_path)
    p("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        p("FAILED")
        p(traceback.format_exc()[-3000:])
        raise
