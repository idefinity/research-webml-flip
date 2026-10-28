"""K2t - export the decoder half of the pipeline: one octree level, and the
Gaussian head.

Why two graphs and not one `decode_latent` graph: the octree sampler is a loop
over eight levels whose token count changes every level - level 1 sees one
voxel, level 8 sees up to num_points - and inside the loop it samples from a
categorical distribution. Neither the changing length nor the sampling has an
ONNX shape, so the loop itself belongs in JS. What does have a shape is one
level: given the parent coordinates at this level, run the transformer and hand
back 8 logits per parent. Export that once with a dynamic token axis and the JS
driver can call it eight times.

The Gaussian head is the opposite: 2048 tokens in, 2048 x 480 out, no loop, no
sampling. One graph, static.

Both go through MatMulNBits(bits=4, block_size=32, symmetric), the same
quantiser the DiT used, so the browser loads one kind of weight.

Also written out: the constants `_build_gaussians` needs (layout, rep_config,
the Hammersley perturbation table and the offset scale). Without them the last
step cannot be reproduced in JS, and they are buffers, not weights - ONNX will
not carry them anywhere the page can see.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn

W = Path("/kaggle/working")
REPO = W / "TripoSplat"
OUT = W / "k2t"
OUT.mkdir(parents=True, exist_ok=True)
HF_TOKEN = "__TOKEN__"
TARGET = "idefinity/flip-web"
SHA = "d8db9e0"

log = []
R = {"notes": [], "graphs": {}}


def p(*a):
    s = " ".join(str(x) for x in a)
    log.append(s)
    R["notes"].append(s)
    print(s, flush=True)
    try:
        (OUT / "report.json").write_text(json.dumps(R, indent=2, default=float))
        (OUT / "log.txt").write_text("\n".join(log))
    except Exception:  # noqa: BLE001
        pass


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    p("$", " ".join(cmd)[:140], "-> rc", r.returncode)
    if r.returncode != 0:
        p("   ", (r.stderr or r.stdout or "")[-500:].replace("\n", " "))
    return r


def ops_hist(path):
    import onnx
    m = onnx.load(path)
    h = {}
    for n in m.graph.node:
        k = (n.domain + "::" if n.domain else "") + n.op_type
        h[k] = h.get(k, 0) + 1
    return h


try:
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
         "onnx", "onnxscript", "onnxruntime", "huggingface_hub",
         "safetensors", "torchvision", "tqdm"], timeout=1200)

    if not (REPO / ".git").exists():
        run(["git", "clone", "-q", "https://github.com/VAST-AI-Research/TripoSplat.git",
             str(REPO)], timeout=600)
    run(["git", "-C", str(REPO), "checkout", "-q", SHA], timeout=300)

    from huggingface_hub import HfApi, snapshot_download

    ckpt = REPO / "ckpts"
    if not (ckpt / ".complete").exists():
        snapshot_download(repo_id="VAST-AI/TripoSplat", local_dir=str(ckpt))
        (ckpt / ".complete").touch()

    sys.path.insert(0, str(REPO))
    import triposplat as T

    files = sorted(ckpt.rglob("*.safetensors"))
    dec_path = None
    for f in files:
        if "decoder" in f.name.lower() or "octree" in f.name.lower():
            dec_path = f
            break
    p("decoder ckpt", dec_path.name if dec_path else "MISSING")

    # fp32 on purpose: fp16 graphs export but ORT-web refuses to load them
    dec = T.load_decoder(str(dec_path), device="cpu", dtype=torch.float32)
    dec.eval()
    p("decoder params", round(sum(q.numel() for q in dec.parameters()) / 1e6, 1), "M")
    p("  octree", round(sum(q.numel() for q in dec.octree.parameters()) / 1e6, 1), "M")
    p("  gs    ", round(sum(q.numel() for q in dec.gs.parameters()) / 1e6, 1), "M")

    # ---- the constants the JS side needs to finish the job
    gs = dec.gs
    meta = {
        "gaussians_per_point": gs.rep_config["num_gaussians"],
        "rep_config": gs.rep_config,
        "layout": gs.layout,
        "out_channels": gs.out_channels,
        "points_offset_perturbation": gs.points_offset_perturbation.tolist(),
        "base_offset_scale": float(gs.base_offset_scale),
        "num_blocks_gs": gs.num_blocks,
        "octree_max_level": T.OctreeGaussianDecoder._MAX_VOXEL_LEVEL,
        "cond_tokens": 8192,
        "cond_channels": 16,
    }
    (OUT / "decoder_meta.json").write_text(json.dumps(meta, indent=2))
    p("layout", json.dumps({k: v["range"] for k, v in gs.layout.items()}))

    # ---------------- octree: one level, dynamic token count ----------------
    class OctreeLevel(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, l, cond, l2):
            return self.m(x, l, cond, l2)["logits"]

    # Static at the longest token count the loop can ever ask for, and padded:
    # the octree transformer is cross-only (ModulatedTransformerCrossOnlyBlock
    # has no self-attention - every token sees the conditioning and nothing
    # else), so a token's output does not depend on its neighbours and padding
    # the parent list with junk cannot change the rows we do use. That buys a
    # graph with no dynamic axis, which the exporter insists on:
    # x.reshape(-1,3).reshape(B,L,-1) turns B and L into Gather nodes under
    # dynamic_axes and the export dies with "ListConstruct is not constant".
    # A dynamic export was tried first and failed exactly there.
    NTOK_OCT = 2048
    oct = OctreeLevel(dec.octree).eval()
    x = torch.randn(1, NTOK_OCT, 3)
    l = torch.tensor([4], dtype=torch.long)        # res = 2**level, level 2 here
    cond = torch.randn(1, 8192, 16)
    l2 = torch.tensor([2048], dtype=torch.long)    # num_points
    with torch.no_grad():
        ref = oct(x, l, cond, l2)
    p("octree ref logits", tuple(ref.shape))

    # the padding claim above is a claim until it is measured: the first 8 rows
    # of a 2048-row run must equal a run of those 8 rows alone
    with torch.no_grad():
        few = oct(x[:, :8].clone(), l, cond, l2)
    pad_err = float((few - ref[:, :8]).abs().max())
    p("padding check: 8 rows alone vs inside 2048, max|diff| =", pad_err)
    R["padding_max_abs"] = pad_err

    fp_oct = OUT / "octree_level.fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(oct, (x, l, cond, l2), str(fp_oct),
                          input_names=["x", "l", "cond", "l2"],
                          output_names=["logits"],
                          opset_version=17, dynamo=False)
    p("octree fp32", round(fp_oct.stat().st_size / 1e6, 1), "MB", ops_hist(str(fp_oct)))
    import onnx as _onnx

    _m = _onnx.load(str(fp_oct))
    R["graphs"]["octree_level"] = {"inputs": [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in _m.graph.input],
                                   "outputs": [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in _m.graph.output]}
    p("octree io", json.dumps(R["graphs"]["octree_level"]))

    # ---------------- gaussian head: 2048 tokens, static --------------------
    class GsHead(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, cond):
            # x is a dict upstream: FixedlenDecoder reads x["points"]. A tensor
            # fails with "too many indices for tensor of dimension 3", which is
            # what the first run did.
            return self.m(x={"points": x}, cond=cond)["features"]

    head = GsHead(gs).eval()
    NTOK = 2048
    xp = torch.randn(1, NTOK, 3)
    condp = torch.randn(1, 8192, 16)
    t0 = time.time()
    with torch.no_grad():
        ref_gs = head(xp, condp)
    p("gs ref features", tuple(ref_gs.shape), "in", round(time.time() - t0, 1), "s")

    fp_gs = OUT / "gs.fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(head, (xp, condp), str(fp_gs),
                          input_names=["x", "cond"], output_names=["features"],
                          opset_version=17, dynamo=False)
    p("gs fp32", round(fp_gs.stat().st_size / 1e6, 1), "MB", ops_hist(str(fp_gs)))

    # ---------------- int4, and parity on CPU ------------------------------
    import numpy as np
    import onnx
    import onnxruntime as ort

    def quant(src, dst):
        """Same call the DiT graph used (k2p), quirks included: the quantiser
        takes a path and its `.model` is one of three things depending on the
        onnxruntime version."""
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        q = MatMulNBitsQuantizer(str(src), block_size=32, is_symmetric=True, bits=4)
        q.process()
        mdl = q.model
        if hasattr(mdl, "save"):
            mdl.save(str(dst))
        elif hasattr(mdl, "model"):
            onnx.save(mdl.model, str(dst))
        else:
            onnx.save(mdl, str(dst))
        return ops_hist(str(dst))

    def parity(fp, q4, feeds, out_name):
        o1 = ort.InferenceSession(str(fp), providers=["CPUExecutionProvider"])
        o2 = ort.InferenceSession(str(q4), providers=["CPUExecutionProvider"])
        # an input the graph never uses is dropped by the exporter, so feed only
        # what the graph actually declares (l2 goes away when the level embedder
        # has no second branch)
        want = {i.name for i in o1.get_inputs()}
        feeds = {k: v for k, v in feeds.items() if k in want}
        t0 = time.time()
        a = o1.run(None, feeds)[0]
        t1 = time.time()
        b = o2.run(None, feeds)[0]
        t2 = time.time()
        x_ = a.reshape(-1).astype(np.float64)
        y_ = b.reshape(-1).astype(np.float64)
        return {
            "cos": float(x_ @ y_ / (np.linalg.norm(x_) * np.linalg.norm(y_) + 1e-12)),
            "rel_l2": float(np.linalg.norm(x_ - y_) / (np.linalg.norm(y_) + 1e-12)),
            "fp32_ms": round((t1 - t0) * 1000, 1),
            "int4_ms": round((t2 - t1) * 1000, 1),
        }

    for name, fp in (("octree_level", fp_oct), ("gs", fp_gs)):
        q4 = OUT / f"{name}.int4.onnx"
        h = quant(fp, q4)
        p(name, "int4", round(q4.stat().st_size / 1e6, 1), "MB", h)
        R["graphs"][name] = {"int4_mb": round(q4.stat().st_size / 1e6, 2),
                             "fp32_mb": round(fp.stat().st_size / 1e6, 2), "ops": h}

    R["graphs"]["octree_level"]["parity"] = parity(
        fp_oct, OUT / "octree_level.int4.onnx",
        {"x": x.numpy(), "l": l.numpy(), "cond": cond.numpy(), "l2": l2.numpy()}, "logits")
    p("octree parity", json.dumps(R["graphs"]["octree_level"]["parity"]))

    # the head at 2048 tokens costs minutes on CPU, so parity is measured on the
    # same graph fed a shorter crop: it is the same weights and the same ops,
    # and a shorter token count only shrinks the attention matrix
    short = OUT / "gs_short.fp32.onnx"
    xs, cs = torch.randn(1, 64, 3), torch.randn(1, 8192, 16)
    with torch.no_grad():
        torch.onnx.export(head, (xs, cs), str(short),
                          input_names=["x", "cond"], output_names=["features"],
                          opset_version=17, dynamo=False)
    quant(short, OUT / "gs_short.int4.onnx")
    R["graphs"]["gs"]["parity_64tok"] = parity(
        str(short), OUT / "gs_short.int4.onnx",
        {"x": xs.numpy(), "cond": cs.numpy()}, "features")
    p("gs parity (64 tok)", json.dumps(R["graphs"]["gs"]["parity_64tok"]))

    # ---------------- publish ---------------------------------------------
    api = HfApi(token=HF_TOKEN)
    api.create_repo(TARGET, private=False, exist_ok=True, repo_type="model")
    for f in sorted(OUT.glob("*")):
        if f.suffix in (".onnx", ".json", ".txt") and "short" not in f.name and "fp32" not in f.name:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"decoder/{f.name}",
                            repo_id=TARGET, repo_type="model")
            p("uploaded", f.name, round(f.stat().st_size / 1e6, 2), "MB")
    p("done -> https://huggingface.co/" + TARGET + "/tree/main/decoder")
except Exception:  # noqa: BLE001
    p("FAILED")
    p(traceback.format_exc()[:3000])

(OUT / "log.txt").write_text("\n".join(log))
print("end", flush=True)
