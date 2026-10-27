"""Export the 2-step student DiT to ONNX, quantise its matmuls to 4 bits, and
publish the graph next to the safetensors.

Inputs are the raw encoder outputs: latent [B,8192,16], camera [B,1,5],
feature1 [B,4101,1280] (DINOv3), feature2 [B,4096,128] (Flux2 VAE), t [B].
The 5 zero tokens the encoder pads in front of the VAE features are inside the
graph, so a caller hands over what the VAE actually produced.

The safetensors repo is a weight file: to run it you need the architecture, the
sampler constants and our packing convention. An ONNX graph is the other half -
it carries the structure, and with MatMulNBits it carries the 4-bit weights too,
so ORT-web can load it directly.

Two things decide what the graph is worth:

  - Whether torch's exporter keeps the attention as 2 MatMul + Softmax. It does,
    at opset 17, and we measured that shape at 406 GFLOP/s against our fused
    kernel's 1789. So this graph is NOT the fast path for the DiT; it is the
    interoperable one. The browser path keeps our WGSL attention and uses
    gpu-buffer to hand tensors to ORT for everything else.
  - Whether the quantised graph still computes the same thing. That is checked
    here, on CPU, at small token counts, against the fp32 graph.

Env:
  FLIP_STUDENT_URL   signed URL for k2e/student.pt (1.48 GB fp32)
  HF_TOKEN           write token, scoped to the idefinity org
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

W = "/kaggle/working"
UPSTREAM = "https://github.com/VAST-AI-Research/TripoSplat.git"
UPSTREAM_SHA = "d8db9e0"
REPO = "idefinity/flip-student-2step"
TOKEN = os.environ.get("HF_TOKEN", "")
STUDENT_URL = os.environ.get("FLIP_STUDENT_URL", "")

log = []


def p(*a):
    s = " ".join(str(x) for x in a)
    log.append(s)
    print(s, flush=True)


def total_size(url):
    """A torch checkpoint is a zip: its central directory sits at the end, so a
    file padded or truncated to a guessed length loads as 'corrupted'. Ask for
    the length instead of assuming 1,482,000,000."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "flip-export"})
    with urllib.request.urlopen(req, timeout=120) as r:
        cr = r.headers.get("Content-Range") or ""
        if "/" in cr:
            return int(cr.split("/")[-1])
        return int(r.headers.get("Content-Length", 0)) or None


def fetch_stream(url, out):
    """Fallback for when the length cannot be asked for: append in order."""
    off = os.path.getsize(out) if os.path.exists(out) else 0
    for attempt in range(6):
        try:
            h = {"User-Agent": "flip-export"}
            if off:
                h["Range"] = f"bytes={off}-"
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600) as r, \
                    open(out, "ab" if off else "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
                    off += len(b)
            return off
        except Exception:
            off = os.path.getsize(out) if os.path.exists(out) else 0
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("streaming fetch failed")


def fetch_ranged(url, out, size, threads=24):
    chunk = max(size // (threads * 4), 1 << 20)
    spans = [(i, min(i + chunk, size)) for i in range(0, size, chunk)]
    with open(out, "wb") as f:
        f.truncate(size)
    err = []

    def work(t):
        for a, b in spans[t::threads]:
            for attempt in range(4):
                try:
                    req = urllib.request.Request(
                        url, headers={"Range": f"bytes={a}-{b - 1}", "User-Agent": "flip-export"})
                    data = urllib.request.urlopen(req, timeout=180).read()
                    if len(data) != b - a:
                        raise IOError(f"short {len(data)} of {b - a}")
                    with open(out, "r+b") as f:
                        f.seek(a)
                        f.write(data)
                    break
                except Exception as e:
                    if attempt == 3:
                        err.append(f"{a}-{b}: {e}")
                    else:
                        time.sleep(2 * (attempt + 1))

    ts = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if err:
        raise RuntimeError("failed spans: " + "; ".join(err[:3]))


def ops_hist(path):
    import onnx
    m = onnx.load(path)
    h = {}
    for n in m.graph.node:
        k = (n.domain + "::" if n.domain else "") + n.op_type
        h[k] = h.get(k, 0) + 1
    return h


try:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
                    "onnxruntime", "onnx_ir", "huggingface_hub"],
                   capture_output=True, text=True, timeout=900)

    # ---- upstream code: we need the architecture and its constants, not its weights
    subprocess.run(["git", "clone", "-q", UPSTREAM, f"{W}/TripoSplat"],
                   capture_output=True, text=True, timeout=600)
    subprocess.run(["git", "-C", f"{W}/TripoSplat", "checkout", "-q", UPSTREAM_SHA],
                   capture_output=True, text=True, timeout=300)
    # The repo ships four files and no requirements.txt; the imports are
    # numpy, torch, safetensors, PIL, torchvision, tqdm. Install the ones that
    # might be missing and continue either way - the import below is the real
    # check, and a failed pip should not be the thing that ends the run.
    reqs = ["safetensors", "torchvision", "tqdm"]
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q"] + reqs,
                       capture_output=True, text=True, timeout=1200)
    p("pip reqs rc", r.returncode, (r.stderr or "")[-300:].replace("\n", " "))

    import torch
    import torch.nn as nn
    sys.path.insert(0, f"{W}/TripoSplat")
    import triposplat as T
    import model as M

    p("torch", torch.__version__)
    args = dict(T.FLOW_MODEL_ARGS)
    p("flow args", json.dumps(args))

    # ---- student weights
    p("fetching student.pt ...")
    size = None
    try:
        size = total_size(STUDENT_URL)
    except Exception as e:
        p("  length probe failed:", str(e)[:120])
    if size:
        p("  size from server:", size)
        fetch_ranged(STUDENT_URL, f"{W}/student.pt", size)
    else:
        fetch_stream(STUDENT_URL, f"{W}/student.pt")
    size = os.path.getsize(f"{W}/student.pt")
    p("student.pt", size, "bytes")
    h = hashlib.sha256()
    with open(f"{W}/student.pt", "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    p("student.pt sha256", h.hexdigest())

    flow = M.LatentSeqMMFlowModel(**args)
    sd = torch.load(f"{W}/student.pt", map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = flow.load_state_dict(sd, strict=False)
    p("loaded:", len(sd), "tensors | missing", len(missing), "| unexpected", len(unexpected))
    if missing:
        p("  missing sample:", missing[:3])
    flow.eval()

    class Wrap(nn.Module):
        """Dicts in, tuple out: ONNX has no dict, and tracing through one
        produces a graph nobody else can feed."""

        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, latent, camera, feature1, feature2, t):
            # The encoder pads 5 zero tokens in front of the VAE features so
            # they line up with the DINO tokens (cls + 4 registers + patches).
            # Upstream does this outside the model, which means the graph's
            # caller would have to know to do it. Bake it in: feature2 is then
            # the raw VAE output, which is what anyone actually has.
            b = feature2.shape[0]
            pad = torch.zeros(b, 5, feature2.shape[2],
                              dtype=feature2.dtype, device=feature2.device)
            out = self.f({"latent": latent, "camera": camera}, t,
                         {"feature1": feature1, "feature2": torch.cat([pad, feature2], dim=1)})
            return out["latent"], out["camera"]

    wrap = Wrap(flow)
    B, LQ, LC1, LC2 = 1, args["q_token_length"], 4101, 4096
    dummy = (torch.randn(B, LQ, args["in_channels"]),
             torch.randn(B, 1, args["cam_channels"]),
             torch.randn(B, LC1, args["cond_channels"]),
             torch.randn(B, LC2, args["cond2_channels"]),
             torch.tensor([500.0]))
    names = ["latent", "camera", "feature1", "feature2", "t"]
    dyn = {n: {0: "B"} for n in names}
    dyn["latent"][1] = "LQ"
    dyn["feature1"][1] = "LC1"
    dyn["feature2"][1] = "LC2"

    fp32 = f"{W}/dit.fp32.onnx"
    p("wrapper forward once, so a tracing failure is not mistaken for a model bug")
    t0 = time.time()
    with torch.no_grad():
        outs = wrap(*dummy)
    p("  ok:", [tuple(o.shape) for o in outs], "in", round(time.time() - t0, 1), "s")
    p("  cond token count: feature1", LC1, "+ pad 5 + feature2", LC2, "=", LC2 + 5)

    p("exporting (one traced forward at LQ=%d, this takes a minute) ..." % LQ)
    t0 = time.time()
    try:
        with torch.no_grad():
            torch.onnx.export(wrap, dummy, fp32, input_names=names,
                              output_names=["out_latent", "out_camera"],
                              opset_version=17, dynamo=False, dynamic_axes=dyn)
    except Exception as e:
        p("tracing export failed:", traceback.format_exc()[-2500:])
        p("retrying with the dynamo exporter, opset 18")
        torch.onnx.export(wrap, dummy, fp32, input_names=names,
                          output_names=["out_latent", "out_camera"],
                          opset_version=18, dynamo=True, dynamic_axes=dyn)
    p("fp32 graph", round(os.path.getsize(fp32) / 1e6, 1), "MB in",
      round(time.time() - t0, 1), "s")
    h = ops_hist(fp32)
    p("fp32 nodes", sum(h.values()))
    p("  ", json.dumps(h))

    # ---- int4
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    int4 = f"{W}/dit.int4.onnx"
    qz = MatMulNBitsQuantizer(fp32, block_size=32, is_symmetric=True, bits=4)
    qz.process()
    mdl = qz.model
    if hasattr(mdl, "save"):
        mdl.save(int4)
    elif hasattr(mdl, "model"):
        onnx.save(mdl.model, int4)
    else:
        onnx.save(mdl, int4)
    p("int4 graph", round(os.path.getsize(int4) / 1e6, 1), "MB")
    h4 = ops_hist(int4)
    p("int4 nodes", sum(h4.values()))
    p("  ", json.dumps(h4))

    # ---- the quantised graph has to compute the same thing
    import numpy as np
    import onnxruntime as ort
    small = (np.random.rand(1, 256, args["in_channels"]).astype("float32"),
             np.random.rand(1, 1, args["cam_channels"]).astype("float32"),
             np.random.rand(1, 64, args["cond_channels"]).astype("float32"),
             np.random.rand(1, 64, args["cond2_channels"]).astype("float32"),
             np.array([500.0], dtype="float32"))
    feeds = dict(zip(names, small))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    a = ort.InferenceSession(fp32, so, providers=["CPUExecutionProvider"]).run(None, feeds)
    b = ort.InferenceSession(int4, so, providers=["CPUExecutionProvider"]).run(None, feeds)
    for i, nm in enumerate(["latent", "camera"]):
        x, y = a[i].astype("float64"), b[i].astype("float64")
        cos = float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y)))
        rel = float(np.linalg.norm(x - y) / np.linalg.norm(x))
        p(f"  {nm}: cosine {cos:.5f} | rel_l2 {rel:.4f}")

    # ---- publish
    if TOKEN:
        from huggingface_hub import HfApi
        api = HfApi(token=TOKEN)
        for path, name in [(int4, "dit.int4.onnx")]:
            sha = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    blk = f.read(1 << 22)
                    if not blk:
                        break
                    sha.update(blk)
            p("uploading", name, round(os.path.getsize(path) / 1e6, 1), "MB")
            api.upload_file(path_or_fileobj=path, path_in_repo=name,
                            repo_id=REPO, repo_type="model")
            p("  sha256", sha.hexdigest())
        p("repo https://huggingface.co/" + REPO)
    else:
        p("no HF_TOKEN, skipping upload")
except Exception:
    p("FAILED", traceback.format_exc()[:1200] + "\n...\n" + traceback.format_exc()[-2500:])

with open(f"{W}/log.txt", "w") as f:
    f.write("\n".join(log))
print("end", flush=True)
