"""Kaggle kernel body for the flip probes.

This file is *copied verbatim* into the kernel by the local tooling, prefixed
with two constants (FLIP_SHA, FLIP_SCRIPT). Everything that carries actual
logic lives in the GitHub repo and is fetched at that pinned commit, so a
kernel run is identified by one SHA instead of "whatever was pasted last time".

Why not fetch the bootstrap itself from GitHub? Because it installs the
packages and clones the upstream repo, and its own integrity is what the rest
depends on. One manual copy is cheaper than a bootstrap chain.

Env it produces:
  /kaggle/working/TripoSplat/ckpts   weights
  /kaggle/working/k0/report.json     probe output (when FLIP_SCRIPT is k0)
"""

import hashlib
import os
import subprocess
import sys
import traceback
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

UPSTREAM_REPO = "https://github.com/VAST-AI-Research/TripoSplat.git"
UPSTREAM_SHA = "d8db9e0"
HF_REPO = "VAST-AI/TripoSplat"

FLIP_REPO = "idefinity/research-webml-flip"


def log(msg=""):
    print(msg, flush=True)


def run(cmd, **kw):
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        log(f"  rc={r.returncode}")
        log("  stdout: " + r.stdout.strip()[-1500:])
        log("  stderr: " + r.stderr.strip()[-1500:])
    return r


# ---------------------------------------------------------------------------
# 0. identify this run
# ---------------------------------------------------------------------------
log("=" * 72)
log(f"FLIP_SHA    {FLIP_SHA}")
log(f"FLIP_SCRIPT {FLIP_SCRIPT}")
log(f"upstream    {UPSTREAM_REPO} @ {UPSTREAM_SHA}")
log("=" * 72)
log(run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"]).stdout.strip())

# ---------------------------------------------------------------------------
# 1. deps
# ---------------------------------------------------------------------------
# onnxruntime-gpu on PyPI is built against one CUDA major at a time, and the
# current wheel wants CUDA 13 while this image ships CUDA 12.8. Left unpinned
# it installs cleanly, then fails at session creation with
#   libcublasLt.so.13: cannot open shared object file
# and silently falls back to CPU - which is how a 370 M DiT got OOM-killed.
run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"])
r = run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu==1.22.0"])
if r.returncode != 0:
    log("pinned onnxruntime-gpu==1.22.0 failed, falling back to unpinned")
    run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu"])
run([sys.executable, "-m", "pip", "install", "-q",
     "safetensors", "pillow", "tqdm", "huggingface_hub"])
try:
    import onnxruntime as _ort
    log(f"onnxruntime {_ort.__version__} providers={_ort.get_available_providers()}")
except Exception:  # noqa: BLE001
    log("onnxruntime import failed")
log("pip done")

# ---------------------------------------------------------------------------
# 2. upstream repo at a pinned commit
# ---------------------------------------------------------------------------
REPO = WORK / "TripoSplat"
if not (REPO / ".git").exists():
    run(["git", "clone", UPSTREAM_REPO, str(REPO)])
r = run(["git", "-C", str(REPO), "checkout", UPSTREAM_SHA])
log("HEAD " + run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip())
log("files: " + str(sorted(os.listdir(REPO))))

# ---------------------------------------------------------------------------
# 3. weights
# ---------------------------------------------------------------------------
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
    log("hf token: found")
except Exception as exc:  # noqa: BLE001
    log(f"hf token: unavailable ({type(exc).__name__}) - trying anonymous")

ckpt = REPO / "ckpts"
if not (ckpt / ".complete").exists():
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_REPO, local_dir=str(ckpt), token=token)
        (ckpt / ".complete").touch()
        log("weights ok")
    except Exception:  # noqa: BLE001
        log("WEIGHTS FAILED")
        log(traceback.format_exc()[-1500:])
log(run(["du", "-sh", str(ckpt)]).stdout.strip())

# ---------------------------------------------------------------------------
# 4. fetch the real script at the pinned SHA and run it
# ---------------------------------------------------------------------------
def fetch(url):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github.raw",
        "User-Agent": "flip-kernel",
    })
    with urllib.request.urlopen(req, timeout=60) as fh:
        return fh.read().decode("utf-8")


# raw.githubusercontent.com is blocked on some networks; api.github.com serves
# the same bytes from a different host, so try both before giving up.
candidates = [
    f"https://raw.githubusercontent.com/{FLIP_REPO}/{FLIP_SHA}/{FLIP_SCRIPT}",
    f"https://api.github.com/repos/{FLIP_REPO}/contents/{FLIP_SCRIPT}?ref={FLIP_SHA}",
]

src = None
for url in candidates:
    log(f"fetching {url}")
    try:
        src = fetch(url)
        log(f"  ok, {len(src)} bytes")
        break
    except Exception:  # noqa: BLE001
        log(f"  FAILED ({type(sys.exc_info()[1]).__name__})")
if src is None:
    log("FETCH FAILED on every candidate - is the commit pushed, the repo public?")
    raise RuntimeError("could not fetch probe script")

digest = hashlib.sha256(src.encode()).hexdigest()[:16]
log(f"fetched {len(src)} bytes, sha256:{digest}")

out = WORK / "fetched_script.py"
out.write_text(src)

exec(compile(src, FLIP_SCRIPT, "exec"), {"__name__": "__main__"})
