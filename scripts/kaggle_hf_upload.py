"""Publish the 2-step student to Hugging Face, from inside a Kaggle kernel.

Why this runs on Kaggle and not on my own machine: huggingface.co is
unreachable from the machine that made the weights (the proxy answers 502 and
the direct route does not connect either), while the weights themselves are
already sitting in a Kaggle session output. Uploading from there costs one
fetch instead of a 209 MB round trip through a network that cannot make it.

What it does, in order:

  1. fetch int4.bin (209 MB, 24 parallel range requests) and manifest.json
  2. check the sha256 against the value recorded in assets/k2n/int4.sha256 -
     anything wrong here makes every later number meaningless
  3. fetch to_safetensors.py and the three small repo files from GitHub at a
     pinned commit
  4. repack into safetensors and re-check every tensor against the manifest
  5. upload

The token is substituted in at submit time. PRIVATE stays True until the
upstream licence of TripoSplat is settled.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import traceback
import urllib.request

TOKEN = "__TOKEN__"
BIN_URL = "__BIN_URL__"
MAN_URL = "__MAN_URL__"
REPO = "flip-student-2step"
PRIVATE = True
SHA = "17aae6bf84f4e2dfcd3104cb048b19bc76553fa8"
SRC = "idefinity/research-webml-flip"
WANT_SHA256 = "3d5b672b01c3d1696fd05f538ef2268c096e1b31e47d42b35bfb917af8c5312b"
SIZE = 209276416

W = "/kaggle/working"
log = []


def p(*a):
    s = " ".join(str(x) for x in a)
    log.append(s)
    print(s, flush=True)


def raw(path):
    for base in ("https://raw.githubusercontent.com", "https://cdn.jsdelivr.net/gh"):
        url = f"{base}/{SRC}/{SHA}/{path}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flip-upload"})
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            continue
    raise RuntimeError("cannot fetch " + path)


def fetch_ranged(url, out, size, threads=24):
    """The signed URL answers 206 to concurrent range requests, so the
    half-hour single stream becomes minutes."""
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
                        url, headers={"Range": f"bytes={a}-{b - 1}", "User-Agent": "flip-upload"})
                    data = urllib.request.urlopen(req, timeout=120).read()
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
                        import time
                        time.sleep(2 * (attempt + 1))

    ts = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if err:
        raise RuntimeError("failed spans: " + "; ".join(err[:3]))


try:
    p("token", TOKEN[:4] + "..." + TOKEN[-4:], f"({len(TOKEN)} chars)")

    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
                    "huggingface_hub"], capture_output=True, text=True, timeout=600)
    from huggingface_hub import HfApi

    # ---- 1. weights
    p("fetching int4.bin ...")
    fetch_ranged(BIN_URL, f"{W}/int4.bin", SIZE)
    got = os.path.getsize(f"{W}/int4.bin")
    p("int4.bin", got, "bytes | expected", SIZE, "| match", got == SIZE)

    h = hashlib.sha256()
    with open(f"{W}/int4.bin", "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    digest = h.hexdigest()
    p("sha256", digest)
    if digest != WANT_SHA256:
        raise RuntimeError("sha256 mismatch - refusing to publish a file I cannot verify")
    p("sha256 matches the one recorded next to the local copy")

    open(f"{W}/manifest.json", "wb").write(urllib.request.urlopen(
        urllib.request.Request(MAN_URL, headers={"User-Agent": "flip-upload"}),
        timeout=120).read())

    # ---- 2. scripts and the small repo files
    open(f"{W}/to_safetensors.py", "wb").write(raw("scripts/to_safetensors.py"))
    for name in ("README.md", "config.json", "quantize.json"):
        open(f"{W}/{name}", "wb").write(raw(f"dist/flip-student-2step/{name}"))
    p("fetched repo files from", SRC, "@", SHA[:8])

    # ---- 3. repack
    sys.path.insert(0, W)
    import importlib.util
    spec = importlib.util.spec_from_file_location("ts", f"{W}/to_safetensors.py")
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    ts.main(f"{W}/manifest.json", f"{W}/model.safetensors")
    p("model.safetensors", round(os.path.getsize(f"{W}/model.safetensors") / 1e6, 1), "MB")

    # ---- 4. the container must describe exactly what the manifest says
    import struct
    f = open(f"{W}/model.safetensors", "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    base = 8 + n
    man = json.load(open(f"{W}/manifest.json"))
    blob = open(f"{W}/int4.bin", "rb").read()
    bad = checked = 0
    for t in man["tensors"]:
        pairs = ([(t["name"] + ".weight", t["offset_q"], t["bytes_q"]),
                  (t["name"] + ".weight_scale", t["offset_s"], t["bytes_s"])]
                 if t["kind"] == "int4" else [(t["name"], t["offset"], t["bytes"])])
        for name, off, ln in pairs:
            a, b = hdr[name]["data_offsets"]
            f.seek(base + a)
            checked += 1
            if f.read(b - a) != blob[off:off + ln]:
                bad += 1
    p("tensors compared", checked, "| mismatched", bad)
    if bad:
        raise RuntimeError("safetensors does not match the source bytes")

    # ---- 5. publish
    api = HfApi(token=TOKEN)
    who = api.whoami()
    user = who.get("name")
    p("hf user", user)
    repo_id = f"{user}/{REPO}"
    api.create_repo(repo_id, private=PRIVATE, exist_ok=True,
                    repo_type="model")
    for name in ("model.safetensors", "README.md", "config.json",
                 "quantize.json", "manifest.json"):
        api.upload_file(path_or_fileobj=f"{W}/{name}", path_in_repo=name,
                        repo_id=repo_id, repo_type="model")
        p("uploaded", name, round(os.path.getsize(f"{W}/{name}") / 1e6, 2), "MB")
    p("repo https://huggingface.co/" + repo_id)
    p("private =", PRIVATE)
except Exception:
    p("FAILED", traceback.format_exc()[:1500])

with open(f"{W}/log.txt", "w") as f:
    f.write("\n".join(log))
print("end", flush=True)
