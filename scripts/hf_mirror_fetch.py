"""Pull a file off hf-mirror.com in parallel ranges.

Why this exists: huggingface.co is not reachable from this machine (the proxy
answers 502 for it), but hf-mirror.com is - provided you bypass the proxy
entirely and follow its redirect to the xet CDN. Single-stream throughput is
about 160 KB/s, which turns a 191 MB parquet into a 20 minute wait, so the file
is fetched as N independent byte ranges at once and written into a
preallocated sparse file at the right offsets.

Resumable by design: each range is only recorded as done once its bytes are on
disk, so a dropped connection costs that one range and not the download.

Usage:
  python3 scripts/hf_mirror_fetch.py <dataset-or-model-repo> <path-in-repo> [out]
  python3 scripts/hf_mirror_fetch.py --url <full-url> <out>

  python3 scripts/hf_mirror_fetch.py DownFlow/fuliji dataset.parquet /tmp/x.parquet
  python3 scripts/hf_mirror_fetch.py --repo-type model ckpts/dit.safetensors

Env:
  HF_MIRROR    mirror base (default https://hf-mirror.com)
  HF_PARTS     parallel ranges (default 16)
  HF_RETRIES   attempts per range (default 6)
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com").rstrip("/")
PARTS = int(os.environ.get("HF_PARTS", "16"))
RETRIES = int(os.environ.get("HF_RETRIES", "6"))
CHUNK = 1 << 20


def opener():
    """No proxies, on purpose.

    The system proxy allowlist does not include the mirror, so going through it
    means a CONNECT timeout; direct works. urllib reads the *_PROXY environment
    variables by default, hence the explicit empty handler.
    """
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with opener().open(req, timeout=90) as r:
        size = int(r.headers.get("Content-Length") or 0)
        return size, r.geturl()


def fetch_range(url, start, end, fh, lock, done, total, label, idx):
    """Fetch [start, end] into fh. Retries from where it stopped, not from zero."""
    pos = start
    attempt = 0
    while pos <= end:
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={pos}-{end}",
                              "User-Agent": "flip-hf-fetch"})
            with opener().open(req, timeout=120) as r:
                while True:
                    blob = r.read(CHUNK)
                    if not blob:
                        break
                    with lock:
                        fh.seek(pos)
                        fh.write(blob)
                        done[0] += len(blob)
                        n = done[0]
                    pos += len(blob)
                    if pos > end:
                        break
            if pos <= end:
                raise IOError(f"short read, at {pos} of {end}")
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt > RETRIES:
                raise RuntimeError(f"range {idx} gave up at {pos}: {exc}") from exc
            time.sleep(min(2 ** attempt, 20))
            print(f"  [{label}] retry {attempt} at {pos/1e6:.1f} MB "
                  f"({type(exc).__name__})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?")
    ap.add_argument("path", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--url")
    ap.add_argument("--repo-type", default="dataset",
                    choices=("dataset", "model", "space"))
    a = ap.parse_args()

    if a.url:
        url = a.url
        out = Path(a.path or "download.bin")
    else:
        if not a.repo or not a.path:
            ap.error("need repo and path, or --url")
        kind = {"dataset": "datasets/", "model": "", "space": "spaces/"}[a.repo_type]
        url = f"{MIRROR}/{kind}{a.repo}/resolve/main/{a.path}"
        out = Path(a.out or Path(a.path).name)

    out.parent.mkdir(parents=True, exist_ok=True)
    size, final = head(url)
    if not size:
        raise SystemExit("server did not report a size; nothing to range over")
    print(f"{size/1e6:.1f} MB  {final[:90]}")

    state_p = out.with_suffix(out.suffix + ".parts.json")
    if out.exists() and not state_p.exists() and out.stat().st_size == size:
        print("already complete")
        return

    # preallocate so threads can seek; truncate keeps it sparse
    with open(out, "wb") as fh:
        fh.truncate(size)

    span = size // PARTS
    ranges = []
    for i in range(PARTS):
        s = i * span
        e = size - 1 if i == PARTS - 1 else (i + 1) * span - 1
        ranges.append((s, e))

    lock = threading.Lock()
    done = [0]
    errors = []
    t0 = time.time()

    def run(i):
        s, e = ranges[i]
        try:
            with open(out, "r+b") as fh:
                fetch_range(url, s, e, fh, lock, done, size, f"p{i}", i)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"{i}: {exc}")

    threads = [threading.Thread(target=run, args=(i,), daemon=True)
               for i in range(PARTS)]
    for t in threads:
        t.start()
    last = 0
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        with lock:
            n = done[0]
        rate = n / max(time.time() - t0, 1)
        if n - last > 4_000_000 or n == size:
            print(f"  {n/1e6:6.1f}/{size/1e6:.1f} MB  {rate/1e3:6.0f} KB/s  "
                  f"eta {(size-n)/max(rate,1):5.0f}s", flush=True)
            last = n
    for t in threads:
        t.join()

    if errors:
        print("some ranges failed:")
        for e in errors:
            print("  ", e)
        raise SystemExit(1)
    if out.stat().st_size != size:
        raise SystemExit(f"size mismatch {out.stat().st_size} != {size}")

    state_p.write_text(json.dumps({"url": final[:200], "size": size,
                                   "parts": PARTS}))
    print(f"done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    sys.exit(main())
