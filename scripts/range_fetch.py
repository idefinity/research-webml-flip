"""Fetch a large file over range requests, many chunks at once.

Kaggle's notebook outputs come back as a one-hour signed URL on
kaggleusercontent.com. A single stream measured 117-170 KB/s, so 209 MB took
half an hour and got truncated the first time; the same URL answers 206 to
concurrent range requests, which is what this uses.

    python3 scripts/range_fetch.py --url-file /tmp/u.txt --out assets/k2n/int4.bin \
        --size 209276416 --threads 24 --chunk 4M

No checksum exists on the Kaggle side, so the post-check is structural: the
final size has to match, and the caller can compare against the manifest.
"""
import argparse
import os
import sys
import threading
import time
import urllib.request

DONE = 0
LOCK = threading.Lock()
FAILED = []


def parse_size(s: str) -> int:
    s = s.strip().upper()
    for suf, mul in (("K", 1024), ("M", 1024 ** 2), ("G", 1024 ** 3)):
        if s.endswith(suf):
            return int(float(s[:-1]) * mul)
    return int(s)


def fetch(url: str, out: str, size: int, start: int, end: int, tries: int = 4) -> None:
    global DONE
    want = end - start
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"status {r.status}")
                data = r.read()
            if len(data) != want:
                raise RuntimeError(f"got {len(data)} of {want}")
            with open(out, "r+b") as f:
                f.seek(start)
                f.write(data)
            with LOCK:
                DONE += want
            return
        except Exception as e:                        # noqa: BLE001 - retry anything
            last = e
            time.sleep(1.5 * (attempt + 1))
    with LOCK:
        FAILED.append((start, end, str(last)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--chunk", default="4M")
    args = ap.parse_args()

    url = open(args.url_file).read().strip()
    size = parse_size(args.size)
    chunk = parse_size(args.chunk)

    # preallocate: every worker writes at its own offset, so the file has to exist
    with open(args.out, "wb") as f:
        f.truncate(size)

    spans = [(s, min(s + chunk, size)) for s in range(0, size, chunk)]
    print(f"{size / 1e6:.1f} MB in {len(spans)} chunks of {chunk / 1e6:.0f} MB, "
          f"{args.threads} at a time", flush=True)

    t0 = time.time()
    queue = list(spans)
    qlock = threading.Lock()

    def worker():
        while True:
            with qlock:
                if not queue:
                    return
                s, e = queue.pop()
            fetch(url, args.out, size, s, e)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.threads)]
    for t in threads:
        t.start()

    while any(t.is_alive() for t in threads):
        time.sleep(2.0)
        el = time.time() - t0
        rate = DONE / el / 1e3 if el else 0
        print(f"\r  {DONE / 1e6:6.1f} / {size / 1e6:.1f} MB  {rate:6.0f} KB/s  "
              f"eta {(size - DONE) / max(rate * 1e3, 1) / 60:4.1f} min", end="", flush=True)
    print()

    if FAILED:
        print(f"{len(FAILED)} chunks failed:")
        for s, e, why in FAILED[:5]:
            print(f"  {s}-{e}: {why}")
        return 1

    got = os.path.getsize(args.out)
    print(f"done in {time.time() - t0:.0f} s, {got} bytes "
          f"({'ok' if got == size else 'SIZE MISMATCH'})")
    return 0 if got == size else 1


if __name__ == "__main__":
    sys.exit(main())
