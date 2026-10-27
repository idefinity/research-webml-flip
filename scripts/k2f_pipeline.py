"""K2f - trace the teacher and train the student in one Kaggle session.

Why one session and not two: a Kaggle kernel's output is not an input to the
next one. The traces k2d writes are ~39 MB per image (4101x1280 conditioning,
40 calls), so 48 images is a couple of gigabytes that would have to be
downloaded, uploaded as a dataset, and attached - and attaching a dataset
through the API is exactly what silently failed on the last run, leaving
/kaggle/input empty. Running both stages in one session has no such step.

The two stages run as subprocesses rather than being imported, because k2d
forks one child per GPU and each child re-executes its own script file. If both
stages lived in one process, every shard worker would start training.

Env (in addition to what k2d, k2e and k2g read):
  FLIP_SHA   commit to fetch the two scripts at
  FLIP_SKIP  "trace" | "train" - skip a stage, e.g. to re-train on old traces
  FLIP_PLY   1 = after training, render the PLYs in the same session

FLIP_PLY exists because the student checkpoint is 1.48 GB and a kernel's output
is not the next kernel's input: the only way back is a signed URL, which is
single-use and runs at ~170 KB/s, and a re-run after any crash pays it again.
Keeping the checkpoint in the session that made it costs nothing.
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
FLIP_REPO = os.environ.get("FLIP_REPO", "idefinity/research-webml-flip")
SHA = os.environ.get("FLIP_SHA", "HEAD")
SCRIPTS = ["scripts/k2d_teacher_trace.py", "scripts/k2e_distill.py",
           "scripts/k2g_student_ply.py"]
SKIP = os.environ.get("FLIP_SKIP", "").strip().lower()
WITH_PLY = os.environ.get("FLIP_PLY", "0").strip() == "1"


def log(msg=""):
    print(msg, flush=True)


def fetch(path):
    """raw.githubusercontent.com is blocked on some networks; api.github.com
    serves the same bytes from a different host, so try both."""
    urls = [
        f"https://raw.githubusercontent.com/{FLIP_REPO}/{SHA}/{path}",
        f"https://api.github.com/repos/{FLIP_REPO}/contents/{path}?ref={SHA}",
    ]
    err = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flip-kernel"})
            with urllib.request.urlopen(req, timeout=60) as fh:
                raw = fh.read()
            if "api.github.com" in url:
                import base64
                import json
                raw = base64.b64decode(json.loads(raw)["content"])
            return raw.decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            err = exc
    raise RuntimeError(f"could not fetch {path} at {SHA}: {err}")


def stage(path, tag):
    dst = WORK / Path(path).name
    dst.write_text(fetch(path))
    log(f"=" * 72)
    log(f"{tag}: {dst}  ({dst.stat().st_size} bytes)")
    log("=" * 72)
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(dst)], env=dict(os.environ)).returncode
    log(f"{tag} rc={rc}  {(time.time() - t0) / 60:.1f} min")
    return rc


def main():
    log("=" * 72)
    log("K2f trace + distill")
    log("=" * 72)
    log(f"sha {SHA}  skip {SKIP or '-'}")

    trace_dir = Path(os.environ.get("FLIP_TRACE_DIR", str(WORK / "k2d")))
    n_before = len(list(trace_dir.glob("trace_*.pt"))) if trace_dir.is_dir() else 0

    rc_d = 0
    if "trace" not in SKIP:
        rc_d = stage(SCRIPTS[0], "trace")

    n_after = len(list(trace_dir.glob("trace_*.pt"))) if trace_dir.is_dir() else 0
    log(f"traces: {n_before} before, {n_after} after")

    if n_after == 0:
        log("no traces - not training")
        return 1

    rc_e = 0
    if "train" not in SKIP:
        rc_e = stage(SCRIPTS[1], "distill")

    out = WORK / "k2e"
    for name in ("student.pt", "grid.json", "report.json"):
        p = out / name
        if p.exists():
            log(f"  {name}  {p.stat().st_size / 1e6:.1f} MB")
        else:
            log(f"  {name}  MISSING")
    grid = out / "grid.json"
    if grid.exists():
        log("grid " + grid.read_text().replace("\n", " "))

    if WITH_PLY:
        rc_g = stage(SCRIPTS[2], "ply")
        if rc_g != 0:
            log(f"ply stage failed rc={rc_g}")
        rc_e = rc_e or rc_g
        plies = sorted((WORK / "k2g").glob("*.ply"))
        log(f"plies: {len(plies)}")
        for p in plies:
            log(f"  {p.name}  {p.stat().st_size / 1e6:.2f} MB")
    return rc_d or rc_e


if __name__ == "__main__":
    sys.exit(main())
