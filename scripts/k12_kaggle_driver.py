import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
LOG = WORK / "k12"
LOG.mkdir(parents=True, exist_ok=True)
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
SHA = "ac72202"
GITHUB = "https://raw.githubusercontent.com/idefinity/research-webml-flip/{}/scripts/{}"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def stage(name, cmd, env=None, timeout=3600):
    log("$", " ".join(map(str, cmd)))
    t0 = time.time()
    path = LOG / f"{name}.log"
    with open(path, "w") as fh:
        r = subprocess.run([str(c) for c in cmd], env=env, stdout=fh,
                           stderr=subprocess.STDOUT, timeout=timeout)
    text = path.read_text(errors="replace")
    log(f"{name} rc={r.returncode} {time.time() - t0:.0f}s")
    print("\n".join(text.splitlines()[-30:]), flush=True)
    return r.returncode, text


log(stage("gpu", ["nvidia-smi", "-L"], timeout=120)[1].strip())

# --- deps, upstream at the pinned commit, teacher weights (public, no token)
rc, _ = stage("pip", [sys.executable, "-m", "pip", "install", "-q",
                      "--no-cache-dir", "huggingface_hub", "safetensors",
                      "torchvision", "tqdm", "Pillow"], timeout=1800)
if rc:
    raise RuntimeError("pip failed")
if not (REPO / ".git").exists():
    rc, _ = stage("clone", ["git", "clone", "-q",
                            "https://github.com/VAST-AI-Research/TripoSplat.git",
                            str(REPO)], timeout=900)
    if rc:
        raise RuntimeError("clone failed")
rc, _ = stage("checkout", ["git", "-C", str(REPO), "checkout", "-q", "d8db9e0"],
              timeout=300)
if rc:
    raise RuntimeError("checkout failed")
if not (CKPT / ".complete").exists():
    from huggingface_hub import snapshot_download
    log("downloading VAST-AI/TripoSplat")
    t0 = time.time()
    snapshot_download(repo_id="VAST-AI/TripoSplat", local_dir=str(CKPT),
                      token=False)
    (CKPT / ".complete").touch()
    log(f"weights {time.time() - t0:.0f}s")
log("ckpts GB", round(sum(f.stat().st_size for f in CKPT.rglob("*")
                         if f.is_file()) / 1e9, 2))


# --- the two scripts that produced the Colab result, at that exact commit
WANT = {"k2d_teacher_trace.py": "7f5f450ac9c6e9ed",
        "k2e_distill.py": "ffe27e5c79c6c293"}


def fetch(name):
    urls = [GITHUB.format(SHA, name),
            "https://cdn.jsdelivr.net/gh/idefinity/research-webml-flip@"
            f"{SHA}/scripts/{name}"]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flip-k12"})
            with urllib.request.urlopen(req, timeout=60) as fh:
                blob = fh.read()
            sha = hashlib.sha256(blob).hexdigest()
            if sha[:16] != WANT[name]:
                log("  wrong sha", url, sha[:16])
                continue
            (WORK / name).write_bytes(blob)
            log(name, len(blob), "sha", sha[:16], "from", url.split("/")[2])
            return
        except Exception as exc:
            log("  miss", url.split("/")[2], type(exc).__name__, str(exc)[:80])
    raise RuntimeError(f"cannot fetch {name}")


fetch("k2d_teacher_trace.py")
fetch("k2e_distill.py")

# --- trace 4 photos, one per GPU pair (k2d forks its own shards)
env = dict(os.environ, FLIP_TRACE_N="4", FLIP_TRACE_STEPS="20",
           FLIP_TRACE_GAIN="3.0", FLIP_TRACE_SHIFT="3.0",
           FLIP_TRACE_COND="teacher", FLIP_FORK="auto",
           PYTHONUNBUFFERED="1")
env.pop("HF_ENDPOINT", None)
rc, _ = stage("trace", [sys.executable, str(WORK / "k2d_teacher_trace.py")],
              env=env, timeout=2700)
if rc:
    raise RuntimeError("trace failed")
traces = sorted((WORK / "k2d").glob("trace_*.pt"))
log("traces", [t.name for t in traces])
if len(traces) < 4:
    raise RuntimeError(f"only {len(traces)} traces")

# --- train on both GPUs; k2e single-process is the fallback if DDP breaks
env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
env.pop("HF_ENDPOINT", None)
cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
       "--standalone", str(WORK / "k12_train_ddp.py")]
try:
    rc, _ = stage("train_ddp", cmd, env=env, timeout=2700)
except subprocess.TimeoutExpired:
    subprocess.run(["pkill", "-9", "-f", "k12_train_ddp.py"], check=False)
    raise
if rc:
    log("DDP failed; checking the cards before the single-GPU fallback")
    subprocess.run(["pkill", "-9", "-f", "k12_train_ddp.py"], check=False)
    time.sleep(8)
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    log("gpu used MiB", mem)
    if not mem or any(int(x) > 2000 for x in mem):
        raise RuntimeError("cards still busy after the DDP failure")
    env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
                FLIP_GRID="index", FLIP_EPOCHS="6", FLIP_HOLDOUT="2",
                FLIP_STEPS="4", FLIP_SAVE_EVERY="2",
                FLIP_TRACE_DIR=str(WORK / "k2d"))
    env1.pop("HF_ENDPOINT", None)
    rc, _ = stage("train_single", [sys.executable, str(WORK / "k2e_distill.py")],
                  env=env1, timeout=2700)
    if rc:
        raise RuntimeError("single-GPU fallback failed")
ck = sorted((WORK / "k2e").glob("student_e*.pt"))
log("checkpoints", [c.name for c in ck])
if not ck:
    raise RuntimeError("training produced no checkpoints")

# --- eval: one process per GPU, trace i goes to shard i mod 2
procs = []
for i in range(2):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), FLIP_SHARD=f"{i}/2",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
    env.pop("HF_ENDPOINT", None)
    fh = open(LOG / f"eval_shard{i}.log", "w")
    procs.append((i, subprocess.Popen([sys.executable, str(WORK / "k12_eval.py")],
                 env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
for i, pr, fh in procs:
    try:
        rc = pr.wait(timeout=2400)
    except subprocess.TimeoutExpired:
        for _, p2, _ in procs:
            p2.kill()
        raise
    fh.close()
    log(f"eval shard {i} rc={rc}")
    tail = (LOG / f"eval_shard{i}.log").read_text(errors="replace").splitlines()[-40:]
    print("\n".join(tail), flush=True)
    if rc:
        raise RuntimeError(f"eval shard {i} failed")

# --- merge the shards and summarise the gate
rows = []
for i in range(2):
    part = json.loads((WORK / "k12_eval" / f"report_part{i}.json").read_text())
    rows += part["rows"]
rows.sort(key=lambda r: r["index"])
last = "s04_" + ck[-1].stem.split("_")[-1]
summary = {}
for tag in ("t04", "g04", last):
    rs = [r[tag] for r in rows if tag in r]
    if rs:
        summary[tag] = {"n": len(rs),
                        "opacity": round(sum(x["opacity"] for x in rs) / len(rs), 3),
                        "shell": round(sum(x["shell"] for x in rs) / len(rs), 3),
                        "gate": f"{sum(1 for x in rs if x['gate'])}/{len(rs)}"}
report = {"rows": rows, "summary": summary,
          "checkpoints": [c.name for c in ck],
          "scripts_sha": SHA}
(WORK / "k12_eval" / "report.json").write_text(json.dumps(report, indent=2, default=float))
log("summary\n" + json.dumps(summary, indent=2))

# --- one download: PLYs, prepared photos, reports, logs
art = LOG / "k12_artifacts.zip"
with zipfile.ZipFile(art, "w", zipfile.ZIP_DEFLATED) as z:
    for pat in ("k2d/prepared_*.png", "k12_eval/*.ply", "k12_eval/report*.json",
                "k2e/grid.json", "k12/*.log"):
        for f in sorted(WORK.glob(pat)):
            z.write(f, f.relative_to(WORK))
log("artifact", art.name, round(art.stat().st_size / 1e6, 1), "MB")
