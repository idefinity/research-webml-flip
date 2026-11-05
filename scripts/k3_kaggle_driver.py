"""K3 on Kaggle: the driver the notebook cell downloads and runs.

The notebook itself is six lines (fetch this file from HF, run it). This file
holds the sequence so a change is one upload, not one edited cell:

    1. restore  inputs.tgz + the three checkpoints from idefinity/flip-student-2step
    2. setup,prep   clone upstream, 4.46 GB teachers, cache 200 images
    3. matte 7 -> 20  (reads k3/weights/matte.pt, which is matte_full.pt)
    4. export  fp32 ONNX + parity, writes encoder_meta.json
    5. e2e     teacher DiT 2-step, student vs teacher conditions

Everything lands in /kaggle/working/k3, which Kaggle saves as the version output.

Why not huggingface_hub for the restore: resolving a file through hf-mirror
loses the x-repo-commit header, and the hub client reads that as "not on
huggingface.co". Plain GET on /resolve/main/<path> works on both hosts, so the
restore goes that way and tries huggingface.co first, the mirror second.

Env: FLIP_NB_STAGES (default "restore,setup,prep,train,export,e2e"),
     FLIP_EPOCHS (matte target, default 20), FLIP_HF_BASE (force one host).
"""

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

WORK = pathlib.Path(os.environ.get("FLIP_WORK", "/kaggle/working"))
OUT = WORK / "k3"
REPO_ID = "idefinity/flip-student-2step"
K3 = WORK / "k3_encoder_distill.py"
STAGES = [s.strip() for s in os.environ.get(
    "FLIP_NB_STAGES", "restore,setup,prep,train,export,e2e").split(",") if s.strip()]
EPOCHS = os.environ.get("FLIP_EPOCHS", "20")
BASES = [b for b in (os.environ.get("FLIP_HF_BASE"),
                     "https://huggingface.co", "https://hf-mirror.com") if b]

os.environ.setdefault("PYTHONUNBUFFERED", "1")
if not os.environ.get("FLIP_HF_BASE"):
    # A mirror in HF_ENDPOINT breaks snapshot_download (its HEAD loses the
    # x-repo-commit header). Off a CUDA box with real internet, go direct.
    os.environ.pop("HF_ENDPOINT", None)


def p(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def probe():
    for base in BASES:
        t0 = time.time()
        try:
            req = urllib.request.Request(f"{base}/api/models/{REPO_ID}",
                                         headers={"User-Agent": "flip-k3"})
            with urllib.request.urlopen(req, timeout=25) as fh:
                p("probe", base, fh.status, f"{time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            p("probe", base, "FAIL", type(exc).__name__, str(exc)[:120])


def get(url, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "flip-k3"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read()


def fetch(path, dst, timeout=600):
    dst = pathlib.Path(dst)
    if dst.exists() and dst.stat().st_size > 1024:
        p("have", dst.name, dst.stat().st_size)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for base in BASES:
        try:
            t0 = time.time()
            blob = get(f"{base}/{REPO_ID}/resolve/main/{path}", timeout)
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(blob)
            os.replace(tmp, dst)
            p("got", dst.name, dst.stat().st_size,
              f"{time.time() - t0:.0f}s from", base)
            return dst
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:100]}"
            p("  miss", base, err)
    raise RuntimeError(f"cannot fetch {path}: {err}")


def restore():
    W = OUT / "weights"
    W.mkdir(parents=True, exist_ok=True)
    fetch("encoder/code/k3_encoder_distill.py", K3)
    fetch("encoder/code/k3_sheet.py", WORK / "k3_sheet.py")
    if not (WORK / "inputs").is_dir():
        tgz = fetch("encoder/checkpoints/inputs.tgz", WORK / "dl/inputs.tgz")
        with tarfile.open(tgz) as tf:
            tf.extractall(WORK)
    for src, dst in (("matte_full.pt", "matte.pt"),
                     ("dino_model.pt", "dino.pt"),
                     ("vae_model.pt", "vae.pt")):
        fetch(f"encoder/checkpoints/{src}", W / dst)
    imgs = sorted((WORK / "inputs").rglob("*.jp*g"))
    p("inputs", len(imgs), "->", imgs[0].name if imgs else None)


def run_k3(stages):
    env = dict(os.environ, FLIP_WORK=str(WORK), FLIP_IMAGE_DIR=str(WORK / "inputs"),
               FLIP_N_IMG="200", FLIP_EPOCHS=EPOCHS)
    cmd = [sys.executable, str(K3), "--stages", stages]
    p("$", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def load_k3():
    os.environ.update(
        FLIP_WORK=str(WORK), FLIP_DINO_ARCH="conv", FLIP_DINO_DIM="512",
        FLIP_DINO_DEPTH="8", FLIP_FP16="0",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    spec = importlib.util.spec_from_file_location("k3", str(K3))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.EPOCHS = int(EPOCHS)
    return mod


def main():
    p("stages", STAGES, "epochs", EPOCHS, "py", sys.version.split()[0])
    probe()
    if "restore" in STAGES:
        restore()
    if "setup" in STAGES or "prep" in STAGES:
        run_k3(",".join(s for s in STAGES if s in ("setup", "prep")))
    k3 = None
    if "train" in STAGES:
        k3 = load_k3()
        p("matte: resume from weights/matte.pt, target epoch", EPOCHS)
        k3.train_matte("cuda")
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if "export" in STAGES:
        k3 = k3 or load_k3()
        p("export")
        p("meta", k3.export())
    if "e2e" in STAGES:
        k3 = k3 or load_k3()
        p("e2e")
        k3.e2e()
        p((OUT / "e2e.json").read_text())
    if "sheet" in STAGES:
        # four renders per photo: teacher 20 / teacher 2 / student 2 / browser grid.
        # Images come from FLIP_SHEET_DIR (default WORK/eval; WORK/inputs also works).
        env = dict(os.environ, FLIP_WORK=str(WORK))
        src = os.environ.get("FLIP_SHEET_DIR", str(WORK / "eval"))
        cmd = [sys.executable, str(WORK / "k3_sheet.py"), src]
        p("$", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)
    p("outputs under", OUT)
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.suffix in (".onnx", ".json", ".pt", ".txt"):
            p(" ", f.relative_to(WORK), f.stat().st_size)


if __name__ == "__main__":
    main()
