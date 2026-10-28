"""K2s - get the upstream source out of Kaggle and somewhere I can read it.

The laptop cannot reach github.com (the proxy answers 502 to CONNECT), so every
previous kernel was written blind: module names, argument order and the shape of
the decode stage all came from guesswork and from what a traceback happened to
reveal. That is the wrong way to write an exporter.

This kernel does one cheap thing: clone TripoSplat at the pinned commit, tar the
sources, concatenate every .py into one text file, and push both to the hub
under idefinity/flip-web. From there hf-mirror can serve them to the laptop.

No GPU, no weights, a couple of minutes.
"""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

W = Path("/kaggle/working")
REPO = W / "TripoSplat"
UPSTREAM = "https://github.com/VAST-AI-Research/TripoSplat.git"
SHA = "d8db9e0"
OUT = W / "k2s"
OUT.mkdir(parents=True, exist_ok=True)
HF_TOKEN = "__TOKEN__"
TARGET = "idefinity/flip-web"

log = []


def p(*a):
    s = " ".join(str(x) for x in a)
    log.append(s)
    print(s, flush=True)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    p("$", " ".join(cmd)[:120], "-> rc", r.returncode)
    if r.returncode != 0:
        p("   ", (r.stderr or r.stdout or "")[-400:].replace("\n", " "))
    return r


SKIP_DIRS = {".git", "ckpts", "__pycache__", "assets", "static"}
SKIP_SUFFIX = {".safetensors", ".bin", ".pt", ".pth", ".png", ".jpg", ".jpeg",
               ".webp", ".ply", ".glb", ".npz", ".npy", ".tar", ".gz"}


def want(path: Path) -> bool:
    rel = path.relative_to(REPO)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    if path.suffix.lower() in SKIP_SUFFIX:
        return False
    return path.is_file()


try:
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
         "huggingface_hub", "safetensors", "torchvision", "tqdm"], timeout=900)

    if not (REPO / ".git").exists():
        run(["git", "clone", "-q", UPSTREAM, str(REPO)], timeout=600)
    run(["git", "-C", str(REPO), "checkout", "-q", SHA], timeout=300)
    p("HEAD", run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip())

    files = sorted(q for q in REPO.rglob("*") if want(q))
    p("files kept", len(files))

    # ---- 1. the tar, so nothing is silently reformatted on the way out
    tar_path = OUT / "upstream_src.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(REPO)))
    p("tar", round(tar_path.stat().st_size / 1e6, 2), "MB")

    # ---- 2. one text file, so a grep on the laptop answers in milliseconds
    txt_path = OUT / "upstream_src.txt"
    with open(txt_path, "w") as out:
        out.write(f"TripoSplat @ {SHA}\n")
        out.write("=" * 72 + "\n")
        for f in files:
            rel = f.relative_to(REPO)
            out.write("\n" + "=" * 72 + f"\n===== {rel} =====\n" + "=" * 72 + "\n")
            try:
                out.write(f.read_text(errors="replace"))
            except Exception as e:  # noqa: BLE001
                out.write(f"[unreadable: {e}]\n")
            out.write("\n")
    p("txt", round(txt_path.stat().st_size / 1e6, 2), "MB")

    # ---- 3. the layout, and the one class whose signature decides everything
    info = {"sha": SHA, "files": [str(f.relative_to(REPO)) for f in files]}
    sys.path.insert(0, str(REPO))
    try:
        import inspect

        import triposplat as T
        info["triposplat_file"] = getattr(T, "__file__", "?")
        info["dir"] = [n for n in dir(T) if not n.startswith("_")]
        try:
            info["pipeline_source"] = inspect.getsource(T.TripoSplatPipeline)
        except Exception as e:  # noqa: BLE001
            info["pipeline_source"] = f"[no source: {type(e).__name__}: {e}]"
        p("imported triposplat:", info["triposplat_file"])
        p("pipeline source", len(info["pipeline_source"]), "chars")
    except Exception:  # noqa: BLE001
        info["import_error"] = traceback.format_exc()[-1200:]
        p("import triposplat FAILED")
        p(traceback.format_exc()[-600:].replace("\n", " | "))

    (OUT / "layout.json").write_text(json.dumps(info, indent=2))

    # ---- 4. publish
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(TARGET, private=False, exist_ok=True, repo_type="model")
    for f in (tar_path, txt_path, OUT / "layout.json"):
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f"upstream/{f.name}",
                        repo_id=TARGET, repo_type="model")
        p("uploaded", f.name, round(f.stat().st_size / 1e6, 2), "MB",
          "sha256", hashlib.sha256(f.read_bytes()).hexdigest()[:16])
    p("done -> https://huggingface.co/" + TARGET)
except Exception:  # noqa: BLE001
    p("FAILED")
    p(traceback.format_exc()[:2500])

(OUT / "log.txt").write_text("\n".join(log))
print("end", flush=True)
