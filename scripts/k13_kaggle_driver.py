"""k13: native-grid distill on 48 balanced jandan photos, one T4.

k12 showed the recipe works (4 photos, 4/4 gate). k13 asks whether it holds on
real jandan inputs with front/side/back balance: 16 per view, six held out
(two per view, the first six after interleaving), teacher conditioning, the
frozen index grid, two epochs, then decode only the holdout and gate it.

The trace and distill scripts are the ac72202 bytes that produced k12's result,
fetched by SHA over two hosts. Nothing here falls back: a missing trace, a
missing checkpoint or a non-zero stage stops the run and leaves the log.
"""

import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

WORK = Path("/kaggle/working")
LOG = WORK / "k13"
LOG.mkdir(parents=True, exist_ok=True)
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
DS = WORK / "photos"
SHA = "ac72202"
GITHUB = "https://raw.githubusercontent.com/idefinity/research-webml-flip/{}/scripts/{}"
FORCE = {s.strip() for s in os.environ.get("FLIP_FORCE", "").split(",") if s.strip()}
N_PHOTOS = 48
HOLDOUT = 6

# p<idx>_<view>.jpg|view|url|bytes -- first 16 of each bucket in score order,
# round-robin so traces[:HOLDOUT] is two fronts, two sides, two backs.
MAN = """
p000_front.jpg|front|https://img.wangmoyu.com/mw600/05d658eda5fbb5a00d75eae8bbd348de.jpeg|163088
p001_side.jpg|side|https://img.wangmoyu.com/mw600/08bd4eca00cca2b22124834e01bbc727.jpg|174740
p002_back.jpg|back|https://img.toto.im/mw600/0084k9iNgy1iiddbb40qtj30u016oq7h.jpg|131121
p003_front.jpg|front|https://img.wangmoyu.com/mw600/ff61032a8d70e1ee9f29173c0670c5ef.jpeg|146210
p004_side.jpg|side|https://img.wangmoyu.com/mw600/6ab69aa1726859a038010a0578a6e3bc.jpeg|130734
p005_back.jpg|back|https://img.wangmoyu.com/mw600/0ca3c9cebbf0d65cc8cce8c82de6b2a9.jpg|79621
p006_front.jpg|front|https://img.wangmoyu.com/mw1024/ca9b9c698d90c66a0732bea50e18884e.jpg|279560
p007_side.jpg|side|https://img.wangmoyu.com/mw1024/bc262413d4d9e81ee0c9f88fca80640d.jpg|158887
p008_back.jpg|back|https://img.wangmoyu.com/mw600/bff11bcebfbe08172739fe56a739f00d.jpg|137261
p009_front.jpg|front|https://img.wangmoyu.com/mw600/48bccc77ca26a25247494b34e5dc37f0.jpeg|127664
p010_side.jpg|side|https://img.wangmoyu.com/mw600/906aa2d311b3b81bca1c3045656c44c3.jpg|199439
p011_back.jpg|back|https://img.wangmoyu.com/mw600/8fcdee501ceacebee85fb48769f30ba2.jpg|143147
p012_front.jpg|front|https://img.wangmoyu.com/mw1024/4b0747d26a2085653d7f3c8e647411be.jpg|535357
p013_side.jpg|side|https://img.wangmoyu.com/mw1024/0c4525739592ca4efc1069c08cc94932.jpg|248139
p014_back.jpg|back|https://img.toto.im/mw600/006U9dmFly1ihy8mds4opj30xi11je6t.jpg|280946
p015_front.jpg|front|https://img.wangmoyu.com/mw600/68eec8336a88f2d9ef8fafb514ff2705.jpeg|86459
p016_side.jpg|side|https://img.wangmoyu.com/mw1024/49ef6dcdb9a44455416af8d9c556ac71.jpg|389412
p017_back.jpg|back|https://img.wangmoyu.com/mw600/566331ac6177a50b4d64fd6293fbe388.jpg|83632
p018_front.jpg|front|https://img.wangmoyu.com/mw600/f65ba8db1ea03f4688b770a382224bf0.jpg|190180
p019_side.jpg|side|https://img.wangmoyu.com/mw600/f2477a22a70f607ba1d174895ec76b97.png|646635
p020_back.jpg|back|https://img.wangmoyu.com/mw600/543ff5cdc87fb84097f57ae6e83e62a2.jpg|159912
p021_front.jpg|front|https://img.wangmoyu.com/mw600/19825aa4fbc70ef49a4cb13f85915146.jpg|163525
p022_side.jpg|side|https://img.wangmoyu.com/mw600/21360ba7436c05eae9c540dd24fd2680.jpg|209039
p023_back.jpg|back|https://img.wangmoyu.com/mw600/b759601647b112975052f2d4776e3882.png|415140
p024_front.jpg|front|https://img.wangmoyu.com/mw1024/144bad4daacb71e666717f858e513dc0.jpg|485255
p025_side.jpg|side|https://img.wangmoyu.com/mw600/bfd79ba2f5aafeec0e91882bde00db76.jpg|152808
p026_back.jpg|back|https://img.wangmoyu.com/mw600/d06486b010b27509f1b59768e4269398.jpg|224780
p027_front.jpg|front|https://img.wangmoyu.com/mw1024/509e653b8c6d80e683bfd406f6783d7a.jpg|248636
p028_side.jpg|side|https://img.wangmoyu.com/mw1024/545efaf57367f445c2502a8e03459e2d.jpg|493693
p029_back.jpg|back|https://img.wangmoyu.com/mw600/aa78a341362ccf1fc2b7881ef0b297d9.png|446182
p030_front.jpg|front|https://img.wangmoyu.com/mw1024/4d2450bce26835c294289605b0608306.jpg|285790
p031_side.jpg|side|https://img.wangmoyu.com/mw600/8c68e54c507d684921ba80807e85cf4b.jpg|149439
p032_back.jpg|back|https://img.wangmoyu.com/mw600/01ccda2289e0475d73e83c5f51437f81.jpg|138917
p033_front.jpg|front|https://img.wangmoyu.com/mw1024/7265b6b50aa25dc4270ec4ba30efafb0.jpg|403004
p034_side.jpg|side|https://img.wangmoyu.com/mw600/959724dbaedb125731f37c5cae7e8c9a.png|639528
p035_back.jpg|back|https://img.wangmoyu.com/mw600/adaaf7a5977c21dccb7c7cac7ef678b8.jpg|164849
p036_front.jpg|front|https://img.wangmoyu.com/mw600/766215cfb4ad75991e0c586dd1829829.jpg|98739
p037_side.jpg|side|https://img.wangmoyu.com/mw1024/628d9961358d39889e8087ee8d07286f.jpg|436331
p038_back.jpg|back|https://img.wangmoyu.com/mw600/8638376e097c29ef699991d830071714.jpg|206717
p039_front.jpg|front|https://img.wangmoyu.com/mw600/b87eb372ebbd41818841c6fed8a5410f.png|425273
p040_side.jpg|side|https://img.wangmoyu.com/mw600/8d435d74f4ab8571a43f6bba871cbb39.jpg|213411
p041_back.jpg|back|https://img.wangmoyu.com/mw600/3126595094d6c9e8da21926d6215fc9e.jpg|135930
p042_front.jpg|front|https://img.wangmoyu.com/mw600/fc2b917274429604694efc5732e1e58c.jpeg|94087
p043_side.jpg|side|https://img.wangmoyu.com/mw600/2fcb5de5a3710c4beaad569555c31973.jpg|159007
p044_back.jpg|back|https://img.wangmoyu.com/mw600/b71bd6585e13fc228822d815f323ef46.png|434895
p045_front.jpg|front|https://img.wangmoyu.com/mw600/7eee3cdce7d353d9066e8a57804e7631.jpg|147196
p046_side.jpg|side|https://img.wangmoyu.com/mw600/15443c902feccb0daea75914f029603d.jpg|170286
p047_back.jpg|back|https://img.wangmoyu.com/mw600/ff11de3f474fe164b0e1382cb97f208c.jpg|145564
""".strip().splitlines()

# ac72202's bytes, the pair that produced k12's 4/4 on Colab and Kaggle.
WANT = {"k2d_teacher_trace.py": "7f5f450ac9c6e9ed",
        "k2e_distill.py": "ffe27e5c79c6c293"}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def stage(name, cmd, env=None, timeout=3600, allow_fail=False):
    log("$", " ".join(map(str, cmd)))
    t0 = time.time()
    path = LOG / f"{name}.log"
    with open(path, "w") as fh:
        r = subprocess.run([str(c) for c in cmd], env=env, stdout=fh,
                           stderr=subprocess.STDOUT, timeout=timeout)
    text = path.read_text(errors="replace")
    log(f"{name} rc={r.returncode} {time.time() - t0:.0f}s")
    print("\n".join(text.splitlines()[-30:]), flush=True)
    if r.returncode and not allow_fail:
        raise RuntimeError(f"{name} failed rc={r.returncode} (log {path})")
    return r.returncode, text


def clean_env(**kw):
    env = dict(os.environ, PYTHONUNBUFFERED="1", **kw)
    env.pop("HF_ENDPOINT", None)
    return env


def get_photos():
    DS.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in MAN:
        name, bucket, url, size = line.split("|")
        rows.append((name, bucket, url, int(size)))
    assert len(rows) == N_PHOTOS, len(rows)
    for name, bucket, url, size in rows:
        dst = DS / name
        if dst.exists() and dst.stat().st_size == size:
            continue
        err = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/122.0 Safari/537.36",
                    "Referer": "https://jandan.net/"})
                with urllib.request.urlopen(req, timeout=60) as fh:
                    blob = fh.read()
                if len(blob) != size:
                    raise RuntimeError(f"size {len(blob)} != {size}")
                im = Image.open(io.BytesIO(blob))
                im.load()
                if min(im.size) < 200:
                    raise RuntimeError(f"image is {im.size}")
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(blob)
                os.replace(tmp, dst)
                err = None
                break
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {str(exc)[:90]}"
                time.sleep(2 + attempt * 3)
        if err:
            raise RuntimeError(f"cannot fetch {name} from {url}: {err}")
    files = sorted(DS.glob("*.jpg"))
    assert len(files) == N_PHOTOS, f"{len(files)} photos"
    counts = {b: sum(1 for r in rows if r[1] == b) for b in ("front", "side", "back")}
    log("photos", len(files), "MB", round(sum(f.stat().st_size for f in files) / 1e6, 1),
        counts)
    sheet = LOG / "sheet_dataset.jpg"
    cols, cell = 12, 160
    canvas = Image.new("RGB", (cols * cell, 4 * (cell + 16)), (18, 18, 18))
    from PIL import ImageDraw  # noqa: PLC0415
    dr = ImageDraw.Draw(canvas)
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        im.thumbnail((cell, cell), Image.LANCZOS)
        x = (i % cols) * cell
        y = (i // cols) * (cell + 16)
        canvas.paste(im, (x + (cell - im.width) // 2, y + 16))
        dr.text((x + 4, y + 2), f.stem.split("_")[1][:4], fill=(230, 230, 230))
    canvas.save(sheet, quality=90)
    log("wrote", sheet, canvas.size)


def get_repo():
    rc, _ = stage("pip", [sys.executable, "-m", "pip", "install", "-q",
                          "--no-cache-dir", "huggingface_hub", "safetensors",
                          "torchvision", "tqdm", "Pillow"], timeout=1800)
    if not (REPO / ".git").exists():
        stage("clone", ["git", "clone", "-q",
                        "https://github.com/VAST-AI-Research/TripoSplat.git",
                        str(REPO)], timeout=900)
    stage("checkout", ["git", "-C", str(REPO), "checkout", "-q", "d8db9e0"],
          timeout=300)
    if not (CKPT / ".complete").exists():
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        log("downloading VAST-AI/TripoSplat")
        t0 = time.time()
        snapshot_download(repo_id="VAST-AI/TripoSplat", local_dir=str(CKPT),
                          token=False)
        (CKPT / ".complete").touch()
        log(f"weights {time.time() - t0:.0f}s")
    log("ckpts GB", round(sum(f.stat().st_size for f in CKPT.rglob("*")
                              if f.is_file()) / 1e9, 2))
    # k2d picks the repo's four examples before FLIP_IMAGE_DIR; those four are
    # exactly the conditioning diversity k12 was criticised for, so take every
    # image under static/ out of the candidate list.
    dropped = 0
    for p in sorted((REPO / "static").rglob("*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            p.unlink()
            dropped += 1
    log("removed example inputs under static/:", dropped)


def fetch_scripts():
    def fetch(name):
        urls = [GITHUB.format(SHA, name),
                "https://cdn.jsdelivr.net/gh/idefinity/research-webml-flip@"
                f"{SHA}/scripts/{name}"]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "flip-k13"})
                with urllib.request.urlopen(req, timeout=60) as fh:
                    blob = fh.read()
                sha = hashlib.sha256(blob).hexdigest()
                if sha[:16] != WANT[name]:
                    log("  wrong sha", url, sha[:16])
                    continue
                (WORK / name).write_bytes(blob)
                log(name, len(blob), "sha", sha[:16], "from", url.split("/")[2])
                return
            except Exception as exc:  # noqa: BLE001
                log("  miss", url.split("/")[2], type(exc).__name__, str(exc)[:80])
        raise RuntimeError(f"cannot fetch {name}")

    for name in WANT:
        fetch(name)


def trace():
    env = clean_env(FLIP_TRACE_N=str(N_PHOTOS), FLIP_TRACE_STEPS="20",
                    FLIP_TRACE_GAIN="3.0", FLIP_TRACE_SHIFT="3.0",
                    FLIP_TRACE_COND="teacher", FLIP_FORK="auto",
                    FLIP_IMAGE_DIR=str(DS))
    stage("trace", [sys.executable, str(WORK / "k2d_teacher_trace.py")],
          env=env, timeout=5400)
    traces = sorted((WORK / "k2d").glob("trace_*.pt"))
    log("traces", len(traces))
    if len(traces) != N_PHOTOS:
        raise RuntimeError(f"{len(traces)} traces, expected {N_PHOTOS}")
    meta = json.loads((WORK / "k2d" / "trace.json").read_text())
    order = [(r["index"], r["image"], r.get("sample_s")) for r in meta["images"]]
    log("trace order", order[:8], "...", order[-2:])
    sheet = LOG / "sheet_prepared.jpg"
    pre = sorted((WORK / "k2d").glob("prepared_*.png"))
    cols, cell = 12, 160
    canvas = Image.new("RGB", (cols * cell, 4 * (cell + 16)), (18, 18, 18))
    from PIL import ImageDraw  # noqa: PLC0415
    dr = ImageDraw.Draw(canvas)
    for i, f in enumerate(pre):
        im = Image.open(f).convert("RGB")
        im.thumbnail((cell, cell), Image.LANCZOS)
        x = (i % cols) * cell
        y = (i // cols) * (cell + 16)
        canvas.paste(im, (x + (cell - im.width) // 2, y + 16))
        dr.text((x + 4, y + 2), f.stem.replace("prepared_", "p"), fill=(230, 230, 230))
    canvas.save(sheet, quality=90)
    log("wrote", sheet, canvas.size)


def train():
    env = clean_env(FLIP_TRACE_DIR=str(WORK / "k2d"), FLIP_GRID="index",
                    FLIP_STEPS="4", FLIP_EPOCHS="2", FLIP_HOLDOUT=str(HOLDOUT),
                    FLIP_SAVE_EVERY="1", FLIP_CKPT="1", OMP_NUM_THREADS="4")
    stage("train", [sys.executable, str(WORK / "k2e_distill.py")],
          env=env, timeout=7200)
    ck = sorted((WORK / "k2e").glob("student_e*.pt"))
    log("checkpoints", [(c.name, round(c.stat().st_size / 1e9, 2)) for c in ck])
    if len(ck) < 2:
        raise RuntimeError(f"expected 2 epoch checkpoints, got {[c.name for c in ck]}")


def evaluate():
    env = clean_env(FLIP_EVAL_N=str(HOLDOUT), OMP_NUM_THREADS="4")
    stage("eval", [sys.executable, str(WORK / "k13_eval.py")],
          env=env, timeout=3600)
    rep = json.loads((WORK / "k13_eval" / "report.json").read_text())
    log("summary\n" + json.dumps(rep.get("summary", {}), indent=2))


def bundle():
    art = LOG / "k13_artifacts.zip"
    with zipfile.ZipFile(art, "w", zipfile.ZIP_DEFLATED) as z:
        for pat in ("k2d/prepared_*.png", "k2d/trace.json",
                    "k13_eval/*.ply", "k13_eval/report.json",
                    "k2e/grid.json", "k2e/report.json",
                    "k13/*.log", "k13/sheet_*.jpg"):
            for f in sorted(WORK.glob(pat)):
                z.write(f, f.relative_to(WORK))
    log("artifact", art.name, round(art.stat().st_size / 1e6, 1), "MB")


def main():
    log("gpu", subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                              text=True).stdout.strip().replace("\n", " | "))
    log("force", sorted(FORCE))
    get_photos()
    get_repo()
    fetch_scripts()

    have = len(list((WORK / "k2d").glob("trace_*.pt")))
    if "trace" in FORCE or have != N_PHOTOS:
        log(f"traces {have}/{N_PHOTOS} - tracing")
        trace()
    else:
        log(f"traces {have}/{N_PHOTOS} - reusing")

    if "train" in FORCE or not (WORK / "k2e" / "student_e002.pt").exists():
        train()
    else:
        log("training done - reusing")

    if "eval" in FORCE or not (WORK / "k13_eval" / "report.json").exists():
        evaluate()
    else:
        log("eval done - reusing")

    bundle()
    log("done")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback  # noqa: PLC0415
        log(traceback.format_exc())
        try:
            bundle()
        except Exception:  # noqa: BLE001
            log("bundle after failure also failed")
        raise
