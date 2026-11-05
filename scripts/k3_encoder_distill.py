"""K3 - distil the front end: photo -> (feature1, feature2), plus the matte.

The DiT is done and the decoder is done. What is still missing is the half that
turns a photo into the two conditioning tensors the DiT was trained on:

    feature1 [1,4101,1280]  DINOv3 ViT-H/16+ output, layer-normed over 1280
    feature2 [1,4096, 128]  Flux2-VAE encoder mean, BatchNorm-standardised

and the alpha matte that decides what the photo even is. All three teachers are
too big for a browser (DINO 840 M, VAE 30 M of pure convolution, BiRefNet 220 M
with deformable convs), so each gets a student:

    matte   small U-Net                   ~2 M   image -> alpha (any H,W)
    dino    ConvNeXt-style token encoder  ~19 M  1024^2 -> 4101x1280
    vae     small strided conv encoder    ~5 M   1024^2 -> 4096x128

Why a conv student for DINO instead of a small ViT: the contract is 4101 tokens,
and a ViT at that length materialises a 4101x4101 score matrix per head unless
the attention is fused. A ConvNeXt block has no such matrix, exports to plain
Conv/LayerNorm/MatMul, and the ORT WebGPU conv path already measured 711 GFLOP/s
(PUBLISH.md section 2). The five prefix tokens (cls + 4 registers) are produced
from global average pooling, which is what those tokens mostly carry anyway.

Stages (FLIP_STAGES, default all):

    setup    clone upstream, download the four teacher checkpoints
    prep     run the teachers once, cache per-image targets + matte masks
    train    train the three students (resumable, checkpoints in OUT)
    export   ONNX + torch-vs-ORT parity, writes encoder_meta.json
    e2e      teacher DiT 2-step + decoder on student conditions vs teacher ones

Everything is sized for a single Colab/Kaggle T4. Defaults: 200 images, 40
epochs, batch 4 for the conv students, batch 1 for matte.

Env:
    FLIP_WORK        /content or /kaggle/working (auto)
    FLIP_OUT         output dir (default WORK/k3)
    FLIP_STAGES      comma list, default "setup,prep,train,export,e2e"
    FLIP_N_IMG       images to trace/train on (default 200)
    FLIP_EPOCHS      epochs per student (default 40)
    FLIP_BATCH       batch size for dino/vae (default 4)
    FLIP_IMG_HF      HF image dataset (default DownFlow/fuliji)
    FLIP_IMAGE_DIR   extra local images
    FLIP_DINO_ARCH   conv (default) or vit
    FLIP_DINO_DIM    student width (default 512)
    FLIP_DINO_DEPTH  blocks (default 8)
    FLIP_FP16        export fp16 instead of fp32 (default 0)
    FLIP_UPLOAD      1 = push encoder/ to idefinity/flip-web
    HF_TOKEN         upload token (or Kaggle secret / Colab userdata)

Run:
    python3 k3_encoder_distill.py            # all stages
    FLIP_STAGES=prep python3 k3_encoder_distill.py
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

WORK = Path(os.environ.get("FLIP_WORK")
            or ("/content" if Path("/content").exists() else "/kaggle/working"))
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = Path(os.environ.get("FLIP_OUT", str(WORK / "k3")))
CACHE = OUT / "cache"
WEIGHTS = OUT / "weights"
ONNX = OUT / "onnx"

SHA = "d8db9e0"
UPSTREAM = "https://github.com/VAST-AI-Research/TripoSplat.git"
TEACHER_REPO = "VAST-AI/TripoSplat"
PUBLISH_REPO = "idefinity/flip-student-2step"

N_IMG = int(os.environ.get("FLIP_N_IMG", "200"))
EPOCHS = int(os.environ.get("FLIP_EPOCHS", "40"))
BATCH = int(os.environ.get("FLIP_BATCH", "4"))
LR = float(os.environ.get("FLIP_LR", "3e-4"))
FP16 = os.environ.get("FLIP_FP16", "0") == "1"
UPLOAD = os.environ.get("FLIP_UPLOAD", "0") == "1"
IMG_HF = os.environ.get("FLIP_IMG_HF", "DownFlow/fuliji")
IMG_DIR = os.environ.get("FLIP_IMAGE_DIR", "")
STAGES = [s.strip() for s in os.environ.get(
    "FLIP_STAGES", "setup,prep,train,export,e2e").split(",") if s.strip()]

CANVAS = 1024
DINO_OUT_TOKENS = 4101
DINO_DIM_OUT = 1280
VAE_TOKENS = 4096
VAE_DIM = 128
MATTE_SIZE = CANVAS

LOG = []


def p(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def note(msg):
    p(msg)


def run(cmd, timeout=1800, env=None, check=False):
    p("$", " ".join(str(c) for c in cmd)[:160])
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       timeout=timeout, env=env)
    if r.returncode != 0:
        p("   ", (r.stderr or r.stdout or "")[-500:].replace("\n", " "))
        if check:
            raise RuntimeError(f"command failed: {cmd}")
    return r


def ensure_dirs():
    for d in (OUT, CACHE, WEIGHTS, ONNX):
        d.mkdir(parents=True, exist_ok=True)


def hf_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient  # noqa: PLC0415
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001
        return None
    try:
        from google.colab import userdata  # noqa: PLC0415
        return userdata.get("HF_TOKEN")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def deps():
    pkgs = ["huggingface_hub", "safetensors", "torchvision", "tqdm",
            "onnx", "onnxruntime", "Pillow"]
    p("installing", " ".join(pkgs))
    run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *pkgs],
        timeout=1800)


def setup():
    deps()
    if not (REPO / ".git").exists():
        run(["git", "clone", "-q", UPSTREAM, str(REPO)], timeout=900, check=True)
    run(["git", "-C", str(REPO), "checkout", "-q", SHA], timeout=300)
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    if not (CKPT / ".complete").exists():
        p("downloading teacher checkpoints (~3.5 GB)")
        snapshot_download(repo_id=TEACHER_REPO, local_dir=str(CKPT),
                          token=hf_token())
        (CKPT / ".complete").touch()
    total = sum(f.stat().st_size for f in CKPT.rglob("*") if f.is_file())
    p("ckpts", round(total / 1e9, 2), "GB")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def fetch_hf(n):
    if not IMG_HF:
        return []
    dst = WORK / "inputs"
    dst.mkdir(parents=True, exist_ok=True)
    column = os.environ.get("FLIP_IMG_HF_COLUMN", "image").strip()
    split = os.environ.get("FLIP_IMG_HF_SPLIT", "train").strip()

    def get(url, timeout=90):
        req = urllib.request.Request(url, headers={"User-Agent": "flip-k3"})
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return fh.read()

    cfg = None
    try:
        q = urllib.parse.urlencode({"dataset": IMG_HF})
        splits = json.loads(get(
            f"https://datasets-server.huggingface.co/splits?{q}"))["splits"]
        if splits:
            cfg, split = splits[0].get("config"), splits[0].get("split", split)
    except Exception as exc:  # noqa: BLE001
        p("  splits lookup failed", type(exc).__name__)

    got, offset = [], 0
    while len(got) < n:
        q = {"dataset": IMG_HF, "split": split, "offset": offset,
             "length": min(100, n - len(got))}
        if cfg:
            q["config"] = cfg
        try:
            js = json.loads(get("https://datasets-server.huggingface.co/rows?"
                                + urllib.parse.urlencode(q)))
        except Exception as exc:  # noqa: BLE001
            p("  rows failed", type(exc).__name__)
            break
        rows = js.get("rows") or []
        if not rows:
            break
        for r in rows:
            cell = (r.get("row") or {}).get(column)
            src = cell.get("src") if isinstance(cell, dict) else None
            if not src:
                continue
            path = dst / f"hf_{len(got):04d}.jpg"
            if path.exists() and path.stat().st_size > 1024:
                got.append(path)
                continue
            try:
                blob = get(src, timeout=60)
                if blob[:2] != b"\xff\xd8" and blob[:8] != b"\x89PNG\r\n\x1a\n":
                    continue
                tmp = path.with_suffix(".jpg.part")
                tmp.write_bytes(blob)
                os.replace(tmp, path)
                got.append(path)
            except Exception:  # noqa: BLE001
                continue
            if len(got) >= n:
                break
        offset += len(rows)
        if offset >= (js.get("num_rows_total") or 0):
            break
    p("fetched", len(got), "images from", IMG_HF)
    return got


def pick_images():
    imgs = []
    if IMG_DIR:
        root = Path(IMG_DIR)
        if root.is_dir():
            imgs += sorted(q for q in root.rglob("*")
                           if q.suffix.lower() in IMG_EXT)
        elif root.is_file():
            imgs.append(root)
        p("local images", len(imgs))
    if len(imgs) < N_IMG:
        imgs += fetch_hf(N_IMG - len(imgs))
    if len(imgs) < N_IMG:
        ex = REPO / "static" / "example_inputs"
        if ex.is_dir():
            imgs += sorted(q for q in ex.rglob("*")
                           if q.suffix.lower() in IMG_EXT)
    if not imgs:
        raise RuntimeError("no input images: set FLIP_IMG_HF or FLIP_IMAGE_DIR")
    p("inputs", len(imgs), "->", imgs[0], "...", imgs[-1])
    return imgs[:N_IMG]


# ---------------------------------------------------------------------------
# teacher + preprocessing
# ---------------------------------------------------------------------------

def load_teacher():
    import torch  # noqa: F401
    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    dino = T.load_dinov3(str(CKPT / "clip_vision/dino_v3_vit_h.safetensors"),
                         device=dev, dtype=__import__("torch").bfloat16)
    vae = T.load_vae_encoder(str(CKPT / "vae/flux2-vae.safetensors"),
                             device=dev, dtype=__import__("torch").bfloat16)
    rmbg = T.load_rmbg(str(CKPT / "background_removal/birefnet.safetensors"),
                       device=dev, dtype=__import__("torch").float16)
    return T, dino, vae, rmbg, dev


def resize_min_side(image, target=CANVAS):
    w, h = image.size
    s = target / min(w, h)
    w2 = max(32, int(round(w * s / 32)) * 32)
    h2 = max(32, int(round(h * s / 32)) * 32)
    return image.resize((w2, h2), __import__("PIL.Image", fromlist=["Image"]).LANCZOS)


def erode_alpha(alpha, radius=1):
    """MinFilter(2r+1) on the alpha channel, the same shrink upstream applies."""
    from PIL import Image, ImageFilter  # noqa: PLC0415
    im = Image.fromarray(alpha, mode="L")
    return im.filter(ImageFilter.MinFilter(2 * radius + 1))


def square_crop_box(alpha, expand=1.2):
    import numpy as np  # noqa: PLC0415
    a = np.asarray(alpha)
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        h, w = a.shape
        side = min(h, w)
        return (w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * expand
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def make_sample(T, dino, vae, rmbg, path, idx):
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from torchvision import transforms  # noqa: PLC0415

    stem = f"{idx:04d}"
    files = {k: CACHE / f"{k}_{stem}.npy" for k in ("f1", "f2", "canvas")}
    matte = {k: CACHE / f"{k}_{stem}.png" for k in ("matte_rgb", "matte_alpha")}
    if all(f.exists() for f in files.values()) and all(f.exists() for f in matte.values()):
        return True

    image = Image.open(path).convert("RGB")
    rgb = resize_min_side(image)
    with torch.no_grad():
        rgba = rmbg.remove_background(rgb)
    alpha_raw = np.array(rgba.getchannel(3), dtype=np.uint8)
    alpha_er = np.array(erode_alpha(alpha_raw, 1), dtype=np.uint8)

    box = square_crop_box(alpha_er)
    cropped = rgba.crop(box).resize((CANVAS, CANVAS), Image.LANCZOS)
    canvas = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    canvas.paste(cropped, mask=cropped.split()[3])

    dino_dtype = next(dino.parameters()).dtype
    vae_dtype = next(vae.parameters()).dtype
    dino_device = next(dino.parameters()).device
    vae_device = next(vae.parameters()).device
    x = transforms.ToTensor()(canvas).unsqueeze(0)
    with torch.no_grad():
        f1 = dino(pixel_values=T._DINOV3_NORMALIZE(x).to(dino_device, dino_dtype))
        f1 = F.layer_norm(f1.float(), f1.shape[-1:])
        f2 = vae.encode(x.to(vae_device, vae_dtype) * 2 - 1, deterministic=True)

    np.save(files["f1"], f1[0].half().cpu().numpy())
    np.save(files["f2"], f2[0].half().cpu().numpy())
    np.save(files["canvas"], (np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)))
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(matte["matte_rgb"])
    Image.fromarray(alpha_raw).save(matte["matte_alpha"])
    return True


def prep():
    imgs = pick_images()
    T, dino, vae, rmbg, dev = load_teacher()
    p("teacher on", dev)
    t0 = time.time()
    ok = 0
    for i, path in enumerate(imgs):
        try:
            if make_sample(T, dino, vae, rmbg, path, i):
                ok += 1
        except Exception:  # noqa: BLE001
            p("  skip", path.name, traceback.format_exc().splitlines()[-1][:160])
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            p(f"  prep {i + 1}/{len(imgs)}  {dt / (i + 1):.2f} s/img")
    p("prep done", ok, "of", len(imgs))


# ---------------------------------------------------------------------------
# students
# ---------------------------------------------------------------------------

def build_dino():
    import torch
    import torch.nn as nn

    class LN2d(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.ln = nn.LayerNorm(dim, eps=eps)

        def forward(self, x):
            return self.ln(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)

    class ConvBlock(nn.Module):
        """ConvNeXt block: depthwise 7x7 then an inverted bottleneck."""

        def __init__(self, dim, mlp_ratio=4):
            super().__init__()
            hidden = int(dim * mlp_ratio)
            self.norm = LN2d(dim)
            self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            self.pw1 = nn.Conv2d(dim, hidden, 1)
            self.act = nn.GELU()
            self.pw2 = nn.Conv2d(hidden, dim, 1)
            self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

        def forward(self, x):
            h = self.norm(x)
            h = self.dw(h)
            h = self.pw1(h)
            h = self.act(h)
            h = self.pw2(h)
            return x + self.gamma.view(1, -1, 1, 1) * h

    class DinoConvStudent(nn.Module):
        def __init__(self, dim=512, depth=8, patch=16, out_dim=DINO_DIM_OUT):
            super().__init__()
            self.stem = nn.Conv2d(3, dim, patch, patch)
            self.stem_norm = LN2d(dim)
            self.blocks = nn.Sequential(*[ConvBlock(dim) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            self.prefix = nn.Linear(dim, 5 * dim)
            self.proj = nn.Linear(dim, out_dim)
            self.out_norm = nn.LayerNorm(out_dim, elementwise_affine=False)

        def forward(self, x):
            h = self.stem_norm(self.stem(x))
            h = self.blocks(h)
            b, c, hh, ww = h.shape
            tok = h.flatten(2).transpose(1, 2)
            tok = self.norm(tok)
            g = tok.mean(dim=1)
            pre = self.prefix(g).reshape(b, 5, c)
            tok = torch.cat([pre, tok], dim=1)
            return self.out_norm(self.proj(tok))

    class VitBlock(nn.Module):
        def __init__(self, dim, heads=8, mlp_ratio=4):
            super().__init__()
            self.n1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.n2 = nn.LayerNorm(dim)
            hidden = dim * mlp_ratio
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)
            self.heads = heads

        def forward(self, x):
            b, n, c = x.shape
            qkv = self.qkv(self.n1(x)).reshape(b, n, 3, self.heads, c // self.heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            att = (q @ k.transpose(-2, -1)) * (c // self.heads) ** -0.5
            att = att.softmax(dim=-1)
            o = (att @ v).transpose(1, 2).reshape(b, n, c)
            x = x + self.proj(o)
            x = x + self.fc2(torch.nn.functional.gelu(self.fc1(self.n2(x))))
            return x

    class DinoVitStudent(nn.Module):
        def __init__(self, dim=384, depth=8, heads=6, patch=16,
                     out_dim=DINO_DIM_OUT, tokens=4096):
            super().__init__()
            self.stem = nn.Conv2d(3, dim, patch, patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.reg = nn.Parameter(torch.zeros(1, 4, dim))
            self.pos = nn.Parameter(torch.zeros(1, tokens, dim))
            self.blocks = nn.ModuleList([VitBlock(dim, heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            self.proj = nn.Linear(dim, out_dim)
            self.out_norm = nn.LayerNorm(out_dim, elementwise_affine=False)

        def forward(self, x):
            h = self.stem(x).flatten(2).transpose(1, 2) + self.pos
            h = torch.cat([self.cls.expand(h.shape[0], -1, -1),
                           self.reg.expand(h.shape[0], -1, -1), h], dim=1)
            for blk in self.blocks:
                h = blk(h)
            return self.out_norm(self.proj(self.norm(h)))

    arch = os.environ.get("FLIP_DINO_ARCH", "conv")
    dim = int(os.environ.get("FLIP_DINO_DIM", "512"))
    depth = int(os.environ.get("FLIP_DINO_DEPTH", "8"))
    if arch == "vit":
        return DinoVitStudent(dim=min(dim, 512), depth=depth)
    return DinoConvStudent(dim=dim, depth=depth)


def build_vae():
    import torch.nn as nn

    class Res(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.n1 = nn.GroupNorm(8, cin)
            self.c1 = nn.Conv2d(cin, cout, 3, 1, 1)
            self.n2 = nn.GroupNorm(8, cout)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
            self.act = nn.SiLU()
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x):
            h = self.c1(self.act(self.n1(x)))
            h = self.c2(self.act(self.n2(h)))
            return h + self.skip(x)

    class VaeStudent(nn.Module):
        def __init__(self, ch=(64, 128, 256, 256, 256), out_dim=VAE_DIM):
            super().__init__()
            self.inp = nn.Conv2d(3, ch[0], 3, 1, 1)
            self.enc = nn.ModuleList()
            self.down = nn.ModuleList()
            prev = ch[0]
            for c in ch[1:]:
                self.enc.append(Res(prev, c))
                self.down.append(nn.Conv2d(c, c, 3, 2, 1))
                prev = c
            self.mid = nn.Sequential(Res(prev, prev), Res(prev, prev))
            self.out = nn.Conv2d(prev, out_dim, 1)

        def forward(self, x):
            h = self.inp(x)
            for res, down in zip(self.enc, self.down):
                h = down(res(h))
            h = self.mid(h)
            h = self.out(h)
            return h.flatten(2).transpose(1, 2)

    return VaeStudent()


def build_matte():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, 1, 1)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
            self.n1 = nn.GroupNorm(8, cout)
            self.n2 = nn.GroupNorm(8, cout)
            self.act = nn.SiLU()

        def forward(self, x):
            x = self.act(self.n1(self.c1(x)))
            return self.act(self.n2(self.c2(x)))

    class MatteStudent(nn.Module):
        def __init__(self, base=24):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8, base * 8]
            self.e0 = Block(3, c[0])
            self.e1 = Block(c[0], c[1])
            self.e2 = Block(c[1], c[2])
            self.e3 = Block(c[2], c[3])
            self.e4 = Block(c[3], c[4])
            self.pool = nn.MaxPool2d(2)
            self.d3 = Block(c[4] + c[3], c[3])
            self.d2 = Block(c[3] + c[2], c[2])
            self.d1 = Block(c[2] + c[1], c[1])
            self.d0 = Block(c[1] + c[0], c[0])
            self.head = nn.Conv2d(c[0], 1, 1)

        def up(self, x):
            return F.interpolate(x, scale_factor=2, mode="bilinear",
                                 align_corners=False)

        def forward(self, x):
            e0 = self.e0(x)
            e1 = self.e1(self.pool(e0))
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            d3 = self.d3(torch.cat([self.up(e4), e3], 1))
            d2 = self.d2(torch.cat([self.up(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up(d2), e1], 1))
            d0 = self.d0(torch.cat([self.up(d1), e0], 1))
            return torch.sigmoid(self.head(d0))

    return MatteStudent()


def params(model):
    return sum(q.numel() for q in model.parameters()) / 1e6


# ---------------------------------------------------------------------------
# datasets / training
# ---------------------------------------------------------------------------

def cache_items():
    items = sorted(CACHE.glob("f1_*.npy"))
    return [q.name[3:-4] for q in items]


def load_canvas(stem):
    import numpy as np  # noqa: PLC0415
    a = np.load(CACHE / f"canvas_{stem}.npy")
    return a


def train_dino(device):
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    stems = cache_items()
    if not stems:
        raise RuntimeError("no cached targets; run prep first")
    model = build_dino().to(device)
    p(f"dino student {params(model):.1f} M params")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    steps = max(1, (len(stems) // BATCH) * EPOCHS)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ckpt = WEIGHTS / "dino.pt"
    start = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["epoch"]
        for _ in range(start * max(1, len(stems) // BATCH)):
            if sched.last_epoch < steps - 1:
                sched.step()
        p("resume dino at epoch", start)

    for ep in range(start, EPOCHS):
        t0 = time.time()
        order = np.random.RandomState(ep).permutation(len(stems))
        acc = n = 0
        for it in range(0, len(order) - BATCH + 1, BATCH):
            idx = order[it:it + BATCH]
            x = torch.stack([torch.from_numpy(load_canvas(stems[j])).float() / 255.0
                             for j in idx]).to(device)
            tgt = torch.stack([
                torch.from_numpy(np.load(CACHE / f"f1_{stems[j]}.npy")).float()
                for j in idx]).to(device)
            with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                pred = model(x)
                cos = F.cosine_similarity(pred.float(), tgt, dim=-1).mean()
                loss = (1 - cos) + 0.1 * F.mse_loss(pred.float(), tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()
            acc += float(cos.detach())
            n += 1
            if n % 20 == 0:
                p(f"  dino ep{ep} {n} cos {acc / n:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "cos": acc / max(n, 1)},
                   ckpt)
        p(f"dino epoch {ep + 1}/{EPOCHS} cos {acc / max(n, 1):.4f} "
          f"({time.time() - t0:.0f}s)")
    return model


def train_vae(device):
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    stems = cache_items()
    model = build_vae().to(device)
    p(f"vae student {params(model):.1f} M params")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    steps = max(1, (len(stems) // BATCH) * EPOCHS)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ckpt = WEIGHTS / "vae.pt"
    start = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["epoch"]
        for _ in range(start * max(1, len(stems) // BATCH)):
            if sched.last_epoch < steps - 1:
                sched.step()
        p("resume vae at epoch", start)

    for ep in range(start, EPOCHS):
        t0 = time.time()
        order = np.random.RandomState(1000 + ep).permutation(len(stems))
        acc = n = 0
        for it in range(0, len(order) - BATCH + 1, BATCH):
            idx = order[it:it + BATCH]
            x = torch.stack([torch.from_numpy(load_canvas(stems[j])).float() / 255.0
                             for j in idx]).to(device)
            x = x * 2 - 1
            tgt = torch.stack([
                torch.from_numpy(np.load(CACHE / f"f2_{stems[j]}.npy")).float()
                for j in idx]).to(device)
            with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                pred = model(x)
                cos = F.cosine_similarity(pred.float(), tgt, dim=-1).mean()
                loss = (1 - cos) + 0.5 * F.mse_loss(pred.float(), tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()
            acc += float(cos.detach())
            n += 1
            if n % 20 == 0:
                p(f"  vae ep{ep} {n} cos {acc / n:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "cos": acc / max(n, 1)}, ckpt)
        p(f"vae epoch {ep + 1}/{EPOCHS} cos {acc / max(n, 1):.4f} "
          f"({time.time() - t0:.0f}s)")
    return model


def train_matte(device):
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    stems = cache_items()
    model = build_matte().to(device)
    p(f"matte student {params(model):.1f} M params")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = max(1, len(stems) * EPOCHS)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ckpt = WEIGHTS / "matte.pt"
    start = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["epoch"]
        for _ in range(start * len(stems)):
            if sched.last_epoch < steps - 1:
                sched.step()
        p("resume matte at epoch", start)

    for ep in range(start, EPOCHS):
        t0 = time.time()
        order = np.random.RandomState(2000 + ep).permutation(len(stems))
        acc = n = 0
        for j in order:
            rgb = Image.open(CACHE / f"matte_rgb_{stems[j]}.png").convert("RGB")
            alpha = Image.open(CACHE / f"matte_alpha_{stems[j]}.png").convert("L")
            x = torch.from_numpy(np.asarray(rgb).astype(np.float32) / 255.0)
            y = torch.from_numpy(np.asarray(alpha).astype(np.float32) / 255.0)
            x = x.permute(2, 0, 1).unsqueeze(0).to(device)
            y = y.unsqueeze(0).unsqueeze(0).to(device)
            with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                pred = model(x).float()
            bce = F.binary_cross_entropy(pred, y)
            inter = (pred * y).sum()
            dice = 1 - (2 * inter + 1) / (pred.sum() + y.sum() + 1)
            loss = bce + dice
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()
            acc += 1 - float(bce.detach())
            n += 1
            if n % 40 == 0:
                p(f"  matte ep{ep} {n}/{len(stems)} bce {float(bce):.4f} "
                  f"dice {float(dice):.4f}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "bce": 1 - acc / max(n, 1)}, ckpt)
        p(f"matte epoch {ep + 1}/{EPOCHS} "
          f"acc {acc / max(n, 1):.4f} ({time.time() - t0:.0f}s)")
    return model


def train():
    import torch  # noqa: PLC0415
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p("train on", device)
    train_dino(device)
    train_vae(device)
    train_matte(device)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def load_weights(model, name, device):
    import torch  # noqa: PLC0415
    ckpt = WEIGHTS / f"{name}.pt"
    if not ckpt.exists():
        raise RuntimeError(f"missing {ckpt}")
    st = torch.load(ckpt, map_location=device)
    model.load_state_dict(st["model"])
    return model.to(device).eval()


def export_model(model, example, path, inputs, outputs, dynamic=None, fp16=False):
    import torch  # noqa: PLC0415
    model = model.eval()
    if fp16:
        model = model.half()
        example = example.half()
    torch.onnx.export(
        model, example, str(path),
        input_names=inputs, output_names=outputs,
        opset_version=17, do_constant_folding=True,
        dynamic_axes=dynamic, dynamo=False,
    )
    return path


def parity(path, inputs, outputs, torch_out, tol=0.999):
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    providers = ["CPUExecutionProvider"]
    try:
        if ort.get_device() == "GPU":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:  # noqa: BLE001
        pass
    sess = ort.InferenceSession(str(path), providers=providers)
    feed = {k: v.detach().cpu().numpy() for k, v in inputs.items()}
    got = sess.run(None, feed)
    worst = 1.0
    for a, b in zip(got, torch_out):
        a = a.astype(np.float32).reshape(-1)
        b = b.detach().cpu().float().numpy().reshape(-1)
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        cos = float(np.dot(a, b) / denom)
        worst = min(worst, cos)
    return worst, [tuple(g.shape) for g in got]


def export():
    import torch  # noqa: PLC0415
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = {"files": {}, "shapes": {}, "params": {}}

    dino = load_weights(build_dino(), "dino", device)
    ex = torch.rand(1, 3, CANVAS, CANVAS, device=device)
    with torch.no_grad():
        ref = dino(ex)
    path = ONNX / "dino_student.onnx"
    export_model(dino.float(), ex, path, ["image"], ["feature1"], fp16=False)
    cos, shapes = parity(path, {"image": ex}, ["feature1"], [ref])
    p(f"dino onnx {path.stat().st_size / 1e6:.1f} MB parity cos {cos:.6f}")
    meta["files"]["dino"] = path.name
    meta["shapes"]["dino"] = {"input": [1, 3, CANVAS, CANVAS],
                              "output": list(shapes[0])}
    meta["params"]["dino"] = round(params(dino), 2)

    vae = load_weights(build_vae(), "vae", device)
    ex = torch.rand(1, 3, CANVAS, CANVAS, device=device) * 2 - 1
    with torch.no_grad():
        ref = vae(ex)
    path = ONNX / "vae_student.onnx"
    export_model(vae.float(), ex, path, ["image"], ["feature2"], fp16=False)
    cos, shapes = parity(path, {"image": ex}, ["feature2"], [ref])
    p(f"vae onnx {path.stat().st_size / 1e6:.1f} MB parity cos {cos:.6f}")
    meta["files"]["vae"] = path.name
    meta["shapes"]["vae"] = {"input": [1, 3, CANVAS, CANVAS],
                             "output": list(shapes[0])}
    meta["params"]["vae"] = round(params(vae), 2)

    matte = load_weights(build_matte(), "matte", device)
    ex = torch.rand(1, 3, 512, 768, device=device)
    with torch.no_grad():
        ref = matte(ex)
    path = ONNX / "matte_student.onnx"
    export_model(matte.float(), ex, path, ["image"], ["alpha"],
                 dynamic={"image": {2: "h", 3: "w"},
                          "alpha": {2: "h", 3: "w"}})
    cos, shapes = parity(path, {"image": ex}, ["alpha"], [ref])
    p(f"matte onnx {path.stat().st_size / 1e6:.1f} MB parity cos {cos:.6f}")
    meta["files"]["matte"] = path.name
    meta["shapes"]["matte"] = {"input": [1, 3, "h", "w"], "output": [1, 1, "h", "w"]}
    meta["params"]["matte"] = round(params(matte), 2)

    meta.update({
        "preprocess": {
            "resize": "shortest side 1024, dims rounded to /32",
            "matte_input": "rgb / 255",
            "dino_input": "canvas rgb / 255",
            "vae_input": "canvas rgb / 255 * 2 - 1",
            "canonical": CANVAS,
        },
        "tokens": {"dino": DINO_OUT_TOKENS, "vae": VAE_TOKENS},
    })
    (OUT / "encoder_meta.json").write_text(json.dumps(meta, indent=2))
    p("wrote", OUT / "encoder_meta.json")
    return meta


def upload():
    if not UPLOAD:
        return
    from huggingface_hub import HfApi  # noqa: PLC0415
    tok = hf_token()
    if not tok:
        raise RuntimeError("FLIP_UPLOAD=1 but no HF token")
    api = HfApi(token=tok)
    api.create_repo(PUBLISH_REPO, private=False, exist_ok=True)
    for f in sorted(ONNX.glob("*.onnx")) + [OUT / "encoder_meta.json"]:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=PUBLISH_REPO)
        p("uploaded", f.name)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def build_teacher_pipeline(T, device):
    files = {
        "ckpt": CKPT / "diffusion_models/triposplat_fp16.safetensors",
        "dec": CKPT / "vae/triposplat_vae_decoder_fp16.safetensors",
    }
    return T.TripoSplatPipeline(
        ckpt_path=str(files["ckpt"]), decoder_path=str(files["dec"]),
        dinov3_path=str(CKPT / "clip_vision/dino_v3_vit_h.safetensors"),
        flux2_vae_encoder_path=str(CKPT / "vae/flux2-vae.safetensors"),
        rmbg_path=str(CKPT / "background_removal/birefnet.safetensors"),
        device=device)


def ply_stats(gauss):
    import numpy as np  # noqa: PLC0415
    op = gauss.get_opacity.detach().float().cpu().numpy().reshape(-1)
    sc = gauss.get_scaling.detach().float().cpu().numpy().reshape(-1)
    xyz = gauss.get_xyz.detach().float().cpu().numpy().reshape(-1, 3)
    return {"opacity": float(op.mean()), "scale": float(sc.mean()),
            "x": [float(xyz[:, 0].min()), float(xyz[:, 0].max())]}


def e2e():
    """Teacher DiT + decoder on student conditions vs teacher conditions.

    This is the acceptance test that matters: a feature cosine of 0.98 means
    nothing if the contact sheet moves. We run the teacher's own sampler for
    two steps (the student's grid) and compare the decoded Gaussian statistics.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from torchvision import transforms  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415
    pipe = build_teacher_pipeline(T, device)
    dino, vae = pipe.dinov3, pipe.vae_encoder
    dino_s = load_weights(build_dino(), "dino", device)
    vae_s = load_weights(build_vae(), "vae", device)

    stems = cache_items()[:4]
    rows = []
    for stem in stems:
        canvas = Image.fromarray(
            np.load(CACHE / f"canvas_{stem}.npy").transpose(1, 2, 0))
        x = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
        dino_dtype = next(dino.parameters()).dtype
        vae_dtype = next(vae.parameters()).dtype
        dino_device = next(dino.parameters()).device
        vae_device = next(vae.parameters()).device
        with torch.no_grad():
            tf1 = F.layer_norm(dino(
                pixel_values=T._DINOV3_NORMALIZE(x).to(dino_device, dino_dtype)
            ).float(), (1280,))
            tf2 = vae.encode(x.to(vae_device, vae_dtype) * 2 - 1,
                             deterministic=True)
            sf1 = dino_s(x)
            sf2 = vae_s(x * 2 - 1)
        cos1 = float(F.cosine_similarity(sf1, tf1, dim=-1).mean())
        cos2 = float(F.cosine_similarity(sf2, tf2, dim=-1).mean())

        def pad2(f):
            z = torch.zeros(f.shape[0], 5, f.shape[2], dtype=f.dtype,
                            device=f.device)
            return torch.cat([z, f], dim=1)

        tf2p, sf2p = pad2(tf2), pad2(sf2)
        out = {}
        lats = {}
        for tag, f1, f2 in (("teacher", tf1, tf2p),
                            ("dino", sf1, tf2p),
                            ("full", sf1, sf2p)):
            cond = {"feature1": f1, "feature2": f2}
            gen = torch.Generator(device=device).manual_seed(42)
            lat = pipe.sample_latent(cond, steps=2, guidance_scale=1.0,
                                     shift=1.0, generator=gen)
            lats[tag] = lat["latent"].detach().float()
            g = pipe.decode_latent(lat["latent"], num_gaussians=65536)
            out[tag] = ply_stats(g)
        lat_cos = float(F.cosine_similarity(
            lats["full"], lats["teacher"], dim=-1).mean())
        lat_cos_d = float(F.cosine_similarity(
            lats["dino"], lats["teacher"], dim=-1).mean())
        rows.append({"stem": stem, "cos1": cos1, "cos2": cos2,
                     "lat_cos_full": lat_cos, "lat_cos_dino": lat_cos_d, **out})
        p(f"e2e {stem} cos1 {cos1:.4f} cos2 {cos2:.4f} "
          f"lat {lat_cos:.4f}/{lat_cos_d:.4f} "
          f"opacity teacher/dino/full "
          f"{out['teacher']['opacity']:.4f}/"
          f"{out['dino']['opacity']:.4f}/{out['full']['opacity']:.4f}")

    (OUT / "e2e.json").write_text(json.dumps(rows, indent=2))
    p("wrote", OUT / "e2e.json")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()
    ensure_dirs()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    p("stages", stages)
    try:
        if "setup" in stages:
            setup()
        if "prep" in stages:
            prep()
        if "train" in stages:
            train()
        if "export" in stages:
            export()
        if "upload" in stages:
            upload()
        if "e2e" in stages:
            e2e()
    except Exception:  # noqa: BLE001
        p("FAILED")
        p(traceback.format_exc()[-4000:])
        raise
    finally:
        (OUT / "log.txt").write_text("\n".join(LOG))
        p("log ->", OUT / "log.txt")


if __name__ == "__main__":
    main()
