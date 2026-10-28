"""K2u - the fixtures the browser needs, and one number it cannot get anywhere else.

k2q ran the int4 student end to end and wrote PLYs, but left two things open:

  1. The browser cannot reproduce a trajectory without the noise it started
     from. The sampler draws two tensors - latent [1,8192,16] and camera
     [1,1,5] - from one generator, and k2q only captured the first. Without the
     second, the page's two steps are two different steps and the comparison
     says nothing. Both are written out here, along with the conditioning and
     the latent after each step.

  2. k2q's PLY statistics moved against k2n's fp32 student (opacity 0.4228 vs
     0.4978). That is either the quantisation or the decoder's own randomness -
     the octree sampler draws uniforms and adds a random jitter inside each
     voxel. Decoding one latent several times settles it: if the spread across
     runs is the size of the gap, the gap is the decoder, not int4.

Also dumped: the fixtures as .npy, small enough for a page to fetch.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

W = Path("/kaggle/working")
REPO = W / "TripoSplat"
OUT = W / "k2u"
OUT.mkdir(parents=True, exist_ok=True)
HF_TOKEN = "__TOKEN__"
TARGET = "idefinity/flip-web"
WEIGHTS_REPO = "idefinity/flip-student-2step"
SHA = "d8db9e0"
SEED = 42
N_GAUSS = 65536

log = []
R = {"notes": [], "decode_spread": [], "fixtures": {}}


def p(*a):
    s = " ".join(str(x) for x in a)
    log.append(s)
    R["notes"].append(s)
    print(s, flush=True)
    try:
        (OUT / "report.json").write_text(json.dumps(R, indent=2, default=float))
    except Exception:  # noqa: BLE001
        pass


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    p("$", " ".join(cmd)[:140], "-> rc", r.returncode)
    if r.returncode != 0:
        p("   ", (r.stderr or r.stdout or "")[-400:].replace("\n", " "))
    return r


DRAWS = []
_orig_randn = torch.randn


def _spy_randn(*a, **kw):
    t = _orig_randn(*a, **kw)
    try:
        if len(DRAWS) < 4:
            DRAWS.append(t.detach().float().cpu().numpy().copy())
            p(f"  randn #{len(DRAWS)} {tuple(t.shape)}")
    except Exception:  # noqa: BLE001
        pass
    return t


try:
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
         "huggingface_hub", "safetensors", "torchvision", "tqdm"], timeout=900)
    if not (REPO / ".git").exists():
        run(["git", "clone", "-q", "https://github.com/VAST-AI-Research/TripoSplat.git",
             str(REPO)], timeout=600)
    run(["git", "-C", str(REPO), "checkout", "-q", SHA], timeout=300)

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    ckpt = REPO / "ckpts"
    if not (ckpt / ".complete").exists():
        snapshot_download(repo_id="VAST-AI/TripoSplat", local_dir=str(ckpt))
        (ckpt / ".complete").touch()

    sys.path.insert(0, str(REPO))
    import triposplat as T

    # ---- int4 weights, the same way k2q did
    from safetensors import safe_open

    wp = hf_hub_download(repo_id=WEIGHTS_REPO, filename="model.safetensors",
                         local_dir=str(OUT / "hub"))
    sd = {}
    with safe_open(wp, framework="numpy") as f:
        keys = list(f.keys())
        for k in keys:
            if not k.endswith(".weight"):
                continue
            w = f.get_tensor(k)
            if w.dtype != np.uint8:
                continue
            sc = f.get_tensor(k + "_scale").astype(np.float32)
            n, half = w.shape
            kd = half * 2
            blocks = kd // 32
            packed = w.reshape(n, blocks, 16)
            u = np.stack([packed & 0x0F, (packed >> 4) & 0x0F], axis=-1).reshape(n, blocks, 32)
            wq = ((u.astype(np.int16) - 8).astype(np.float32) * sc[:, :, None]).reshape(n, kd)
            sd[k] = torch.from_numpy(wq.copy())
    with safe_open(wp, framework="numpy") as f:
        for k in f.keys():
            if k.endswith("_scale"):
                continue
            t = f.get_tensor(k)
            if t.dtype == np.uint8:
                continue
            sd[k] = torch.from_numpy(np.ascontiguousarray(t.astype(np.float32)))
    p("dequantised", len(sd), "tensors")

    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(ckpt.rglob("*.safetensors"))

    def pick(*ks):
        for f in files:
            if all(k in f.name.lower() for k in ks):
                return f
        return None

    dino_p, vae_p = pick("dino"), pick("flux")
    dec_p = pick("decoder") or pick("octree")
    rmbg_p = pick("biref") or pick("rmbg")
    taken = {q for q in (dino_p, vae_p, dec_p, rmbg_p) if q}
    rest = [f for f in files if f not in taken][0]
    pipe = T.TripoSplatPipeline(ckpt_path=str(rest), decoder_path=str(dec_p),
                                dinov3_path=str(dino_p), flux2_vae_encoder_path=str(vae_p),
                                rmbg_path=str(rmbg_p), device=str(DEV))
    p("load_state_dict", pipe.flow_model.load_state_dict(sd, strict=False))

    # ---- the image, the same first fuliji row k2q and k2n used
    import urllib.request

    from PIL import Image

    q = ("https://datasets-server.huggingface.co/rows?dataset=DownFlow%2Ffuliji"
         "&config=default&split=train&offset=0&length=1")
    with urllib.request.urlopen(q, timeout=120) as r:
        row = json.loads(r.read().decode())["rows"][0]["row"]
    url = None
    for v in row.values():
        if isinstance(v, dict) and v.get("src"):
            url = v["src"]
            break
    with urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "flip-k2u"}), timeout=180) as r:
        (OUT / "hf_0000.jpg").write_bytes(r.read())
    image = Image.open(OUT / "hf_0000.jpg").convert("RGB")
    p("image", image.size)

    prepared = pipe.preprocess_image(image)
    prepared.save(OUT / "prepared.png")
    cond = pipe.encode_image(prepared)
    for k, v in cond.items():
        if torch.is_tensor(v):
            a = v.detach().float().cpu().numpy()
            np.save(OUT / f"cond_{k}.npy", a)
            R["fixtures"][f"cond_{k}"] = list(a.shape)
    p("cond", json.dumps(R["fixtures"]))

    # ---- the noise, both of them
    DRAWS.clear()
    torch.randn = _spy_randn
    try:
        gen = torch.Generator(device=str(DEV)).manual_seed(SEED)
        torch.cuda.synchronize() if DEV.type == "cuda" else None
        lat2 = pipe.sample_latent(cond, steps=2, guidance_scale=1.0, shift=1.0, generator=gen)
    finally:
        torch.randn = _orig_randn

    for i, d in enumerate(DRAWS):
        np.save(OUT / f"noise_{i}.npy", d)
        p(f"noise_{i}.npy", d.shape, d.dtype)
    R["noise_shapes"] = [list(d.shape) for d in DRAWS]
    for k, v in lat2.items():
        if torch.is_tensor(v):
            np.save(OUT / f"latent2_{k}.npy", v.detach().float().cpu().numpy())

    # one step, from the same seed, so the page can check each step separately
    DRAWS.clear()
    torch.randn = _spy_randn
    try:
        gen = torch.Generator(device=str(DEV)).manual_seed(SEED)
        lat1 = pipe.sample_latent(cond, steps=1, guidance_scale=1.0, shift=1.0, generator=gen)
    finally:
        torch.randn = _orig_randn
    for k, v in lat1.items():
        if torch.is_tensor(v):
            np.save(OUT / f"latent1_{k}.npy", v.detach().float().cpu().numpy())
    p("latent1 vs latent2 cos",
      float((lat1["latent"].reshape(-1).float() @ lat2["latent"].reshape(-1).float())
            / (lat1["latent"].norm().float() * lat2["latent"].norm().float())))

    # ---- how much of the PLY difference is the decoder's own randomness
    latent = lat2["latent"]
    for i in range(4):
        t0 = time.time()
        g = pipe.decode_latent(latent, num_gaussians=N_GAUSS)
        torch.cuda.synchronize() if DEV.type == "cuda" else None
        rec = {
            "run": i,
            "decode_s": round(time.time() - t0, 2),
            "opacity_mean": round(float(g.get_opacity.detach().float().mean()), 4),
            "scale_mean": round(float(g.get_scaling.detach().float().mean()), 5),
        }
        R["decode_spread"].append(rec)
        p("decode", json.dumps(rec))
        if i == 0:
            g.save_ply(str(OUT / "hf_0000_int4_i02.ply"))
    ops = [r["opacity_mean"] for r in R["decode_spread"]]
    scs = [r["scale_mean"] for r in R["decode_spread"]]
    R["spread"] = {
        "opacity_min": min(ops), "opacity_max": max(ops),
        "scale_min": min(scs), "scale_max": max(scs),
        "note": "k2n fp32 student: opacity 0.4978 scale 0.00571; k2q int4: 0.4228 / 0.00641",
    }
    p("spread", json.dumps(R["spread"]))

    api = HfApi(token=HF_TOKEN)
    api.create_repo(TARGET, private=False, exist_ok=True, repo_type="model")
    for f in sorted(OUT.glob("*")):
        if f.suffix in (".npy", ".json", ".ply", ".jpg", ".png"):
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"k2u/{f.name}",
                            repo_id=TARGET, repo_type="model")
            p("uploaded", f.name, round(f.stat().st_size / 1e6, 2), "MB")
    p("done")
except Exception:  # noqa: BLE001
    p("FAILED")
    p(traceback.format_exc()[:3000])

(OUT / "log.txt").write_text("\n".join(log))
print("end", flush=True)
