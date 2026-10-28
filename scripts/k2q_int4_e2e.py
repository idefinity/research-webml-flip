"""K2q - run the published int4 student through the real pipeline, end to end.

Everything so far judged the 4-bit student inside the DiT: a matmul here, a
cosine there. The graph on the hub has never produced a single Gaussian. That
is the gap this kernel closes: take model.safetensors off the hub, dequantise it
back into the flow model, run the stock pipeline over the same three fuliji
images k2n used, and write PLYs.

What makes it a check and not a demo: the fp32 student already ran on these
images (k2n, assets/k2n/k2g_report.json). Same seed, same two steps, same
decoder, only the weights differ. The numbers to match are recorded there -
opacity_mean 0.4978, scale_mean 0.00571, and the latent cosine against the
teacher's 20 steps. If int4 moves those, the quantisation is not free and the
release note is wrong.

It also writes the fixtures the browser needs: the conditioning, the initial
noise, and the latent after each step, so the ONNX graph in the page can be
compared against this run number for number instead of against a picture.

Env:
  FLIP_N      images (default 3)
  FLIP_SEED   sampling seed (default 42, the one k2n used)
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
OUT = W / "k2q"
OUT.mkdir(parents=True, exist_ok=True)
HF_TOKEN = "__TOKEN__"
TARGET = "idefinity/flip-web"
WEIGHTS_REPO = "idefinity/flip-student-2step"
SHA = "d8db9e0"

N_IMG = int(os.environ.get("FLIP_N", "3"))
SEED = int(os.environ.get("FLIP_SEED", "42"))
N_GAUSS = 65536
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = []
R = {"notes": [], "images": [], "model": {}}


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


def stats_of(name, a, b=None):
    """Two arrays: cosine and relative L2, the pair used everywhere else here."""
    x = a.reshape(-1).astype(np.float64)
    out = {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std())}
    if b is not None:
        y = b.reshape(-1).astype(np.float64)
        out["cos"] = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        out["rel_l2"] = float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12))
    return out


def dequantise(path):
    """model.safetensors -> a state dict the flow model will accept.

    int4 weights are U8 [n, k/2] with a F16 [n, k/32] scale beside them; the
    value is (nibble - 8) * scale. Everything else stayed fp16 and only needs
    widening to fp32, which load_state_dict does on copy anyway.
    """
    from safetensors import safe_open

    sd = {}
    n_int4 = n_fp = 0
    params = 0
    with safe_open(path, framework="numpy") as f:
        keys = list(f.keys())
        for k in keys:
            if not k.endswith(".weight"):
                continue
            w = f.get_tensor(k)
            if w.dtype != np.uint8:
                continue
            sname = k + "_scale"
            if sname not in keys:
                raise RuntimeError(f"{k} is U8 but {sname} is missing")
            sc = f.get_tensor(sname).astype(np.float32)
            n, half = w.shape
            k_dim = half * 2
            blocks = k_dim // 32
            packed = w.reshape(n, blocks, 16)
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            u = np.stack([lo, hi], axis=-1).reshape(n, blocks, 32)
            q = u.astype(np.int16) - 8
            wq = (q.astype(np.float32) * sc[:, :, None]).reshape(n, k_dim)
            sd[k] = torch.from_numpy(wq.copy())
            n_int4 += 1
            params += n * k_dim
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            if k.endswith("_scale"):
                continue
            t = f.get_tensor(k)
            if t.dtype == np.uint8:
                continue
            sd[k] = torch.from_numpy(np.ascontiguousarray(t.astype(np.float32)))
            n_fp += 1
            params += t.size
    p(f"dequantised {n_int4} int4 weights + {n_fp} fp16 tensors = {params/1e6:.1f} M params")
    return sd


# ---------------------------------------------------------------------------
# the noise the sampler starts from. The browser has to start from the same one
# or its two steps are not the two steps measured here, so torch.randn is
# watched for the first tensor shaped like a latent and copied out.
# ---------------------------------------------------------------------------
NOISE = {}
_orig_randn = torch.randn


def _spy_randn(*a, **kw):
    t = _orig_randn(*a, **kw)
    try:
        if not NOISE and t.shape[-1] == 16 and t.numel() >= 8192 * 16:
            NOISE["x"] = t.detach().float().cpu().numpy().copy()
            p(f"  captured initial noise {tuple(t.shape)}")
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
    p("ckpts", sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file()) / 1e9, "GB")

    # ---- 1. the published weights, from the hub, not from a training run
    wp = hf_hub_download(repo_id=WEIGHTS_REPO, filename="model.safetensors",
                         local_dir=str(OUT / "hub"))
    p("model.safetensors", round(os.path.getsize(wp) / 1e6, 2), "MB")
    sd = dequantise(wp)
    R["model"]["params_M"] = round(sum(v.numel() for v in sd.values()) / 1e6, 1)
    R["model"]["tensors"] = len(sd)

    # ---- 2. the pipeline
    sys.path.insert(0, str(REPO))
    import triposplat as T

    files = sorted(ckpt.rglob("*.safetensors"))

    def pick(*keys):
        for f in files:
            n = f.name.lower()
            if all(k in n for k in keys):
                return f
        return None

    dino_p, vae_p = pick("dino"), pick("flux")
    dec_p = pick("decoder") or pick("octree")
    rmbg_p = pick("biref") or pick("rmbg")
    taken = {q for q in (dino_p, vae_p, dec_p, rmbg_p) if q}
    rest = [f for f in files if f not in taken]
    if not rest:
        raise RuntimeError(f"no main ckpt among {[f.name for f in files]}")
    p("ckpt parts:", {k: v.name for k, v in
                      (("main", rest[0]), ("dino", dino_p), ("flux", vae_p),
                       ("decoder", dec_p), ("rmbg", rmbg_p)) if v})

    pipe = T.TripoSplatPipeline(
        ckpt_path=str(rest[0]), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))
    teacher_sd = {k: v.detach().clone() for k, v in pipe.flow_model.state_dict().items()}

    # the one thing that decides whether this run says anything about int4
    missing, unexpected = pipe.flow_model.load_state_dict(sd, strict=False)
    p("load_state_dict  missing", len(missing), "unexpected", len(unexpected))
    R["model"]["missing"] = missing[:20]
    R["model"]["unexpected"] = unexpected[:20]
    if missing or unexpected:
        raise RuntimeError("the dequantised dict is not the flow model")

    # ---- 3. the same three images k2n ran
    import urllib.request

    from PIL import Image

    urls = []
    for off in range(0, N_IMG):
        q = ("https://datasets-server.huggingface.co/rows?dataset=DownFlow%2Ffuliji"
             f"&config=default&split=train&offset={off}&length=1")
        with urllib.request.urlopen(q, timeout=120) as r:
            row = json.loads(r.read().decode())["rows"][0]["row"]
        img_url = None
        for v in row.values():
            if isinstance(v, dict) and v.get("src"):
                img_url = v["src"]
                break
            if isinstance(v, list) and v and isinstance(v[0], dict) and v[0].get("src"):
                img_url = v[0]["src"]
                break
        if not img_url:
            raise RuntimeError(f"row {off} has no image: {list(row)}")
        urls.append(img_url)
    p("images", urls)

    imgs = []
    for i, u in enumerate(urls):
        name = f"hf_{i:04d}.jpg"
        with urllib.request.urlopen(urllib.request.Request(
                u, headers={"User-Agent": "flip-k2q"}), timeout=180) as r:
            (OUT / name).write_bytes(r.read())
        imgs.append((name, Image.open(OUT / name).convert("RGB")))
        p("  ", name, imgs[-1][1].size)

    # ---- 4. run
    for name, image in imgs:
        rec = {"image": name}
        p("-" * 60)
        p(name)
        prepared = pipe.preprocess_image(image)
        prepared.save(OUT / f"prepared_{Path(name).stem}.png")

        t0 = time.time()
        cond = pipe.encode_image(prepared)
        rec["encode_s"] = round(time.time() - t0, 2)
        rec["cond_keys"] = sorted(cond.keys()) if isinstance(cond, dict) else str(type(cond))

        # fixtures: what the browser must feed the ONNX graph
        if isinstance(cond, dict):
            shapes = {}
            for k, v in cond.items():
                if torch.is_tensor(v):
                    a = v.detach().float().cpu().numpy()
                    np.save(OUT / f"{Path(name).stem}_{k}.npy", a)
                    shapes[k] = list(a.shape)
            rec["cond_shapes"] = shapes

        lat_by_tag = {}

        def run_cfg(tag, steps, guidance, shift, weights):
            pipe.flow_model.load_state_dict(weights)
            NOISE.clear()
            torch.randn = _spy_randn
            try:
                gen = torch.Generator(device=str(DEV)).manual_seed(SEED)
                torch.cuda.synchronize() if DEV.type == "cuda" else None
                t0 = time.time()
                lat = pipe.sample_latent(cond, steps=steps, guidance_scale=guidance,
                                         shift=shift, generator=gen)
                torch.cuda.synchronize() if DEV.type == "cuda" else None
                dt = round(time.time() - t0, 2)
            finally:
                torch.randn = _orig_randn
            if "x" in NOISE:
                np.save(OUT / f"{Path(name).stem}_{tag}_noise.npy", NOISE["x"])
            lat_by_tag[tag] = {k: v.detach().float().cpu().numpy()
                               for k, v in lat.items() if torch.is_tensor(v)}
            for k, v in lat_by_tag[tag].items():
                np.save(OUT / f"{Path(name).stem}_{tag}_{k}.npy", v)
            return {"sample_s": dt,
                    "latent_shapes": {k: list(v.shape) for k, v in lat_by_tag[tag].items()}}

        run20 = run_cfg("t20", 20, 3.0, 3.0, teacher_sd)
        run01 = run_cfg("i01", 1, 1.0, 1.0, sd)      # the int4 student, first step
        run02 = run_cfg("i02", 2, 1.0, 1.0, sd)      # and its second
        rec["runs"] = {"t20": run20, "i01": run01, "i02": run02}

        pairs = {}
        for tag in ("i01", "i02"):
            a = lat_by_tag[tag].get("latent")
            b = lat_by_tag["t20"].get("latent")
            if a is not None and b is not None:
                pairs[tag] = stats_of(tag, a, b)
        # i01 vs i02 is the only comparison that is purely the second step
        if "latent" in lat_by_tag["i01"]:
            pairs["i01_vs_i02"] = stats_of("i01", lat_by_tag["i01"]["latent"],
                                           lat_by_tag["i02"]["latent"])
        rec["latent"] = pairs

        t0 = time.time()
        g = pipe.decode_latent(lat_by_tag["i02"]["latent"].__class__ and
                               torch.from_numpy(lat_by_tag["i02"]["latent"]).to(DEV),
                               num_gaussians=N_GAUSS)
        torch.cuda.synchronize() if DEV.type == "cuda" else None
        rec["decode_s"] = round(time.time() - t0, 2)
        rec["n_gaussians"] = int(g.get_xyz.shape[0])
        rec["opacity_mean"] = round(float(g.get_opacity.detach().float().mean()), 4)
        rec["scale_mean"] = round(float(g.get_scaling.detach().float().mean()), 5)
        ply = OUT / f"{Path(name).stem}_i02.ply"
        g.save_ply(str(ply))
        rec["ply"] = ply.name
        rec["ply_mb"] = round(ply.stat().st_size / 1e6, 2)
        p(f"  int4 2-step: sample {run02['sample_s']}s decode {rec['decode_s']}s "
          f"opacity {rec['opacity_mean']} scale {rec['scale_mean']}")
        p(f"  latent vs t20: cos {pairs.get('i02', {}).get('cos')} "
          f"rel_l2 {pairs.get('i02', {}).get('rel_l2')}")
        R["images"].append(rec)

    # ---- 5. publish
    api = HfApi(token=HF_TOKEN)
    api.create_repo(TARGET, private=False, exist_ok=True, repo_type="model")
    for f in sorted(OUT.glob("*")):
        if f.suffix in (".npy", ".ply", ".json", ".jpg", ".png"):
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"k2q/{f.name}",
                            repo_id=TARGET, repo_type="model")
            p("uploaded", f.name, round(f.stat().st_size / 1e6, 2), "MB")
    p("done -> https://huggingface.co/" + TARGET + "/tree/main/k2q")
except Exception:  # noqa: BLE001
    p("FAILED")
    p(traceback.format_exc()[:3000])

(OUT / "log.txt").write_text("\n".join(log))
print("end", flush=True)
