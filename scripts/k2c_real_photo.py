"""K2c - run the real pipeline on a real photo the user picked.

Everything so far used `static/example_inputs/building_stone_house.webp`, a
miniature diorama that sits squarely inside the object-level training
distribution. That proves the pipeline runs. It does not prove anything about
the actual product input, which is a person.

So this takes an arbitrary image URL, runs preprocess (RMBG cutout) -> encode ->
sample -> decode at several step counts, and dumps both the prepared cutout and
the Gaussians as PLY. The PLYs get rendered locally in Spark at eight azimuths,
which is where the "does it hallucinate a back" question actually gets answered.

The prepared cutout matters as much as the PLY: if the subject is gone, clipped,
or half-fused with the background before the model sees it, no amount of
sampling fixes that, and it is the cheapest thing to look at first.

Env:
  FLIP_IMAGE_URL   image to fetch (required unless FLIP_IMAGE_B64 is set)
  FLIP_IMAGE_B64   base64 of the image, used when the URL is unreachable
  FLIP_STEPS       comma separated step counts, default "20,8,4"
  FLIP_TAG         output name prefix, default "photo"
"""

import base64
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch
from PIL import Image

warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k2c"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
URL = os.environ.get("FLIP_IMAGE_URL", "").strip()
B64 = os.environ.get("FLIP_IMAGE_B64", "").strip()
TAG = os.environ.get("FLIP_TAG", "photo").strip() or "photo"
STEPS = [int(s) for s in os.environ.get("FLIP_STEPS", "20,8,4").split(",") if s.strip()]
SEED = 42
N_GAUSS = 65536
R = {"runs": {}, "url": URL, "tag": TAG}


def note(msg=""):
    print(msg, flush=True)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=2))


UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")


def fetch_image():
    """URL first, base64 second. Returns (PIL image, provenance dict)."""
    prov = {}
    if URL:
        import urllib.request

        for tag, hdrs in [
            ("plain", {"User-Agent": UA}),
            ("referer", {"User-Agent": UA, "Referer": "https://www.dfyun.com.cn/",
                         "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"}),
        ]:
            try:
                req = urllib.request.Request(URL, headers=hdrs)
                with urllib.request.urlopen(req, timeout=60) as fh:
                    raw = fh.read()
                    code = fh.status
                prov[f"url_{tag}"] = {"status": code, "bytes": len(raw)}
                note(f"fetch {tag}: HTTP {code}, {len(raw)} B")
                if len(raw) > 1024:
                    (OUT / "raw.bin").write_bytes(raw)
                    prov["source"] = "url"
                    return Image.open(OUT / "raw.bin").convert("RGB"), prov
            except Exception as exc:  # noqa: BLE001
                prov[f"url_{tag}"] = {"error": f"{type(exc).__name__}: {exc}"}
                note(f"fetch {tag} FAILED: {type(exc).__name__}: {exc}")

    if B64:
        raw = base64.b64decode(B64)
        (OUT / "raw.bin").write_bytes(raw)
        prov["source"] = "base64"
        prov["bytes"] = len(raw)
        note(f"using embedded base64: {len(raw)} B")
        return Image.open(OUT / "raw.bin").convert("RGB"), prov

    raise RuntimeError("no image: URL unreachable and no base64 fallback given")


def main():
    note(f"torch {torch.__version__}  device "
         f"{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}")
    image, prov = fetch_image()
    R["fetch"] = prov
    R["input_size"] = list(image.size)
    note(f"image {image.size[0]}x{image.size[1]}  mode {image.mode}")
    image.save(OUT / "input.png")

    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415

    files = sorted(CKPT.rglob("*.safetensors"))

    def pick(*keys):
        for f in files:
            n = f.name.lower()
            if all(k in n for k in keys):
                return f
        return None

    dino_p = pick("dino")
    vae_p = pick("flux")
    dec_p = pick("decoder") or pick("octree")
    rmbg_p = pick("biref") or pick("rmbg")
    taken = {p for p in (dino_p, vae_p, rmbg_p, dec_p) if p}
    ck = [f for f in files if f not in taken][0]
    missing = [k for k, v in (("dino", dino_p), ("flux", vae_p),
                              ("decoder", dec_p), ("rmbg", rmbg_p)) if v is None]
    if missing:
        raise RuntimeError(f"missing ckpt parts {missing}: {[f.name for f in files]}")

    pipe = T.TripoSplatPipeline(
        ckpt_path=str(ck), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))

    t0 = time.time()
    prepared = pipe.preprocess_image(image)
    R["preprocess_s"] = round(time.time() - t0, 2)
    R["prepared_size"] = list(prepared.size)
    prepared.save(OUT / "prepared.png")
    note(f"preprocess {R['preprocess_s']}s  prepared {prepared.size}")

    # how much of the 1024 box the cutout actually covers - a subject that got
    # clipped or that RMBG mistook for background shows up here as a tiny or
    # lopsided alpha area before it shows up in the Gaussians
    try:
        import numpy as np  # noqa: PLC0415
        a = np.asarray(prepared.convert("RGBA"))[..., 3].astype("float32") / 255.0
        ys, xs = np.nonzero(a > 0.5)
        if len(xs):
            R["cutout"] = {
                "coverage": round(float((a > 0.5).mean()), 4),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "bbox_frac": [round(float((xs.max() - xs.min()) / a.shape[1]), 3),
                              round(float((ys.max() - ys.min()) / a.shape[0]), 3)],
            }
            note(f"cutout coverage {R['cutout']['coverage']}  bbox {R['cutout']['bbox_xyxy']}")
    except Exception as exc:  # noqa: BLE001
        note(f"cutout stats skipped: {exc}")
    save()

    gen = torch.Generator(device=DEV).manual_seed(SEED)
    t0 = time.time()
    cond = pipe.encode_image(prepared, generator=gen)
    R["encode_s"] = round(time.time() - t0, 2)
    note(f"encode {R['encode_s']}s")

    for steps in STEPS:
        note("-" * 64)
        note(f"steps={steps}")
        run = {"steps": steps}
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            gen_i = torch.Generator(device=DEV).manual_seed(SEED)
            lat = pipe.sample_latent(cond, steps=steps, guidance_scale=3.0,
                                     shift=3.0, generator=gen_i)
            torch.cuda.synchronize()
            run["sample_s"] = round(time.time() - t0, 2)

            t0 = time.time()
            g = pipe.decode_latent(lat["latent"], num_gaussians=N_GAUSS)
            torch.cuda.synchronize()
            run["decode_s"] = round(time.time() - t0, 2)

            xyz = g.get_xyz.detach().float()
            run["n_gaussians"] = int(xyz.shape[0])
            run["opacity_mean"] = round(g.get_opacity.detach().float().mean().item(), 4)
            run["scale_mean"] = round(g.get_scaling.detach().float().mean().item(), 5)
            run["sh_degree"] = int(getattr(g, "max_sh_degree", -1))
            lo = xyz.min(0).values.tolist()
            hi = xyz.max(0).values.tolist()
            run["bbox_min"] = [round(v, 4) for v in lo]
            run["bbox_max"] = [round(v, 4) for v in hi]
            run["extent"] = [round(h - l, 4) for l, h in zip(lo, hi)]

            ply = OUT / f"{TAG}_{steps:02d}.ply"
            g.save_ply(str(ply))
            run["ply"] = ply.name
            run["ply_bytes"] = ply.stat().st_size
            note(f"  sample {run['sample_s']}s  decode {run['decode_s']}s")
            note(f"  {ply.name} {run['ply_bytes'] / 1e6:.1f} MB  n={run['n_gaussians']} "
                 f"opacity {run['opacity_mean']} scale {run['scale_mean']}")
            note(f"  extent {run['extent']}")
        except Exception:  # noqa: BLE001
            run["error"] = traceback.format_exc()[-1500:]
            note("  FAILED")
            note(traceback.format_exc()[-1500:])
        R["runs"][str(steps)] = run
        save()
        torch.cuda.empty_cache()

    note("=" * 72)
    for steps in STEPS:
        r = R["runs"].get(str(steps), {})
        note(f"  {steps:2d} steps  {r.get('sample_s')}s  {r.get('ply')}  "
             f"{r.get('ply_bytes', 0) / 1e6:.1f} MB  opacity {r.get('opacity_mean')}")


if __name__ == "__main__":
    main()
