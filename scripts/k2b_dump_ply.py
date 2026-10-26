"""K2b - settle the 4-step question with a real rasteriser instead of my own.

K2a concluded "4 steps is fine" from a renderer that is orthographic, painter's
algorithm, isotropic discs, 256 px. It can answer "is the geometry there and is
the colour roughly right". It cannot answer "is this too blurry to ship", which
is exactly the question an outside user answered differently: they found 20
steps visibly acceptable and 4 steps too low quality.

So stop arguing with a proxy. Dump the Gaussians at each step count as a PLY
using the repo's own save_ply, then look at them in Spark / SuperSplat, which
are the rasterisers people actually judge this stuff in.

Writes k2b/g{steps}.ply plus a small report. The PLYs are the deliverable; the
report only records what produced them.
"""

import json
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
OUT = WORK / "k2b"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = [20, 8, 4]
SEED = 42
N_GAUSS = 65536
R = {"runs": {}, "notes": []}


def note(msg=""):
    print(msg, flush=True)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=2))


def main():
    note(f"torch {torch.__version__}  device "
         f"{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}")
    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415

    files = sorted(CKPT.rglob("*.safetensors"))

    def pick(*keys):
        for f in files:
            n = f.name.lower()
            if all(k in n for k in keys):
                return f
        return None

    # explicit keys, because "triposplat_vae_decoder_fp16" also matches "vae"
    dino_p = pick("dino")
    vae_p = pick("flux")
    dec_p = pick("decoder") or pick("octree")
    rmbg_p = pick("biref") or pick("rmbg")
    taken = {p for p in (dino_p, vae_p, rmbg_p, dec_p) if p}
    ck = [f for f in files if f not in taken][0]
    missing = [k for k, v in (("dino", dino_p), ("flux", vae_p),
                              ("decoder", dec_p), ("rmbg", rmbg_p)) if v is None]
    if missing:
        raise RuntimeError(f"missing ckpt parts {missing} in {[f.name for f in files]}")

    note(f"ckpt {ck.name}")
    note(f"decoder {dec_p.name if dec_p else None}")
    note(f"dino {dino_p.name if dino_p else None}")
    note(f"vae {vae_p.name if vae_p else None}")
    note(f"rmbg {rmbg_p.name if rmbg_p else None}")
    R["ckpt"] = {"ck": str(ck), "decoder": str(dec_p), "dino": str(dino_p),
                 "vae": str(vae_p), "rmbg": str(rmbg_p)}

    pipe = T.TripoSplatPipeline(
        ckpt_path=str(ck), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))

    ex = REPO / "static" / "example_inputs"
    cands = sorted(p for p in ex.glob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not cands:
        cands = [p for p in sorted((REPO / "static").rglob("*"))
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    img_path = cands[0]
    note(f"using {img_path.name}")
    R["image"] = img_path.name
    image = Image.open(img_path).convert("RGB")

    t0 = time.time()
    prepared = pipe.preprocess_image(image)
    R["preprocess_s"] = round(time.time() - t0, 2)
    note(f"preprocess {R['preprocess_s']}s  size {prepared.size}")
    prepared.save(OUT / "prepared.png")
    image.save(OUT / "input.png")

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

            run["n_gaussians"] = int(g.get_xyz.shape[0])
            run["opacity_mean"] = round(
                g.get_opacity.detach().float().mean().item(), 4)
            run["scale_mean"] = round(
                g.get_scaling.detach().float().mean().item(), 5)
            run["sh_degree"] = int(getattr(g, "max_sh_degree", -1))

            # the repo's own writer - no hand-rolled PLY header
            ply = OUT / f"g{steps:02d}.ply"
            g.save_ply(str(ply))
            run["ply"] = ply.name
            run["ply_bytes"] = ply.stat().st_size
            note(f"  sample {run['sample_s']}s  decode {run['decode_s']}s")
            note(f"  {ply.name}  {run['ply_bytes'] / 1e6:.1f} MB  "
                 f"n={run['n_gaussians']}  opacity {run['opacity_mean']}  "
                 f"scale {run['scale_mean']}  sh {run['sh_degree']}")
        except Exception:  # noqa: BLE001
            run["error"] = traceback.format_exc()[-1500:]
            note("  FAILED")
            note(traceback.format_exc()[-1500:])
        R["runs"][str(steps)] = run
        save()
        torch.cuda.empty_cache()

    note("=" * 72)
    note("SUMMARY")
    for steps in STEPS:
        r = R["runs"].get(str(steps), {})
        note(f"  {steps:2d} steps  {r.get('sample_s')}s  "
             f"{r.get('ply')}  {r.get('ply_bytes', 0) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
