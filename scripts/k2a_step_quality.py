"""K2a - before spending hours on distillation, check what fewer steps already do.

PLAN.md has distillation as mandatory: 20 Euler steps is 475 TFLOP and no
browser will run that. But "mandatory" and "8 hours of T4" are different
claims, and nobody has actually looked at what 4 steps produce. Flow models
are often more forgiving than their step count suggests, especially at
guidance 3 with a shifted schedule.

So: same image, same seed, same conditioner, steps in {20, 8, 4, 2}. Measure
time and compare the Gaussian sets. Then render all of them from four
azimuths so the comparison is a picture rather than a number.

The renderer is deliberately crude - orthographic, painter's algorithm,
isotropic discs - because the question is whether the silhouette and the
colour survive, not whether it matches a real 3DGS rasteriser. If 4 steps
still produces a person, distillation is an optimisation. If it produces
mush, distillation is the project.

Writes k2a/report.json and k2a/steps_grid.png.
"""

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k2a"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = [20, 8, 4, 2]
SEED = 42
N_GAUSS = 65536
SH_C0 = 0.28209479177387814
R = {"runs": {}, "notes": []}


def note(msg=""):
    print(msg, flush=True)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=2))


# ---------------------------------------------------------------------------
# a renderer just good enough to judge silhouettes
# ---------------------------------------------------------------------------

def render(g, size=256, azimuth_deg=0.0, elev_deg=0.0, zoom=1.35):
    """Orthographic splat. Painter's algorithm by depth, alpha from opacity."""
    xyz = g.get_xyz.detach().float().cpu().numpy()
    op = g.get_opacity.detach().float().cpu().numpy().reshape(-1)
    sc = g.get_scaling.detach().float().cpu().numpy().mean(axis=1)
    dc = g._features_dc.detach().float().cpu().numpy().reshape(-1, 3)
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)

    a = np.deg2rad(azimuth_deg)
    e = np.deg2rad(elev_deg)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    x = ca * xyz[:, 0] - sa * xyz[:, 2]
    z = sa * xyz[:, 0] + ca * xyz[:, 2]
    y = ce * xyz[:, 1] - se * z
    depth = se * xyz[:, 1] + ce * z

    px = (x * zoom + 0.5) * size
    py = (0.5 - y * zoom) * size
    rad = np.maximum(sc * zoom * size * 0.6, 0.7)

    order = np.argsort(-depth)  # far first
    img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    d = ImageDraw.Draw(img, "RGBA")
    for i in order:
        r = float(rad[i])
        if not (0.0 <= px[i] <= size and 0.0 <= py[i] <= size) and r < 2:
            continue
        col = (int(rgb[i, 0] * 255), int(rgb[i, 1] * 255), int(rgb[i, 2] * 255),
               int(np.clip(op[i], 0.0, 1.0) * 255))
        d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=col)
    return img.convert("RGB")


def silhouette(img, thresh=248):
    a = np.asarray(img.convert("L"))
    return a < thresh


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

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

    dino_p, vae_p = pick("dino"), (pick("vae") or pick("flux"))
    rmbg_p, dec_p = (pick("biref") or pick("rmbg")), (pick("decoder") or pick("octree"))
    taken = {p for p in (dino_p, vae_p, rmbg_p, dec_p) if p}
    ck = [f for f in files if f not in taken][0]

    note("building pipeline ...")
    pipe = T.TripoSplatPipeline(
        ckpt_path=str(ck), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))

    # a subject with a clear silhouette; the repo's own examples are the least
    # likely to be out of distribution
    cands = [p for p in sorted((REPO / "static").rglob("*"))
             if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    note(f"candidate images: {[p.name for p in cands]}")
    img_path = cands[0]
    note(f"using {img_path}")
    image = Image.open(img_path).convert("RGB")
    R["image"] = img_path.name

    t0 = time.time()
    prepared = pipe.preprocess_image(image)
    R["preprocess_s"] = round(time.time() - t0, 2)
    note(f"preprocess {R['preprocess_s']}s  size {prepared.size}")

    gen = torch.Generator(device=DEV).manual_seed(SEED)
    t0 = time.time()
    cond = pipe.encode_image(prepared, generator=gen)
    R["encode_s"] = round(time.time() - t0, 2)
    note(f"encode {R['encode_s']}s")

    renders = {}
    for steps in STEPS:
        note("-" * 60)
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

            xyz = g.get_xyz.detach().float().cpu()
            op = g.get_opacity.detach().float().cpu().reshape(-1)
            sc = g.get_scaling.detach().float().cpu()
            run.update({
                "n_gaussians": int(xyz.shape[0]),
                "xyz_mean": [round(v, 4) for v in xyz.mean(0).tolist()],
                "xyz_std": [round(v, 4) for v in xyz.std(0).tolist()],
                "opacity_mean": round(op.mean().item(), 4),
                "scale_mean": round(sc.mean().item(), 5),
                "time_total_s": round(run["sample_s"] + run["decode_s"], 2),
                "s_per_step": round(run["sample_s"] / steps, 3),
            })
            for az in (0, 90, 180, 270):
                renders[(steps, az)] = render(g, azimuth_deg=az)
            # quantitative shape drift against the 20-step baseline
            if steps != 20:
                base = R["runs"].get("20") or {}
                if base:
                    run["iou_vs_20"] = {
                        az: round(iou(silhouette(renders[(steps, az)]),
                                      silhouette(renders[(20, az)])), 4)
                        for az in (0, 90, 180, 270)}
            R["runs"][str(steps)] = run
            note(f"  sample {run['sample_s']}s ({run['s_per_step']}s/step) "
                 f"decode {run['decode_s']}s")
            note(f"  xyz std {run['xyz_std']} opacity {run['opacity_mean']} "
                 f"scale {run['scale_mean']}")
            if "iou_vs_20" in run:
                note(f"  IoU vs 20 steps: {run['iou_vs_20']}")
        except Exception:  # noqa: BLE001
            run["error"] = traceback.format_exc()[-1500:]
            note("  FAILED")
            note(traceback.format_exc()[-1500:])
        R["runs"][str(steps)] = run
        save()
        torch.cuda.empty_cache()

    # grid: rows are step counts, columns are azimuths
    label_w, cell = 110, 256
    grid = Image.new("RGB", (label_w + cell * 4, cell * len(STEPS)), (250, 250, 250))
    d = ImageDraw.Draw(grid)
    for r_i, steps in enumerate(STEPS):
        d.text((8, r_i * cell + 10), f"{steps} steps", fill=(20, 20, 20))
        run = R["runs"].get(str(steps), {})
        d.text((8, r_i * cell + 28), f"{run.get('sample_s', '?')}s", fill=(90, 90, 90))
        for c_i, az in enumerate((0, 90, 180, 270)):
            im = renders.get((steps, az))
            if im is not None:
                grid.paste(im, (label_w + c_i * cell, r_i * cell))
            d.text((label_w + c_i * cell + 6, r_i * cell + 6), f"{az}deg",
                   fill=(120, 120, 120))
    grid.save(OUT / "steps_grid.png")
    note(f"\nwrote {OUT / 'steps_grid.png'} and {OUT / 'report.json'}")

    note("=" * 72)
    note("SUMMARY")
    note("=" * 72)
    for steps in STEPS:
        r = R["runs"].get(str(steps), {})
        note(f"  {steps:2d} steps  {r.get('sample_s')}s  "
             f"IoU vs 20: {r.get('iou_vs_20')}")


if __name__ == "__main__":
    main()
