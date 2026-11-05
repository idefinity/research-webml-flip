"""K4 - same size, fewer steps: what it buys and what it costs, on real photos.

The DiT is the only module with a step loop. There, "same size, fewer steps" is
exactly what the student already is: the teacher's 370.5 M architecture, 2 calls
instead of 20 (and no CFG pass, so 2 forwards instead of 40). The front end has
no steps - DINOv3, the Flux2 VAE and BiRefNet are single pass - so "same size"
there means running those teachers themselves, and the comparison is a size
ablation, not a step one.

So this script runs six variants per photo, all through the teacher's decoder:

    t20  teacher front end, teacher DiT, 20 steps, guidance 3   quality ceiling
    t08  ... 8 steps                                             truncation curve
    t04  ... 4 steps
    t02  ... 2 steps
    s20  student front end, teacher DiT, 20 steps                front-end cost
    s02  student front end, shipped 2-step DiT, guidance 1       what the browser runs

The 2-step DiT is loaded from the shipped int4 safetensors, dequantized to
float32 and cast into the pipeline's fp16 module: the weights are literally the
ones the browser downloads, minus the 4-bit rounding. Everything else is
straight from the checkpoints.

Per stage wall clock is measured with cuda.synchronize(). Feature cosines,
matte IoU/BCE against BiRefNet on the same resized image, latent/camera cos vs
t20, opacity and scale go into k4/report.json after every image, so a stopped
run still has usable rows.

Run on the Colab/Kaggle box:
    FLIP_WORK=/content python3 k4_compare.py /content/input_example
"""

import copy
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

WORK = Path(os.environ.get("FLIP_WORK", "/content"))
K3 = Path(os.environ.get("FLIP_K3", WORK / "k3_encoder_distill.py"))
REPO = WORK / "TripoSplat"
STUDENT_INT4 = Path(os.environ.get("FLIP_STUDENT_INT4",
                                   WORK / "student/model.safetensors"))
IMGDIR = Path(sys.argv[1] if len(sys.argv) > 1 else WORK / "input_example")
OUT = Path(os.environ.get("FLIP_OUT", WORK / "k4"))
OUT.mkdir(parents=True, exist_ok=True)

N_GAUSS = int(os.environ.get("FLIP_GAUSS", "65536"))
LIMIT = int(os.environ.get("FLIP_N", "0"))
SMOKE = os.environ.get("FLIP_SMOKE", "0") == "1"
TAGS = [t.strip() for t in os.environ.get("FLIP_TAGS", "").split(",") if t.strip()]
# FLIP_SHARD="i:n": this process takes images i, i+n, i+2n... of the sorted list.
# On a T4x2 box the shards are separate processes, each with its own FLIP_OUT, and
# the wall time halves because the matrix is per-image, not batched.
SHARD = os.environ.get("FLIP_SHARD", "")

# (tag, condition, steps, guidance, shift, whose flow-model weights)
# The last slot is separate from the second on purpose. In the first run both
# were keyed off the condition, so `s20` moved the encoders and the int4 weights
# at the same time and could not attribute the gap to either. u* and v* fill in
# the cells that were missing: u20 is the front end alone, u04 is the naive step
# cut under those same conditions, v02 is the shipped int4 weights alone.
VARIANTS = [
    ("t20", "teacher", 20, 3.0, 3.0, "teacher"),
    ("t08", "teacher", 8, 3.0, 3.0, "teacher"),
    ("t04", "teacher", 4, 3.0, 3.0, "teacher"),
    ("t02", "teacher", 2, 3.0, 3.0, "teacher"),
    ("s20", "student", 20, 3.0, 3.0, "student"),
    ("s02", "student", 2, 1.0, 1.0, "student"),
    ("u20", "student", 20, 3.0, 3.0, "teacher"),
    ("u04", "student", 4, 3.0, 3.0, "teacher"),
    ("v02", "teacher", 2, 1.0, 1.0, "student"),
]
if TAGS:
    VARIANTS = [v for v in VARIANTS if v[0] in TAGS]


def p(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Clock:
    def __enter__(self):
        sync()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.dt = time.perf_counter() - self.t


def load_k3():
    spec = importlib.util.spec_from_file_location("k3", str(K3))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pad2(f):
    z = torch.zeros(f.shape[0], 5, f.shape[2], dtype=f.dtype, device=f.device)
    return torch.cat([z, f], dim=1)


def dequant_int4(packed, scale):
    """flip-int4: even k in the low nibble, (q - 8) * scale, block 32 along K."""
    n, kh = packed.shape
    k = kh * 2
    p8 = packed.to(torch.int16)
    lo = p8 & 0x0F
    hi = (p8 >> 4) & 0x0F
    u = torch.stack([lo, hi], dim=-1).reshape(n, k)
    q = (u - 8).float()
    s = scale.float().unsqueeze(-1).expand(n, k // 32, 32).reshape(n, k)
    return q * s


def load_int4_into(model, path):
    from safetensors.torch import load_file  # noqa: PLC0415
    blob = load_file(str(path), device="cpu")
    scales = {k[:-len(".weight_scale")]: v for k, v in blob.items()
              if k.endswith(".weight_scale")}
    sd, quant = {}, 0
    for k, v in blob.items():
        if k.endswith(".weight_scale"):
            continue
        stem = k[:-len(".weight")] if k.endswith(".weight") else k
        if stem in scales:
            sd[k] = dequant_int4(v, scales[stem])
            quant += 1
        else:
            sd[k] = v.float()
    have = set(model.state_dict())
    missing = sorted(have - set(sd))
    extra = sorted(set(sd) - have)
    model.load_state_dict(sd, strict=False)
    del sd
    return {"quantized": quant, "missing": missing[:8], "n_missing": len(missing),
            "unexpected": extra[:8], "n_unexpected": len(extra)}


def gstats(g):
    return {
        "n": int(g.get_xyz.shape[0]),
        "opacity": round(float(g.get_opacity.detach().float().mean()), 4),
        "scale": round(float(g.get_scaling.detach().float().mean()), 5),
        "x": [round(float(g.get_xyz[:, 0].min()), 3),
              round(float(g.get_xyz[:, 0].max()), 3)],
    }


def matte_scores(alpha, ref):
    a = alpha.astype(np.float32)
    b = ref.astype(np.float32)
    inter = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum() + 1e-9
    eps = 1e-6
    bce = -(b * np.log(a + eps) + (1 - b) * np.log(1 - a + eps)).mean()
    return {"iou": round(float(inter / union), 4), "bce": round(float(bce), 4),
            "fill": round(float(a.mean()), 4), "ref_fill": round(float(b.mean()), 4)}


def params_m(model):
    return round(sum(q.numel() for q in model.parameters()) / 1e6, 2)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    k3 = load_k3()
    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415

    t0 = time.time()
    pipe = k3.build_teacher_pipeline(T, dev)
    dino_s = k3.load_weights(k3.build_dino(), "dino", dev)
    vae_s = k3.load_weights(k3.build_vae(), "vae", dev)
    matte_s = k3.load_weights(k3.build_matte(), "matte", dev)
    p("models loaded", round(time.time() - t0, 1), "s")

    teacher_model = pipe.flow_model
    student_model = copy.deepcopy(teacher_model)
    t0 = time.time()
    load_info = load_int4_into(student_model, STUDENT_INT4)
    p("student int4 loaded", load_info, round(time.time() - t0, 1), "s")
    if load_info["n_missing"] or load_info["n_unexpected"]:
        p("KEY MISMATCH", load_info)
    if SMOKE:
        with torch.no_grad():
            gen = torch.Generator(device=dev).manual_seed(7)
            pipe.flow_model = student_model
            out = pipe.sample_latent({"feature1": torch.zeros(1, 5, 1280, device=dev),
                                      "feature2": torch.zeros(1, 5, 128, device=dev)},
                                     steps=2, guidance_scale=1.0, shift=1.0,
                                     generator=gen)
            g = pipe.decode_latent(out["latent"], num_gaussians=N_GAUSS)
        p("smoke ok", {k: tuple(v.shape) for k, v in out.items()}, gstats(g))
        return

    sizes = {
        "teacher_front_M": round(params_m(pipe.dinov3) + params_m(pipe.vae_encoder)
                                 + params_m(pipe.rmbg), 2),
        "student_front_M": round(params_m(dino_s) + params_m(vae_s)
                                 + params_m(matte_s), 2),
        "dit_teacher_M": params_m(teacher_model),
        "dit_student_M": params_m(student_model),
        "dit_teacher_fp16_MB": round(sum(q.numel() * q.element_size()
                                         for q in teacher_model.parameters()) / 1e6, 1),
        "dit_student_int4_MB": round(STUDENT_INT4.stat().st_size / 1e6, 1),
        "encoder_onnx_MB": {"dino": 77.6, "vae": 31.3, "matte": 11.1},
    }

    imgs = sorted(q for q in IMGDIR.rglob("*")
                  if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                  and not q.name.startswith("."))
    if LIMIT:
        imgs = imgs[:LIMIT]
    if SHARD:
        i, n = (int(v) for v in SHARD.split(":"))
        imgs = imgs[i::n]
    if not imgs:
        raise SystemExit(f"no images under {IMGDIR}")
    p("images", len(imgs), "variants", [v[0] for v in VARIANTS])

    report = {"env": {"gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
                      "torch": torch.__version__},
              "sizes": sizes, "student_load": load_info,
              "variants": [{"tag": t, "cond": c, "steps": s, "g": g, "shift": sh,
                            "weights": w}
                           for t, c, s, g, sh, w in VARIANTS],
              "images": []}
    rep_path = OUT / "report.json"

    # one warm-up call so the first timed image is not paying kernel autotune
    with torch.no_grad():
        gen = torch.Generator(device=dev).manual_seed(7)
        pipe.flow_model = teacher_model
        warm = pipe.sample_latent({"feature1": torch.zeros(1, 5, 1280, device=dev),
                                   "feature2": torch.zeros(1, 5, 128, device=dev)},
                                  steps=2, guidance_scale=3.0, shift=3.0,
                                  generator=gen)
        del warm
    p("warm-up done")

    for i, path in enumerate(imgs):
        stem = path.stem
        row = {"image": path.name, "stem": stem, "runs": {}}
        p(f"--- {i + 1}/{len(imgs)} {stem}")
        try:
            image = Image.open(path).convert("RGB")
            row["size"] = list(image.size)

            # teacher front end: its own resize, matte, crop
            with Clock() as ck:
                prepared = pipe.preprocess_image(image)
            row["t_prep_s"] = round(ck.dt, 2)
            prepared.save(OUT / f"{stem}_prepared.png")
            with Clock() as ck:
                gen = torch.Generator(device=dev).manual_seed(42)
                cond_t = pipe.encode_image(prepared, generator=gen)
            row["t_encode_s"] = round(ck.dt, 2)

            # student front end: the browser's own path
            rgb = k3.resize_min_side(image)
            w, h = rgb.size
            xin = transforms.ToTensor()(rgb).unsqueeze(0).to(dev)
            with Clock() as ck:
                with torch.no_grad():
                    a = matte_s(xin)[0, 0].float().cpu().numpy()
            row["s_matte_s"] = round(ck.dt, 2)
            alpha = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(alpha, "L").save(OUT / f"{stem}_mask.png")
            er = np.array(k3.erode_alpha(alpha, 1), dtype=np.uint8)
            box = k3.square_crop_box(er)
            rgba = rgb.convert("RGBA")
            rgba.putalpha(Image.fromarray(alpha, "L"))
            with Clock() as ck:
                cropped = rgba.crop(box).resize((k3.CANVAS, k3.CANVAS), Image.LANCZOS)
                canvas = Image.new("RGB", (k3.CANVAS, k3.CANVAS), (0, 0, 0))
                canvas.paste(cropped, mask=cropped.split()[3])
            row["s_crop_s"] = round(ck.dt, 2)
            canvas.save(OUT / f"{stem}_canvas.png")
            row["mask"] = {"resized": [w, h], "box": [int(v) for v in box],
                           "coverage": round(float((alpha > 127).mean()), 4)}

            # BiRefNet on the same resized image: the matte target, not the crop
            with Clock() as ck:
                rgba_t = pipe.rmbg.remove_background(rgb.copy())
            row["t_matte_s"] = round(ck.dt, 2)
            ref = np.array(rgba_t.split()[3].resize((w, h)), dtype=np.uint8) / 255.0
            row["matte"] = matte_scores(alpha.astype(np.float32) / 255.0, ref)

            x = transforms.ToTensor()(canvas).unsqueeze(0).to(dev)
            with Clock() as ck:
                with torch.no_grad():
                    sf1 = dino_s(x)
                    sf2 = vae_s(x * 2 - 1)
            row["s_encode_s"] = round(ck.dt, 2)
            with Clock() as ck:
                with torch.no_grad():
                    pf1 = F.layer_norm(pipe.dinov3(
                        pixel_values=T._DINOV3_NORMALIZE(x).to(
                            next(pipe.dinov3.parameters()).device,
                            next(pipe.dinov3.parameters()).dtype)).float(), (1280,))
                    pf2 = pipe.vae_encoder.encode((x * 2 - 1).to(
                        next(pipe.vae_encoder.parameters()).device,
                        next(pipe.vae_encoder.parameters()).dtype), deterministic=True)
            row["t_encode_canvas_s"] = round(ck.dt, 2)
            row["cos1"] = round(float(F.cosine_similarity(sf1, pf1, dim=-1).mean()), 4)
            row["cos2"] = round(float(F.cosine_similarity(sf2, pf2, dim=-1).mean()), 4)

            # the same comparison on the teacher's prepared image: encoder alone,
            # without our matte and crop in the way
            pc = prepared.convert("RGB")
            if pc.size != (k3.CANVAS, k3.CANVAS):
                pc = pc.resize((k3.CANVAS, k3.CANVAS), Image.LANCZOS)
            xp = transforms.ToTensor()(pc).unsqueeze(0).to(dev)
            with torch.no_grad():
                qf1 = dino_s(xp)
                qf2 = vae_s(xp * 2 - 1)
                tf1 = F.layer_norm(pipe.dinov3(
                    pixel_values=T._DINOV3_NORMALIZE(xp).to(
                        next(pipe.dinov3.parameters()).device,
                        next(pipe.dinov3.parameters()).dtype)).float(), (1280,))
                tf2 = pipe.vae_encoder.encode((xp * 2 - 1).to(
                    next(pipe.vae_encoder.parameters()).device,
                    next(pipe.vae_encoder.parameters()).dtype), deterministic=True)
            row["cos_enc1"] = round(float(F.cosine_similarity(qf1, tf1, dim=-1).mean()), 4)
            row["cos_enc2"] = round(float(F.cosine_similarity(qf2, tf2, dim=-1).mean()), 4)

            cond_s = {"feature1": sf1, "feature2": pad2(sf2)}
            latents = {}
            for tag, which, steps, gs, shift, weights in VARIANTS:
                pipe.flow_model = teacher_model if weights == "teacher" else student_model
                cond = cond_t if which == "teacher" else cond_s
                gen = torch.Generator(device=dev).manual_seed(42)
                with Clock() as ck:
                    lat = pipe.sample_latent(cond, steps=steps, guidance_scale=gs,
                                             shift=shift, generator=gen)
                sample_s = ck.dt
                latents[tag] = {k: v.detach().float().cpu() for k, v in lat.items()}
                with Clock() as ck:
                    g = pipe.decode_latent(lat["latent"], num_gaussians=N_GAUSS)
                decode_s = ck.dt
                with Clock() as ck:
                    ply = OUT / f"{stem}_{tag}.ply"
                    g.save_ply(str(ply))
                ply_s = ck.dt
                entry = {**gstats(g), "ply": ply.name,
                         "mb": round(ply.stat().st_size / 1e6, 2),
                         "steps": steps, "g": gs, "shift": shift, "cond": which,
                         "weights": weights,
                         "fwd": steps * (2 if gs > 1 else 1),
                         "sample_s": round(sample_s, 2),
                         "decode_s": round(decode_s, 2), "ply_s": round(ply_s, 2)}
                row["runs"][tag] = entry
                p(f"  {tag:4s} {which:7s} s{steps:2d} g{gs:.0f} sh{shift:.0f} "
                  f"sample {sample_s:6.1f}s decode {decode_s:4.2f}s "
                  f"op {entry['opacity']} sc {entry['scale']}")
                del g
            for tag in latents:
                if tag == "t20" or "t20" not in latents:
                    continue
                cos = float(F.cosine_similarity(latents[tag]["latent"].flatten(),
                                                latents["t20"]["latent"].flatten(), 0))
                rel = float(torch.linalg.vector_norm(
                    latents[tag]["latent"] - latents["t20"]["latent"])
                    / (torch.linalg.vector_norm(latents["t20"]["latent"]) + 1e-12))
                cam = float(F.cosine_similarity(latents[tag]["camera"].flatten(),
                                                latents["t20"]["camera"].flatten(), 0))
                row["runs"][tag]["vs_t20"] = {"latent_cos": round(cos, 4),
                                             "latent_rel_l2": round(rel, 4),
                                             "camera_cos": round(cam, 4)}
                p(f"  {tag:4s} vs t20 latent cos {cos:.4f} rel {rel:.4f} "
                  f"camera {cam:.4f}")

            del cond_t, cond_s, sf1, sf2, pf1, pf2, qf1, qf2, tf1, tf2
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            row["error"] = traceback.format_exc()[-2000:]
            p(stem, "FAILED", type(exc).__name__, str(exc)[:200])
        report["images"].append(row)
        rep_path.write_text(json.dumps(report, indent=2, default=float))

    # summary: means over images that produced each tag
    summary = {}
    for tag, *_ in VARIANTS:
        rows = [r["runs"][tag] for r in report["images"]
                if tag in r.get("runs", {})]
        if not rows:
            continue
        def mean(key, sub=""):
            vals = [r["vs_t20"][key] if sub else r[key] for r in rows
                    if (sub and "vs_t20" in r) or (not sub and key in r)]
            return round(float(np.mean(vals)), 4) if vals else None
        summary[tag] = {
            "n": len(rows),
            "sample_s": round(float(np.mean([r["sample_s"] for r in rows])), 2),
            "decode_s": round(float(np.mean([r["decode_s"] for r in rows])), 2),
            "opacity": mean("opacity"), "scale": mean("scale"),
            "latent_cos": mean("latent_cos", "sub"),
            "camera_cos": mean("camera_cos", "sub"),
        }
    report["summary"] = summary
    report["stage_s"] = {
        "t_prep": round(float(np.mean([r["t_prep_s"] for r in report["images"]
                                       if "t_prep_s" in r])), 2),
        "t_encode": round(float(np.mean([r["t_encode_s"] for r in report["images"]
                                         if "t_encode_s" in r])), 2),
        "t_matte_alone": round(float(np.mean([r["t_matte_s"] for r in report["images"]
                                              if "t_matte_s" in r])), 2),
        "s_matte": round(float(np.mean([r["s_matte_s"] for r in report["images"]
                                        if "s_matte_s" in r])), 2),
        "s_crop": round(float(np.mean([r["s_crop_s"] for r in report["images"]
                                       if "s_crop_s" in r])), 2),
        "s_encode": round(float(np.mean([r["s_encode_s"] for r in report["images"]
                                         if "s_encode_s" in r])), 2),
        "matte_iou": round(float(np.mean([r["matte"]["iou"] for r in report["images"]
                                          if "matte" in r])), 4),
        "cos1": round(float(np.mean([r["cos1"] for r in report["images"]
                                     if "cos1" in r])), 4),
        "cos2": round(float(np.mean([r["cos2"] for r in report["images"]
                                     if "cos2" in r])), 4),
        "cos_enc1": round(float(np.mean([r["cos_enc1"] for r in report["images"]
                                         if "cos_enc1" in r])), 4),
        "cos_enc2": round(float(np.mean([r["cos_enc2"] for r in report["images"]
                                         if "cos_enc2" in r])), 4),
    }
    report["secs"] = round(time.time() - t0, 1)
    rep_path.write_text(json.dumps(report, indent=2, default=float))
    p("wrote", rep_path)
    p(json.dumps({"sizes": sizes, "stage_s": report["stage_s"],
                  "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
