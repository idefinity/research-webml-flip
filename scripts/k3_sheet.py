"""K3 sheet - four renders per photo, so quality is looked at instead of argued about.

The acceptance test we never ran: put the teacher's 20-step result next to what our
chain produces for the same photo, and look. Numbers like "lat cos 0.84" are a proxy
for this image, and they were made on cosplay studio photos, not on the photos people
actually drop in.

Four PLYs per input, all through the teacher's own decoder so the only differences are
the ones we care about:

    t20      teacher conditions, 20 steps, its own sampler (guidance 3, shift 3)  <- truth
    t2       teacher conditions,  2 steps, guidance 3                          <- cost of fewer steps
    s2       student conditions,  2 steps, guidance 3                          <- cost of our encoder
    s2g      student conditions,  2 steps, guidance 1, shift 1                 <- what the browser runs

Also written per image: the canvas the student encoder saw (1024^2, black), the alpha
the student matte produced, and the teacher's own prepared image. When a sheet cell is
fog, one of those three says which half broke.

Run it from the k3 driver (stage "sheet"), or directly:
    FLIP_WORK=/kaggle/working python3 k3_sheet.py /kaggle/working/eval
"""

import importlib.util
import json
import os
import pathlib
import sys
import tarfile
import time

WORK = pathlib.Path(os.environ.get("FLIP_WORK", "/kaggle/working"))
K3 = WORK / "k3_encoder_distill.py"
IMGDIR = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else str(WORK / "eval"))
OUT = WORK / "k3" / "sheet"
OUT.mkdir(parents=True, exist_ok=True)
# 高斯数决定 decoder 取多少个八叉树点（每点 32 个）：65536 -> 2048 点，
# 262144 -> 全部 8192 点。浏览器按 65536 做预算，教师要证明"原厂水平"就得给满。
N_GAUSS = int(os.environ.get("FLIP_SHEET_GAUSS", "65536"))

# 输入分辨率：upstream 的 preprocess_image 用模块级 _CANVAS_SIZE，改它会同时改
# 抠图/裁切/最终画布。DINO patch 数按 size/16 变：1024 -> 4101 token，
# 512 -> 1029 token。flow model 的 cond 是 concat + self-attn，长度不写死；
# 输出高斯数不受影响。只有显式设了 FLIP_SHEET_IN 才去动 upstream 的全局。
IN_SIZE = int(os.environ.get("FLIP_SHEET_IN", "1024"))

os.environ.setdefault("FLIP_WORK", str(WORK))


def p(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_k3():
    spec = importlib.util.spec_from_file_location("k3", str(K3))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pad2(f):
    import torch  # noqa: PLC0415
    z = torch.zeros(f.shape[0], 5, f.shape[2], dtype=f.dtype, device=f.device)
    return torch.cat([z, f], dim=1)


def parse_variants():
    """tag:cond:steps:guidance:shift, comma separated.

    cond is "t" (teacher conditions) or "s" (student conditions). The student was
    distilled with the guidance folded into the target, so its contract grid is
    g1/shift1; running it at the teacher's g3 is informative but off-contract.
    """
    specs = os.environ.get(
        "FLIP_SHEET_VARIANTS",
        "t20:t:20:3:3,t2:t:2:3:3,s2:s:2:3:3,s2g:s:2:1:1").split(",")
    out = []
    for spec in specs:
        tag, which, steps, gs, shift = [s.strip() for s in spec.split(":")]
        out.append((tag, which, int(steps), float(gs), float(shift)))
    return out


def stats(g):
    return {
        "n": int(g.get_xyz.shape[0]),
        "opacity": round(float(g.get_opacity.detach().float().mean()), 4),
        "scale": round(float(g.get_scaling.detach().float().mean()), 5),
        "x": [round(float(g.get_xyz[:, 0].min()), 3),
              round(float(g.get_xyz[:, 0].max()), 3)],
    }


def main():
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from torchvision import transforms  # noqa: PLC0415

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p("device", dev, "torch", torch.__version__)
    k3 = load_k3()
    sys.path.insert(0, str(k3.REPO))
    import triposplat as T  # noqa: PLC0415
    if IN_SIZE != 1024:
        T._CANVAS_SIZE = IN_SIZE
        p("input canvas patched ->", IN_SIZE, "(teacher DINO tokens ->",
          (IN_SIZE // 16) ** 2 + 5, ")")

    imgs = sorted(q for q in IMGDIR.rglob("*")
                  if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                  and not q.name.startswith("."))
    limit = int(os.environ.get("FLIP_SHEET_N", "0"))
    if limit:
        imgs = imgs[:limit]
    if not imgs:
        raise SystemExit(f"no images under {IMGDIR}")
    p("images", len(imgs), [q.name for q in imgs])
    variants = parse_variants()
    p("variants", [f"{t}·{c}{s}st·g{g}·sh{sh}" for t, c, s, g, sh in variants])

    t0 = time.time()
    pipe = k3.build_teacher_pipeline(T, dev)
    p("teacher pipeline", round(time.time() - t0, 1), "s")
    dino_s = k3.load_weights(k3.build_dino(), "dino", dev)
    vae_s = k3.load_weights(k3.build_vae(), "vae", dev)
    matte_s = k3.load_weights(k3.build_matte(), "matte", dev)
    p("student encoders loaded")

    report = []
    for i, path in enumerate(imgs):
        stem = path.stem
        row = {"image": path.name, "stem": stem, "runs": {}}
        try:
            t_img = time.time()
            image = Image.open(path).convert("RGB")
            row["size"] = list(image.size)

            # --- teacher path: its own preprocess, its own encoder
            prepared = pipe.preprocess_image(image)
            prepared.save(OUT / f"{stem}_prepared.png")
            row["prepared"] = list(prepared.size)
            gen = torch.Generator(device=dev).manual_seed(42)
            cond_t = pipe.encode_image(prepared, generator=gen)

            # --- student path: mirror of site/src/pipeline.js loadCondition
            rgb = k3.resize_min_side(image)
            w, h = rgb.size
            xin = transforms.ToTensor()(rgb).unsqueeze(0).to(dev)
            with torch.no_grad():
                a = matte_s(xin)[0, 0].float().cpu().numpy()
            alpha = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(alpha, "L").save(OUT / f"{stem}_mask.png")
            er = np.array(k3.erode_alpha(alpha, 1), dtype=np.uint8)
            box = k3.square_crop_box(er)
            rgba = rgb.convert("RGBA")
            rgba.putalpha(Image.fromarray(alpha, "L"))
            cropped = rgba.crop(box).resize((k3.CANVAS, k3.CANVAS), Image.LANCZOS)
            canvas = Image.new("RGB", (k3.CANVAS, k3.CANVAS), (0, 0, 0))
            canvas.paste(cropped, mask=cropped.split()[3])
            canvas.save(OUT / f"{stem}_canvas.png")
            fx = alpha.astype(np.float32) / 255.0
            row["mask"] = {
                "resized": [w, h],
                "box": [int(v) for v in box],
                "fill": round(float(fx.mean()), 4),
                "coverage": round(float((fx > 0.5).mean()), 4),
            }

            x = transforms.ToTensor()(canvas).unsqueeze(0).to(dev)
            dino_dev = next(pipe.dinov3.parameters()).device
            dino_dt = next(pipe.dinov3.parameters()).dtype
            vae_dev = next(pipe.vae_encoder.parameters()).device
            vae_dt = next(pipe.vae_encoder.parameters()).dtype

            def teacher_feats(t):
                with torch.no_grad():
                    t1 = F.layer_norm(pipe.dinov3(
                        pixel_values=T._DINOV3_NORMALIZE(t).to(dino_dev, dino_dt)
                    ).float(), (1280,))
                    t2 = pipe.vae_encoder.encode(
                        (t * 2 - 1).to(vae_dev, vae_dt), deterministic=True)
                return t1, t2

            with torch.no_grad():
                sf1 = dino_s(x)
                sf2 = vae_s(x * 2 - 1)
                # teacher features on the SAME canvas: isolates encoder error from crop
                tf1, tf2 = teacher_feats(x)
            row["cos1"] = round(float(F.cosine_similarity(sf1, tf1, dim=-1).mean()), 4)
            row["cos2"] = round(float(F.cosine_similarity(sf2, tf2, dim=-1).mean()), 4)

            # and once more on the teacher's own prepared image: encoder alone, no
            # matte and no crop of ours in the way. The gap between the two pairs is
            # what our matte costs, which cos1 above can't separate.
            pc = prepared.convert("RGB")
            if pc.size != (k3.CANVAS, k3.CANVAS):
                pc = pc.resize((k3.CANVAS, k3.CANVAS), Image.LANCZOS)
            xp = transforms.ToTensor()(pc).unsqueeze(0).to(dev)
            with torch.no_grad():
                pf1 = dino_s(xp)
                pf2 = vae_s(xp * 2 - 1)
                qf1, qf2 = teacher_feats(xp)
            row["cos_enc1"] = round(float(F.cosine_similarity(pf1, qf1, dim=-1).mean()), 4)
            row["cos_enc2"] = round(float(F.cosine_similarity(pf2, qf2, dim=-1).mean()), 4)
            p(stem, "cos", row["cos1"], row["cos2"], "| encoder only",
              row["cos_enc1"], row["cos_enc2"], "| mask",
              row["mask"]["coverage"], "box", row["mask"]["box"])

            cond_s = {"feature1": sf1, "feature2": pad2(sf2)}
            for tag, which, steps, gs, shift in variants:
                cond = cond_t if which == "t" else cond_s
                t1 = time.time()
                gen = torch.Generator(device=dev).manual_seed(42)
                lat = pipe.sample_latent(cond, steps=steps, guidance_scale=gs,
                                         shift=shift, generator=gen)
                g = pipe.decode_latent(lat["latent"], num_gaussians=N_GAUSS)
                ply = OUT / f"{stem}_{tag}.ply"
                g.save_ply(str(ply))
                row["runs"][tag] = {
                    **stats(g), "ply": ply.name, "steps": steps,
                    "g": gs, "shift": shift, "cond": which,
                    "mb": round(ply.stat().st_size / 1e6, 2),
                    "secs": round(time.time() - t1, 1),
                }
                p(f"  {tag:6s} s{steps} g{gs} sh{shift} {row['runs'][tag]['secs']:6.1f}s "
                  f"op {row['runs'][tag]['opacity']} sc {row['runs'][tag]['scale']}")
            row["secs"] = round(time.time() - t_img, 1)
        except Exception as exc:  # noqa: BLE001
            import traceback  # noqa: PLC0415
            row["error"] = traceback.format_exc()[-2000:]
            p(stem, "FAILED", type(exc).__name__, str(exc)[:200])
        report.append(row)
        (OUT / "report.json").write_text(json.dumps(report, indent=2))

    tgz = OUT / "sheet.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        for f in sorted(OUT.iterdir()):
            if f.name != tgz.name:
                tf.add(f, arcname=f.name)
    p("wrote", tgz, round(tgz.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
