"""CPU 3DGS rasterizer + contact sheets for the k13 PLYs.

Camera follows site/src/viewer.js: the PLY is Y-down (positions flipped), home
azimuth 3.35, elevation 0.18, Y up.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL = 448
ART = Path(os.environ.get("FLIP_ART", "/tmp/k13_kaggle/artifacts"))
OUT = Path(os.environ.get("FLIP_SHEETS", "/tmp/k13_kaggle/sheets"))
OUT.mkdir(parents=True, exist_ok=True)


def read_ply(path):
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.read(1)
        head = head.decode("latin1")
        props, count = [], 0
        for ln in head.split("\n"):
            if ln.startswith("element vertex"):
                count = int(ln.split()[2])
            if ln.startswith("property float"):
                props.append(ln.split()[-1])
        row = np.dtype([(p, "<f4") for p in props])
        body = np.frombuffer(f.read(count * row.itemsize), dtype=row, count=count)
    return body


def quat_to_rot(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    r = np.empty((len(q), 3, 3), np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def render(path, W=CELL, H=CELL, az=3.35, el=0.18, bg=0.05):
    b = read_ply(path)
    xyz = np.nan_to_num(np.stack([b["x"], b["y"], b["z"]], 1).astype(np.float64))
    xyz[:, 1] *= -1.0
    sc = np.clip(np.exp(np.nan_to_num(np.stack(
        [b["scale_0"], b["scale_1"], b["scale_2"]], 1).astype(np.float64))), 1e-8, 1.0)
    op = 1.0 / (1.0 + np.exp(-np.nan_to_num(b["opacity"].astype(np.float64))))
    rgb = np.clip(0.5 + 0.28209479177387814 * np.stack(
        [b["f_dc_0"], b["f_dc_1"], b["f_dc_2"]], 1).astype(np.float64), 0, 1)
    R = quat_to_rot(np.stack([b["rot_0"], b["rot_1"], b["rot_2"], b["rot_3"]], 1).astype(np.float64))

    lo, hi = xyz.min(0), xyz.max(0)
    target = (lo + hi) / 2.0
    radius = max(0.5 * float(np.linalg.norm(hi - lo)), 1e-6)
    fov = 0.70
    dist = radius / np.sin(fov / 2) * 1.08
    eye = target + dist * np.array([np.sin(az) * np.cos(el),
                                    np.sin(el),
                                    np.cos(az) * np.cos(el)])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    Rv = np.stack([right, up, fwd])
    if not np.isfinite(Rv).all():
        raise RuntimeError("camera is not finite")

    tc = (xyz - eye) @ Rv.T
    z = tc[:, 2]
    keep = z > 1e-3
    xyz, sc, op, rgb, R, tc, z = (a[keep] for a in (xyz, sc, op, rgb, R, tc, z))
    if len(z) == 0:
        return Image.new("RGB", (W, H), (int(bg * 255),) * 3)

    f = 0.5 * H / np.tan(fov / 2)
    px = W / 2.0 + f * tc[:, 0] / z
    py = H / 2.0 - f * tc[:, 1] / z

    cov = R @ (sc[:, :, None] ** 2 * np.eye(3)[None]) @ np.transpose(R, (0, 2, 1))
    iz = 1.0 / z
    J = np.zeros((len(z), 2, 3))
    J[:, 0, 0] = f * iz
    J[:, 0, 2] = -f * tc[:, 0] * iz * iz
    J[:, 1, 1] = f * iz
    J[:, 1, 2] = -f * tc[:, 1] * iz * iz
    cov2 = J @ cov @ np.transpose(J, (0, 2, 1))
    a = cov2[:, 0, 0] + 1e-6
    c = cov2[:, 1, 1] + 1e-6
    bb = cov2[:, 0, 1]
    lam = 0.5 * (a + c) + np.sqrt(np.maximum(0.25 * (a - c) ** 2 + bb * bb, 0))
    rad = 2.4 * np.sqrt(lam)
    ok = (rad > 0.35) & (rad < 0.25 * max(W, H)) & (px > -rad) & (px < W + rad) & (py > -rad) & (py < H + rad)
    idx = np.nonzero(ok)[0]
    order = idx[np.argsort(-z[idx])]
    inv = np.linalg.inv(cov2[ok])

    out = np.full((H, W, 3), bg, np.float64)
    for n, i in enumerate(order):
        r = rad[i]
        x0 = int(max(0, np.floor(px[i] - r)))
        x1 = int(min(W - 1, np.ceil(px[i] + r)))
        y0 = int(max(0, np.floor(py[i] - r)))
        y1 = int(min(H - 1, np.ceil(py[i] + r)))
        if x0 > x1 or y0 > y1:
            continue
        dx = (np.arange(x0, x1 + 1) - px[i])[None, :]
        dy = (np.arange(y0, y1 + 1) - py[i])[:, None]
        e = inv[n, 0, 0] * dx * dx + 2 * inv[n, 0, 1] * dx * dy + inv[n, 1, 1] * dy * dy
        al = np.clip(op[i] * np.exp(-0.5 * e), 0.0, 1.0)
        if al.max() < 1e-3:
            continue
        sub = out[y0:y1 + 1, x0:x1 + 1]
        sub *= (1.0 - al)[:, :, None]
        sub += rgb[i][None, None, :] * al[:, :, None]
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def cell(path, label, size=CELL):
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    strip = Image.new("RGB", (size, 26), (0, 0, 0))
    ImageDraw.Draw(strip).text((6, 4), label, fill=(255, 255, 255), font=font(18))
    out = Image.new("RGB", (size, size + 26), (0, 0, 0))
    out.paste(im, (0, 26))
    out.paste(strip, (0, 0))
    return out


def prep(path, size):
    out = ART / "png" / f"{path.stem}_{size}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        render(path, W=size, H=size).save(out)
        print("rendered", out.name, flush=True)
    return out


def cell(path, label, size=CELL):
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    strip = Image.new("RGB", (size, 26), (0, 0, 0))
    ImageDraw.Draw(strip).text((6, 4), label, fill=(255, 255, 255), font=font(18))
    out = Image.new("RGB", (size, size + 26), (0, 0, 0))
    out.paste(im, (0, 26))
    out.paste(strip, (0, 0))
    return out


def sheet(rows, out_path, size=CELL):
    cols = max(len(r) for r in rows)
    W, H = cols * size, len(rows) * (size + 26)
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    for r, row in enumerate(rows):
        for c, (path, label) in enumerate(row):
            canvas.paste(cell(path, label, size), (c * size, r * (size + 26)))
    canvas.save(out_path, quality=92)
    print("wrote", out_path, canvas.size)
    return out_path


def main():
    d = ART / "k13_eval"
    pre = ART / "k2d"
    names = sorted(set(p.name.rsplit("_t20.ply", 1)[0] for p in d.glob("*_t20.ply")))
    eps = sorted(set(p.name.split("_s04_")[-1].replace(".ply", "")
                     for p in d.glob("*_s04_e*.ply")))
    print("photos", names, "epochs", eps)
    rows = []
    for i, n in enumerate(names):
        row = [(pre / f"prepared_{i:04d}.png", n.split("_")[1][:4])]
        for t in ("t20", "t04", "g04"):
            row.append((prep(d / f"{n}_{t}.ply", CELL), t))
        for e in eps:
            row.append((prep(d / f"{n}_s04_{e}.ply", CELL), "s04-" + e))
        rows.append(row)
    sheet(rows, OUT / "sheet_main.jpg")

    if eps and names:
        n0 = names[0]
        row = [(pre / f"prepared_{0:04d}.png", n0.split("_")[1][:4])]
        for t in ("t20", "t04", "g04"):
            row.append((prep(d / f"{n0}_{t}.ply", 512), t))
        for e in eps:
            row.append((prep(d / f"{n0}_s04_{e}.ply", 512), "s04-" + e))
        sheet([row], OUT / "sheet_first.jpg", size=512)


if __name__ == "__main__":
    main()
