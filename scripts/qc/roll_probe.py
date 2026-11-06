#!/usr/bin/env python3
"""Where does the lean come from: the photo, or the model?

A) photo roll  - dominant near-vertical edge orientation in the *background*
                 (Sobel gradient histogram). A Dutch-angle shot shows up here.
B) subject lean - slab-centroid spine of the matted silhouette (canvas_*.npy,
                 background already removed), i.e. how the person stands in the
                 frame the encoder actually sees.
C) 3D lean     - spine of the generated cloud (same estimator as spine_lean).

Correlating A/B with C says whether the generator copies the camera's roll.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/research-webml-flip/scripts/qc")
from tilt_probe import load_ply  # noqa: E402
from spine_lean import spine  # noqa: E402

ROOT = Path("/workspace/research-webml-flip")
CACHE = Path("/root/encoder-upgrade/cache")
PHOTO = ROOT / "w/qc/photos"


def photo_roll(path, size=512):
    a = np.asarray(Image.open(path).convert("L").resize((size, size)), np.float32) / 255
    gy, gx = np.gradient(a)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx + 1e-9))          # -180..180
    ang = ((ang + 90) % 180) - 90                        # edge direction, fold to -90..90
    near_v = np.abs(np.abs(ang) - 90) < 45               # edges close to vertical
    sel = mag[near_v]
    if sel.sum() < 1e-3:
        return np.nan
    # energy-weighted mean deviation from vertical (signed, mod 180 ambiguity aside)
    dev = ang[near_v] - np.sign(ang[near_v]) * 90
    return float(np.average(dev, weights=sel))


def subject_lean(arr):
    """canvas npy is CHW uint8 on black; spine of the non-black silhouette."""
    g = arr.mean(0) if arr.ndim == 3 else arr
    m = g > 12
    ys, xs = np.nonzero(m)
    if len(ys) < 500:
        return np.nan
    y0, y1 = ys.min(), ys.max()
    span = y1 - y0
    if span < 40:
        return np.nan
    lo = y0 + 0.15 * span
    hi = y1 - 0.15 * span
    bot = ys <= lo
    top = ys >= hi
    if bot.sum() < 50 or top.sum() < 50:
        return np.nan
    dx = xs[top].mean() - xs[bot].mean()
    dy = ys[top].mean() - ys[bot].mean()
    return float(np.degrees(np.arctan2(dx, dy))) if abs(dy) > 1 else np.nan


def main():
    stems = [f.stem for f in sorted(PHOTO.glob("hf_00*_*.jpg")) if "thumb" not in f.name] \
        if False else [f"hf_{i:04d}" for i in range(24)]
    print(f"{'photo':<10}{'photoRoll':>10}{'subjLean':>10}{'teach3D':>9}{'v1':>7}{'v2':>7}{'v3':>7}")
    rows = []
    for s in stems:
        p = PHOTO / f"{s}.jpg"
        if not p.exists():
            continue
        pr = photo_roll(p)
        cpath = CACHE / f"canvas_{s}.npy"
        sl = subject_lean(np.load(cpath)) if cpath.exists() else np.nan
        d = {}
        for tag in ("teacher", "v1", "v2", "v3"):
            f = ROOT / f"w/qc/ply/{s}_{tag}.ply"
            if not f.exists():
                d[tag] = np.nan
                continue
            arr, idx = load_ply(f)
            r = spine(arr, idx)
            d[tag] = r["lean_x"] if r else np.nan
        rows.append((s, pr, sl, d))
        print(f"{s:<10}{pr:10.1f}{sl:10.1f}{d['teacher']:9.1f}{d['v1']:7.1f}"
              f"{d['v2']:7.1f}{d['v3']:7.1f}")
    ok = [r for r in rows if np.isfinite(r[1])]
    pr = np.array([r[1] for r in ok])
    sl = np.array([r[2] for r in ok])
    print(f"\nphoto roll: mean {pr.mean():+.1f}deg  |mean| {np.abs(pr).mean():.1f}  "
          f"sd {pr.std():.1f}  range {pr.min():+.1f}..{pr.max():+.1f}")
    if np.isfinite(sl).sum() > 3:
        m = np.isfinite(sl)
        print(f"subject lean in frame: mean {np.nanmean(sl):+.1f}deg  "
              f"|mean| {np.nanmean(np.abs(sl)):.1f}  corr(photoRoll) "
              f"{np.corrcoef(pr[m], sl[m])[0,1]:+.2f}")
    for tag in ("teacher", "v1", "v2", "v3"):
        v = np.array([r[3][tag] for r in ok])
        c = np.corrcoef(pr, v)[0, 1]
        print(f"3D lean_x {tag:<8} mean {v.mean():+6.1f}  |mean| {np.abs(v).mean():5.1f}  "
              f"corr with photo roll {c:+.2f}")


main()
