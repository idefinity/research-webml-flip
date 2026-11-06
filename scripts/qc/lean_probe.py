#!/usr/bin/env python3
"""Perceived lean of a rendered object, measured in image space.

For each view we take the silhouette (pixels away from the corner colour), then
compare the horizontal centroid of the bottom slab with that of the top slab:
lean = atan2(dx, dy). A perfectly upright object gives ~0; a leaning one gives
the angle of the imaginary plumb line. Signed, so a consistent left/right lean
shows up as a non-zero mean rather than averaging away.

teacher vs student on the same photo separates an inherited pose bias from
something our distillation added.
"""
import glob
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

FRAC = 0.25


def mask_of(im):
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    h, w, _ = a.shape
    edge = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3),
                           a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    bg = np.median(edge, 0)
    d = np.abs(a - bg).sum(2)
    m = d > 40
    if m.sum() < 0.005 * h * w:
        m = d > 15
    return m


def lean_of(m):
    ys, xs = np.nonzero(m)
    if len(ys) < 200:
        return None
    y0, y1 = ys.min(), ys.max()
    span = y1 - y0
    if span < 20:
        return None
    lo = ys <= y0 + FRAC * span
    hi = ys >= y1 - FRAC * span
    cx_lo, cx_hi = xs[lo].mean(), xs[hi].mean()
    dy = (ys[hi].mean() - ys[lo].mean())
    dx = cx_hi - cx_lo
    ang = np.degrees(np.arctan2(dx, dy)) if dy else np.nan
    width = xs.max() - xs.min() + 1
    height = span + 1
    return ang, 0.5 * (m[:, xs].any(1).sum() and height / max(width, 1))


def main(pat):
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no files for {pat}")
    acc = defaultdict(list)
    rows = []
    for f in files:
        b = f.split("/")[-1]
        mm = re.match(r"(.*?_)?(teacher|v\d+)_(front|side|back)\.png", b)
        if not mm:
            continue
        tag, view = mm.group(2), mm.group(3)
        r = lean_of(mask_of(Image.open(f)))
        if not r:
            continue
        ang, hw = r
        acc[(tag, view)].append((b.split("_teacher")[0].split("_v")[0], ang, hw))
        rows.append((tag, view, ang, hw))
    print(f"{'tag':<9}{'view':<7}{'n':>4}  signed lean (deg)          |h|  h/w")
    for (tag, view), g in sorted(acc.items()):
        a = np.array([x[1] for x in g])
        hw = np.array([x[2] for x in g])
        print(f"{tag:<9}{view:<7}{len(a):>4}  mean {a.mean():+6.1f} med {np.median(a):+6.1f} "
              f"min {a.min():+6.1f} max {a.max():+6.1f}  | {np.abs(a).mean():5.1f} "
              f"{np.nanmean(hw):5.2f}")
    print("\nfront-view lean by photo (deg, signed):")
    fr = {t: {x[0]: x[1] for x in g} for t, g in acc.items() if g and g[0] and True}
    front = {t: dict((x[0], x[1]) for x in g) for (t, v), g in acc.items() if v == "front"}
    stems = sorted({s for d in front.values() for s in d})
    print("  " + "photo".ljust(12) + "".join(f"{t:>10}" for t in sorted(front)))
    n = 0
    for s in stems[:24]:
        vals = [front[t].get(s) for t in sorted(front)]
        if any(v is None for v in vals):
            continue
        print("  " + s.ljust(12) + "".join(f"{v:10.1f}" for v in vals))
        n += 1
    for t in sorted(front):
        d = np.array([v for s, v in front[t].items()])
        print(f"  {t:<12} mean {d.mean():+6.1f}  |mean| {np.abs(d).mean():5.1f}  sd {d.std():5.1f}")


main(sys.argv[1] or "/workspace/research-webml-flip/w/qc/views/*_front.png")
