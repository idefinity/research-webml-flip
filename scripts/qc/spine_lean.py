#!/usr/bin/env python3
"""Physical lean of a splat cloud, measured in 3D and not on the silhouette.

Slice the cloud along the viewer's up axis (Y), take the centre of mass of each
slice, fit a straight line through those centres: its slope in the X and Z
directions is how far the object's spine drifts from vertical. Silhouette
asymmetry (an arm, a tail) cancels out over slices, so a few degrees here means
the geometry really is off-axis.
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/workspace/research-webml-flip/scripts/qc")
from tilt_probe import load_ply  # noqa: E402

ROOT = Path("/workspace/research-webml-flip")
NSL = 12


def spine(arr, idx):
    xyz = np.stack([arr[:, idx["x"]], arr[:, idx["y"]], arr[:, idx["z"]]], 1)
    w = np.asarray(1.0 / (1.0 + np.exp(-(arr[:, idx["opacity"]] if "opacity" in idx else 0))), np.float32)
    y = xyz[:, 1]
    lo, hi = np.quantile(y, [0.08, 0.92])
    if hi - lo < 1e-3:
        return None
    edges = np.linspace(lo, hi, NSL + 1)
    xs, cz, cy = [], [], []
    for k in range(NSL):
        m = (y >= edges[k]) & (y < edges[k + 1])
        if m.sum() < 50:
            continue
        ww = w[m]
        xs.append(np.average(xyz[m, 0], weights=ww))
        cz.append(np.average(xyz[m, 2], weights=ww))
        cy.append(y[m].mean())
    if len(cy) < 5:
        return None
    cy = np.array(cy)
    sx = np.polyfit(cy, np.array(xs), 1)[0]
    sz = np.polyfit(cy, np.array(cz), 1)[0]
    drift = np.hypot(np.array(xs) - np.mean(xs), np.array(cz) - np.mean(cz))
    return {
        "lean_x": float(np.degrees(np.arctan(sx))),     # tilt toward/away ±X
        "lean_z": float(np.degrees(np.arctan(sz))),
        "mag": float(np.hypot(sx, sz) and np.degrees(np.arctan(np.hypot(sx, sz)))),
        "wob": float(drift.std()),
        "height": float(hi - lo),
    }


def main(pats):
    files = sorted({f for p in pats for f in ROOT.glob(p)})
    if not files:
        sys.exit(f"no files for {pats}")
    by = defaultdict(list)
    per = defaultdict(dict)
    for f in files:
        parts = f.stem.split("_")
        tag = parts[-1]
        stem = "_".join(parts[:-1])
        arr, idx = load_ply(f)
        s = spine(arr, idx)
        if not s:
            continue
        by[tag].append(s)
        per[stem][tag] = s
    print(f"{'tag':<9}{'n':>4}  |lean| mean  median  p10   p90   signed lean_x  lean_z  wob  height")
    for tag, g in sorted(by.items()):
        mag = np.array([x["mag"] for x in g])
        print(f"{tag:<9}{len(g):>4}{mag.mean():12.1f}{np.median(mag):8.1f}"
              f"{np.percentile(mag,10):7.1f}{np.percentile(mag,90):7.1f}"
              f"{np.mean([x['lean_x'] for x in g]):12.1f}{np.mean([x['lean_z'] for x in g]):8.1f}"
              f"{np.mean([x['wob'] for x in g]):6.3f}{np.mean([x['height'] for x in g]):8.2f}")
    common = [s for s, d in per.items() if "teacher" in d and len(d) > 2]
    print(f"\nper-photo |lean| (deg), {len(common)} photos with all variants:")
    tags = sorted({t for s in common for t in per[s]})
    print("  " + "photo".ljust(11) + "".join(f"{t:>9}" for t in tags) + "   student-teacher")
    for s in sorted(common):
        vals = [per[s][t]["mag"] for t in tags]
        st = [per[s][t]["mag"] for t in tags if t != "teacher"]
        print("  " + s.ljust(11) + "".join(f"{v:9.1f}" for v in vals)
              + f"{np.mean(st) - per[s]['teacher']['mag']:14.1f}")
    d = np.array([np.mean([per[s][t]["mag"] for t in tags if t != "teacher"]) - per[s]["teacher"]["mag"]
                  for s in common])
    print(f"\nstudent minus teacher |lean|: mean {d.mean():+.1f}deg  median {np.median(d):+.1f}  "
          f"win(less tilted) {int((d < 0).sum())}/{len(d)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["w/qc/ply/hf_00*_*.ply"])
