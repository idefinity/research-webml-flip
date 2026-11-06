#!/usr/bin/env python3
"""Measure how far each generated splat cloud leans from the viewer's up axis (Y).

PCA of the gaussian centers gives three orthogonal axes; their angles to world Y
say whether the object stands along Y or is leaning. The per-gaussian minor axis
(smallest local scale) is a surface-normal proxy, so the foot/head slabs tell the
same story locally. Comparing teacher vs student on the same photo separates an
upstream (DiT/decoder/data) bias from a distillation error.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/research-webml-flip")


def load_ply(path):
    raw = path.read_bytes()
    txt, _, rest = raw.partition(b"end_header\n")
    props, n = [], None
    for line in txt.decode("ascii", "ignore").splitlines():
        t = line.split()
        if t[:2] == ["element", "vertex"]:
            n = int(t[2])
        elif t[:2] == ["property", "float"]:
            props.append(t[2])
    idx = {p: i for i, p in enumerate(props)}
    arr = np.frombuffer(rest[: n * 4 * len(props)], dtype=np.float32).reshape(n, len(props))
    return arr, idx


def ang_to(v, axis):
    return np.degrees(np.arccos(np.clip(abs(v[axis]), 0, 1)))


def analyze(path):
    arr, idx = load_ply(path)
    xyz = np.stack([arr[:, idx["x"]], arr[:, idx["y"]], arr[:, idx["z"]]], 1)
    c = xyz - xyz.mean(0)
    ev, V = np.linalg.eigh(np.cov(c.T))
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    lean = [ang_to(V[:, k], 1) for k in range(3)]          # PCA axes vs world Y
    zang = [ang_to(V[:, k], 2) for k in range(3)]          # and vs world Z

    slab = np.nan
    if "scale_0" in idx and "rot_0" in idx:
        s = np.stack([arr[:, idx[f"scale_{i}"]] for i in range(3)], 1)
        q = np.stack([arr[:, idx[f"rot_{i}"]] for i in range(4)], 1)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        w_, x_, y_, z_ = (q[:, i] for i in range(4))
        R = np.stack([
            1 - 2 * (y_**2 + z_**2), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_),
            2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_**2 + z_**2), 2 * (y_ * z_ - x_ * w_),
            2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_**2 + y_**2),
        ], 1).reshape(-1, 3, 3)
        minor = s.argmin(1)
        nrm = R[np.arange(len(R)), :, minor]
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
        yy = xyz[:, 1]
        lo, hi = np.quantile(yy, [0.03, 0.97])
        cap = nrm[(yy <= lo) | (yy >= hi)]
        if len(cap):
            slab = np.degrees(np.arccos(np.clip(np.abs(cap[:, 1]).mean(), 0, 1)))

    span = xyz.max(0) - xyz.min(0)
    height_ratio = span[1] / max(span[0], span[2])
    return {
        "stem": path.stem, "n": len(xyz),
        "lean": lean, "zang": zang, "ev": ev, "slab": slab,
        "span": span, "hr": height_ratio,
        "xyz": xyz, "V": V,
    }


def main():
    pats = sys.argv[1:] or ["w/qc/ply/hf_00*_*.ply"]
    files = sorted({f for p in pats for f in ROOT.glob(p)})
    if not files:
        sys.exit(f"no files for {pats}")
    rows = [analyze(f) for f in files]
    print(f"{'file':<20}{'tag':<9}{'PCA1>Y':>7}{'PCA2>Y':>7}{'PCA3>Y':>7}"
          f"{'slab>Y':>7}{'span x/y/z':>18}{'h/w':>6}")
    for r in rows:
        tag = r["stem"].split("_")[-1]
        print(f"{r['stem']:<20}{tag:<9}{r['lean'][0]:7.1f}{r['lean'][1]:7.1f}"
              f"{r['lean'][2]:7.1f}{r['slab']:7.1f}"
              f"   {r['span'][0]:5.2f} {r['span'][1]:5.2f} {r['span'][2]:5.2f}"
              f"{r['hr']:6.2f}")
    print()
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["stem"].split("_")[-1], []).append(r)
    for tag, g in sorted(by_tag.items()):
        best = np.array([min(r["lean"]) for r in g])
        zbest = np.array([min(r["zang"]) for r in g])
        print(f"{tag:<9} n={len(g):<3} min-lean-to-Y mean {best.mean():5.1}deg "
              f"median {np.median(best):5.1} p90 {np.percentile(best, 90):5.1} "
              f"| min-lean-to-Z {zbest.mean():5.1}deg "
              f"| slab {np.nanmean([r['slab'] for r in g]):5.1}deg "
              f"| h/w {np.mean([r['hr'] for r in g]):.2f}")

    if len(by_tag) > 1:
        print("\npairwise vs teacher (same photo): angle between PCA axes + centroid shift")
        tch = {r["stem"].split("_")[0]: r for r in rows if r["stem"].endswith("teacher")}
        for tag, g in sorted(by_tag.items()):
            if tag == "teacher":
                continue
            d = []
            for r in g:
                t = tch.get(r["stem"].split("_")[0])
                if not t:
                    continue
                a = np.degrees(np.arccos(np.clip(abs(np.dot(r["V"][:, 0], t["V"][:, 0])), 0, 1)))
                d.append((a, np.linalg.norm(r["xyz"].mean(0) - t["xyz"].mean(0))))
            if d:
                dd = np.array(d)
                print(f"{tag:<9} major-axis vs teacher mean {dd[:, 0].mean():5.1f}deg "
                      f"median {np.median(dd[:, 0]):5.1f} | centroid offset mean "
                      f"{dd[:, 1].mean():.3f}")


if __name__ == "__main__":
    main()
