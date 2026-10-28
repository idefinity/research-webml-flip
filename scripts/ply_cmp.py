"""Compare two TripoSplat PLYs the way the viewer reads them.

The PLY stores pre-activation values (opacity as a logit, scale as a log), so
the numbers that mean anything are sigmoid(opacity) and exp(scale). Both runs
are the same image and the same seed; the only thing that can move them is the
weights and the decoder's own randomness.
"""
import sys

import numpy as np


def read_ply(path):
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.read(1)
        head = head.decode("latin1")
        props = []
        count = 0
        for ln in head.split("\n"):
            if ln.startswith("element vertex"):
                count = int(ln.split()[2])
            if ln.startswith("property float"):
                props.append(ln.split()[-1])
        row = np.dtype([(p, "<f4") for p in props])
        body = np.frombuffer(f.read(count * row.itemsize), dtype=row, count=count)
        return props, body


def stats(path):
    props, b = read_ply(path)
    xyz = np.stack([b["x"], b["y"], b["z"]], 1)
    sc = np.stack([b["scale_0"], b["scale_1"], b["scale_2"]], 1)
    op = 1 / (1 + np.exp(-b["opacity"]))
    return {
        "n": len(b),
        "opacity": float(op.mean()),
        "scale": float(np.exp(sc).mean()),
        "bbox_min": [float(v) for v in xyz.min(0)],
        "bbox_max": [float(v) for v in xyz.max(0)],
        "center": [float(v) for v in (xyz.min(0) + xyz.max(0)) / 2],
        "xyz": xyz,
    }


def main():
    pairs = []
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        pairs.append((args[i], args[i + 1]))
    if not pairs:
        print(__doc__)
        return 2
    out = {}
    for tag, path in pairs:
        s = stats(path)
        out[tag] = s
        print(f"{tag:28s} n={s['n']}  opacity {s['opacity']:.4f}  scale {s['scale']:.5f}")
        print(f"{'':28s} bbox min {np.round(s['bbox_min'], 3)}  max {np.round(s['bbox_max'], 3)}")
    # nearest-neighbour distance between the two clouds, as a coarse shape check
    if len(pairs) == 2:
        a = out[pairs[0][0]]["xyz"]
        b = out[pairs[1][0]]["xyz"]
        step = max(1, len(a) // 4096)
        sa = a[::step]
        d = np.linalg.norm(sa[:, None, :] - b[None, ::step, :], axis=2)
        print(f"抽样 {len(sa)}×{len(b[::step])} 最近邻距离: 中位 {np.median(d.min(1)):.4f} "
              f"均值 {d.min(1).mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
