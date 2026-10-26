"""Shoot a 3DGS PLY from several azimuths with a real rasteriser.

Nothing here invents a renderer - it drives Spark (the library the TripoSplat
repo's own viewer uses) in headless Chrome and screenshots the canvas. The only
things this script decides are the camera position and the background.

The centre and radius are read from the PLY header + positions rather than from
Spark's Box3, because Spark does not expose a populated bounding box at a
predictable moment; guessing when it is ready is how you end up with four blank
screenshots that look like "the model is empty".

Usage:
    python3 shoot.py g04.ply g20.ply --outdir shots
"""

import argparse
import json
import struct
import subprocess
from pathlib import Path

NODE_DIR = Path("/tmp/flipview/node")
OUT = Path("/tmp/flipview")


def ply_bounds(path):
    """Read a binary little-endian PLY and return (centre, radius)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError(f"{path}: no end_header")
    hdr = raw[:end].decode("ascii", "replace").splitlines()
    n_vert, props = 0, []
    in_vertex = False
    for line in hdr:
        t = line.split()
        if not t:
            continue
        if t[0] == "element":
            in_vertex = t[1] == "vertex"
            if in_vertex:
                n_vert = int(t[2])
            continue
        if t[0] == "property" and in_vertex and t[1] == "float":
            props.append(t[2])
    off = end + len(b"end_header")
    off += 1 if raw[off:off + 1] in (b"\n", b"\r") else 0
    stride = 4 * len(props)
    ix = [props.index(k) for k in ("x", "y", "z")]
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for i in range(n_vert):
        base = off + i * stride
        for a, k in enumerate(ix):
            v = struct.unpack_from("<f", raw, base + 4 * k)[0]
            if v < lo[a]:
                lo[a] = v
            if v > hi[a]:
                hi[a] = v
    centre = [(lo[a] + hi[a]) / 2 for a in range(3)]
    # radius of the bounding sphere, so tall thin objects still fit
    radius = 0.5 * max(hi[a] - lo[a] for a in range(3))
    return centre, radius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plys", nargs="+")
    ap.add_argument("--outdir", default="shots")
    ap.add_argument("--az", default="0,90,180,270")
    ap.add_argument("--el", default="0")
    ap.add_argument("--pad", default="1.45")
    args = ap.parse_args()

    outdir = OUT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    azs = [a.strip() for a in args.az.split(",")]

    rows = []
    for p in args.plys:
        path = Path(p)
        centre, radius = ply_bounds(path)
        cstr = ",".join(f"{v:.6f}" for v in centre)
        for az in azs:
            out = outdir / f"{path.stem}_az{int(az):03d}.png"
            cmd = ["node", "shot.js", path.name, az, str(out),
                   args.el, args.pad, "40", cstr, f"{radius:.6f}"]
            r = subprocess.run(cmd, cwd=NODE_DIR, capture_output=True, text=True)
            line = (r.stdout or r.stderr).strip().splitlines()
            info = line[-1] if line else ""
            print(f"{path.name} az{az} -> {out.name}  {info}", flush=True)
            rows.append({"ply": path.name, "az": az, "out": str(out),
                         "centre": centre, "radius": radius, "log": info})
    (outdir / "shoot_report.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
