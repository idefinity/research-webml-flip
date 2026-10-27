"""Audit an image set before spending GPU hours distilling on it.

The question this answers is narrow and practical: does this set look like the
input distribution the model was built for, and is it big enough to be worth
anything after duplicates are removed?

Five numbers, each of which has bitten this project already:

  * resolution - the pipeline resizes to 512; a 300px source is an upscale, and
    every straightening error downstream gets blamed on the student.
  * aspect - object-centric renders are near-square. Phone photos are 3:2 or
    9:16, and the subject gets cropped by the resize, not by the model.
  * duplicates - 222 images from one shoot are not 222 samples. A perceptual
    hash finds the near-copies that inflate a "we have more data now" claim.
  * border busyness - fraction of border pixels sitting on an edge. Clean
    backgrounds score low; a cluttered street scores high. This is the cheapest
    proxy for "can anything segment a single subject out of this".
  * subject occupancy - how much of the frame differs from the border colour.
    An object render fills the middle; a portrait crop also does, which is why
    occupancy alone does not settle it.

Usage:
  python3 scripts/audit_image_set.py <dir|parquet> [more dirs...]

Writes a json report next to nothing in particular; prints a table.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = (".png", ".jpg", ".jpeg", ".webp")
TARGET = 512


def load_images(src):
    """Accepts a directory or a parquet with an image column."""
    src = Path(src)
    if src.is_dir():
        return [(p.name, p) for p in sorted(src.iterdir())
                if p.suffix.lower() in EXTS]
    if src.suffix == ".parquet":
        import pyarrow.parquet as pq  # noqa: PLC0415
        out = []
        tbl = pq.read_table(src)
        img_col = None
        for name in tbl.column_names:
            if "image" in name.lower():
                img_col = name
                break
        if img_col is None:
            raise SystemExit(f"no image column in {tbl.column_names}")
        for i, cell in enumerate(tbl[img_col].to_pylist()):
            if isinstance(cell, dict) and cell.get("bytes"):
                blob = cell["bytes"]
            elif isinstance(cell, (bytes, bytearray)):
                blob = bytes(cell)
            else:
                continue
            out.append((f"row{i:04d}", blob))
        return out
    raise SystemExit(f"cannot read {src}")


def as_image(item):
    name, v = item
    if isinstance(v, (bytes, bytearray)):
        import io  # noqa: PLC0415
        return name, Image.open(io.BytesIO(v)).convert("RGB")
    return name, Image.open(v).convert("RGB")


def dhash(img, size=8):
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def edge_fraction(gray, mask):
    """Fraction of the masked pixels that sit on a strong gradient.

    PIL has no canny and pulling in cv2 for this is not worth it: a thresholded
    sum of absolute differences ranks a clean background below a cluttered
    street, which is all this number is asked to do.
    """
    a = np.asarray(gray, dtype=np.float32)
    mag = np.zeros_like(a)
    # diff shortens each axis by one, so both contributions land on the same
    # (h-1, w-1) window
    mag[:-1, :-1] += np.abs(np.diff(a, axis=1))[:-1, :]
    mag[:-1, :-1] += np.abs(np.diff(a, axis=0))[:, :-1]
    return float((mag[mask] > 24).mean())


def border_mask(h, w, frac=0.15):
    m = np.zeros((h, w), dtype=bool)
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    m[:bh, :] = m[-bh:, :] = True
    m[:, :bw] = m[:, -bw:] = True
    return m


def audit(items):
    rows = []
    hashes = {}
    for item in items:
        try:
            name, img = as_image(item)
        except Exception as exc:  # noqa: BLE001
            rows.append({"name": name if isinstance(item[0], str) else "?", "error": str(exc)[:120]})
            continue
        w, h = img.size
        small = img.resize((TARGET, TARGET), Image.BILINEAR)
        g = small.convert("L")
        bm = border_mask(TARGET, TARGET)
        a = np.asarray(g, dtype=np.float32)

        # occupancy: how far the middle is from the border's mean colour
        border_mean = float(a[bm].mean())
        inner = float(np.abs(a[~bm] - border_mean).mean())

        eh = dhash(img)
        hashes.setdefault(eh, []).append(name)

        rows.append({
            "name": name, "w": w, "h": h,
            "aspect": round(w / h, 3),
            "min_side": min(w, h),
            "border_edge_fraction": round(edge_fraction(g, bm), 4),
            "border_contrast_to_inner": round(inner, 2),
        })
    dups = {h: n for h, n in hashes.items() if len(n) > 1}
    return rows, dups


def summarise(label, rows, dups):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print(f"{label}: no readable images")
        return {}
    sides = np.array([r["min_side"] for r in ok])
    aspects = np.array([r["aspect"] for r in ok])
    busy = np.array([r["border_edge_fraction"] for r in ok])
    occ = np.array([r["border_contrast_to_inner"] for r in ok])
    s = {
        "label": label, "n": len(ok),
        "min_side_p10": int(np.percentile(sides, 10)),
        "min_side_median": int(np.median(sides)),
        "below_512": int((sides < TARGET).sum()),
        "aspect_median": round(float(np.median(aspects)), 3),
        "aspect_iqr": round(float(np.percentile(aspects, 75)
                                  - np.percentile(aspects, 25)), 3),
        "square_ish_0.9_1.1": int(((aspects >= 0.9) & (aspects <= 1.1)).sum()),
        "border_edge_fraction_median": round(float(np.median(busy)), 4),
        "occupancy_median": round(float(np.median(occ)), 2),
        "unique_hashes": len({r["name"] for r in ok}) - sum(
            len(v) - 1 for v in dups.values()),
        "duplicate_groups": len(dups),
    }
    print(f"\n{label}  (n={s['n']})")
    print(f"  shortest side      p10 {s['min_side_p10']}  median {s['min_side_median']}"
          f"  below {TARGET}px: {s['below_512']} ({100*s['below_512']/s['n']:.0f}%)")
    print(f"  aspect             median {s['aspect_median']}  iqr {s['aspect_iqr']}"
          f"  square-ish: {s['square_ish_0.9_1.1']} ({100*s['square_ish_0.9_1.1']/s['n']:.0f}%)")
    print(f"  border edge frac   median {s['border_edge_fraction_median']}")
    print(f"  occupancy          median {s['occupancy_median']}")
    print(f"  near-duplicates    {s['duplicate_groups']} groups, "
          f"{s['unique_hashes']} distinct of {s['n']}")
    return s


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    report = {}
    for src in argv:
        try:
            items = load_images(src)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"{src}: {exc}")
            continue
        rows, dups = audit(items)
        report[Path(src).name or "set"] = summarise(src, rows, dups)
    print()
    print(json.dumps(report, indent=1)[:2000])


if __name__ == "__main__":
    main(sys.argv[1:])
