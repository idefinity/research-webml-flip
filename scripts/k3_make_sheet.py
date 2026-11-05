"""Assemble the four-render comparison sheet.

Rows are photos, columns are: the photo itself, then
    teacher 20 steps | teacher 2 steps | student cond 2 steps | student cond, browser grid.

Renders come from site/ply-test.html screenshots (one PNG per PLY, named
<stem>_<tag>.png); the photo itself is copied from the eval directory. Pure PIL,
no browser, so it can be re-run cheaply whenever a render changes.

    python3 scripts/k3_make_sheet.py --shots /tmp/sheet/shots \\
        --images input_example --out assets/k3/sheet.png
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

TAGS = [
    ("t20", "teacher 20"),
    ("t2", "teacher 2"),
    ("s2", "student 2"),
    ("s2g", "student 2 · browser grid"),
]
TILE = 300
HEAD = 34
LEFT = 150


def label_font():
    try:
        from PIL import ImageFont
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", required=True)
    ap.add_argument("--images", required=True, help="originals, for the first column")
    ap.add_argument("--out", default="sheet.png")
    ap.add_argument("--report", default="", help="report.json from k3_sheet.py")
    args = ap.parse_args()

    shots = Path(args.shots)
    imgs = Path(args.images)
    stems = sorted({p.stem.rsplit("_", 1)[0] for p in shots.glob("*.png")
                    if p.stem.rsplit("_", 1)[-1] in dict(TAGS)})
    if not stems:
        raise SystemExit(f"no <stem>_<tag>.png under {shots}")

    report = {}
    if args.report and Path(args.report).is_file():
        for row in json.loads(Path(args.report).read_text()):
            report[row.get("stem", "")] = row

    cols = 1 + len(TAGS)
    W = LEFT + cols * TILE
    H = HEAD + len(stems) * TILE
    sheet = Image.new("RGB", (W, H), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    font = label_font()

    def text(xy, s, fill=(210, 210, 205)):
        d.text(xy, s, fill=fill, font=font)

    text((14, 10), "photo")
    for j, (_, name) in enumerate(TAGS):
        text((LEFT + j * TILE + 8, 10), name)

    for i, stem in enumerate(stems):
        y = HEAD + i * TILE
        row = report.get(stem, {})
        note = row.get("image", stem)
        if "cos1" in row:
            note = f"{stem}\ncos1 {row['cos1']}\ncos2 {row['cos2']}"
        for k, line in enumerate(str(note).split("\n")[:4]):
            text((8, y + 8 + k * 17), line)

        # first column: the original photo
        cand = [p for p in imgs.rglob(f"{stem}.*") if p.suffix.lower() != ".py"]
        if cand:
            im = Image.open(cand[0]).convert("RGB")
            im.thumbnail((TILE - 8, TILE - 8))
            sheet.paste(im, (LEFT - TILE + (TILE - im.width) // 2 + 60,
                             y + (TILE - im.height) // 2))
        for j, (tag, _) in enumerate(TAGS):
            f = shots / f"{stem}_{tag}.png"
            if not f.is_file():
                text((LEFT + j * TILE + 8, y + TILE // 2), "missing")
                continue
            im = Image.open(f).convert("RGB")
            im.thumbnail((TILE - 8, TILE - 8))
            sheet.paste(im, (LEFT + j * TILE + (TILE - im.width) // 2,
                             y + (TILE - im.height) // 2))

    # thin column separators, so a row reads left to right
    for j in range(cols + 1):
        x = LEFT - TILE + 56 + j * TILE - 56
        d.line([(LEFT + j * TILE, 0), (LEFT + j * TILE, H)], fill=(34, 34, 34))
    sheet.save(args.out)
    print(args.out, sheet.size, len(stems), "photos")


if __name__ == "__main__":
    main()
