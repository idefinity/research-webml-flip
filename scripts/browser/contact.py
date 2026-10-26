"""Assemble shot grids into one labelled image.

K2b's comparison sheet was built by hand in a throwaway HTML file, which is how
you get a figure nobody can regenerate. This does the same job from the filenames
shoot.py already writes (`<row>_az<deg>.png`), so any future comparison is one
command.

Usage:
    python3 contact.py --dir shots --out sheet.png \
        --rows runner_20,runner_08,runner_04 --azs 0,45,90,135,180,225,270,315
"""

import argparse
import subprocess
from pathlib import Path

OUT_ROOT = Path("/tmp/flipview")
NODE_DIR = OUT_ROOT / "node"
TILE = 320
HEAD = 46
LEFT = 160


def collect(directory, rows, azs):
    found = {}
    for p in sorted(Path(directory).glob("*.png")):
        stem = p.stem
        if "_az" not in stem:
            continue
        row, az = stem.rsplit("_az", 1)
        if not az.isdigit():
            continue
        # relative to the served root, not just the basename: the sheet page is
        # served from / and the shots live in a subdirectory
        found[(row, int(az))] = str(p.resolve().relative_to(OUT_ROOT.resolve()))
    if not found:
        return [], [], {}
    order = rows or sorted({r for r, _ in found})
    cols = azs or sorted({a for _, a in found})
    return order, cols, found


def build_html(order, cols, found):
    w = LEFT + TILE * len(cols)
    h = HEAD + TILE * len(order) + 12
    css = f"""
body{{margin:0;background:#fbfbf9;font:13px/1.4 -apple-system,Helvetica,Arial;color:#1b1b1b}}
#sheet{{position:relative;width:{w}px;height:{h}px}}
.lab{{position:absolute;font-weight:600;color:#111}}
.col{{text-align:center}}
.row{{display:flex;align-items:center;justify-content:flex-end;
  padding-right:12px;box-sizing:border-box;text-align:right}}
img{{position:absolute;object-fit:cover;background:#e9e9e4;display:block}}
.missing{{position:absolute;background:#f0efea;color:#aaa;
  display:flex;align-items:center;justify-content:center;font-size:12px}}
"""
    parts = [
        "<!doctype html><meta charset=utf-8>",
        f"<style>{css}</style><div id=sheet>",
    ]
    for c, az in enumerate(cols):
        parts.append(
            f"<div class='lab col' style='left:{LEFT + c * TILE}px;top:14px;"
            f"width:{TILE}px'>{az}\u00b0</div>")
    for r, row in enumerate(order):
        y = HEAD + r * TILE
        parts.append(
            f"<div class='lab row' style='left:0;top:{y}px;width:{LEFT}px;"
            f"height:{TILE}px'>{row}</div>")
        for c, az in enumerate(cols):
            x = LEFT + c * TILE
            name = found.get((row, az))
            if name:
                parts.append(
                    f"<img src='{name}' style='left:{x}px;top:{y}px;"
                    f"width:{TILE}px;height:{TILE}px'>")
            else:
                parts.append(
                    f"<div class='missing' style='left:{x}px;top:{y}px;"
                    f"width:{TILE}px;height:{TILE}px'>no shot</div>")
    parts.append("</div>")
    html = OUT_ROOT / "_sheet.html"
    html.write_text("\n".join(parts))
    return html, w, h


SHOT_JS = """
const { chromium } = require("playwright");
const [url, out, w, h] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch({ channel: "chrome", headless: true });
  const p = await b.newPage({
    viewport: { width: Number(w), height: Number(h) },
    deviceScaleFactor: 2,
  });
  await p.goto(url, { waitUntil: "load", timeout: 60000 });
  await p.waitForTimeout(1200);
  await p.screenshot({ path: out, fullPage: true });
  await b.close();
  console.log("wrote " + out);
})();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="shots")
    ap.add_argument("--out", default="sheet.png")
    ap.add_argument("--rows", default="")
    ap.add_argument("--azs", default="")
    args = ap.parse_args()

    rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    azs = [int(a) for a in args.azs.split(",") if a.strip()]
    order, cols, found = collect(OUT_ROOT / args.dir, rows, azs)
    if not found:
        raise SystemExit(f"no <row>_az<deg>.png under {OUT_ROOT / args.dir}")

    html, w, h = build_html(order, cols, found)
    js = NODE_DIR / "sheet.js"
    js.write_text(SHOT_JS)
    out = OUT_ROOT / args.out
    cmd = ["node", "sheet.js", f"http://127.0.0.1:8765/{html.name}",
           str(out), str(w), str(h)]
    r = subprocess.run(cmd, cwd=NODE_DIR, capture_output=True, text=True)
    print((r.stdout or r.stderr).strip())
    missing = [(row, az) for row in order for az in cols
               if (row, az) not in found]
    if missing:
        print(f"missing cells: {len(missing)} -> {missing[:8]}")
    print(f"{out}  {w}x{h}")


if __name__ == "__main__":
    main()
