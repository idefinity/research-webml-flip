"""Build a manifest.json for site/ply-gallery.html.

The gallery is a static page: it cannot list a directory, so something has to
enumerate the PLYs. That something is this script — run it after any sheet run
and the page picks up whatever is new.

    python3 scripts/make_ply_manifest.py out/teacher-2026-10-29

Pairs ``<stem>_<tag>.ply`` with ``<stem>.webp|jpg|png`` under input_example/ so
each entry can show the photo it came from, and reads ``report.json`` in the
same directory (if present) for the per-run numbers.

Paths in the manifest are relative to the repo root, which is what the page
prepends its own ``?root=`` to.
"""

import argparse
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
PHOTO_DIR = REPO / "input_example"

# 前缀缩写 -> 读得懂的名字。没匹配上的原样显示。
TAGS = {
    "t20": "teacher cut-out · 20 steps",
    "t20hi": "teacher cut-out · 20 steps · 262k",
    "t20s": "our cut-out · 20 steps",
    "t2": "teacher cut-out · 2 steps",
    "t2s1": "teacher · 2 steps · g3/sh1",
    "t2s1g": "teacher · 2 steps · g1/sh1",
    "t4s1": "teacher · 4 steps · g3/sh1",
    "t8s1": "teacher · 8 steps · g3/sh1",
    "s2": "our cut-out · 2 steps",
    "s2g": "our cut-out · 2 steps · g1/sh1",
}


def photo_for(stem):
    for ext in (".webp", ".jpg", ".jpeg", ".png"):
        p = PHOTO_DIR / f"{stem}{ext}"
        if p.exists():
            return str(p.relative_to(REPO))
    return None


def load_report(d):
    f = d / "report.json"
    if not f.exists():
        return {}
    try:
        rows = json.loads(f.read_text())
    except json.JSONDecodeError:
        return {}
    return {r.get("stem"): r for r in rows if isinstance(r, dict)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = pathlib.Path(args.dir).resolve()
    out = pathlib.Path(args.out) if args.out else d / "manifest.json"
    report = load_report(d)

    entries = []
    for ply in sorted(d.glob("*.ply")):
        m = re.match(r"^(?P<stem>.+?)_(?P<tag>[a-z0-9]+)$", ply.stem)
        if not m:
            continue
        stem, tag = m.group("stem"), m.group("tag")
        row = report.get(stem, {})
        run = (row.get("runs") or {}).get(tag, {})
        entries.append({
            "stem": stem,
            "tag": tag,
            "label": TAGS.get(tag, tag),
            "ply": str(ply.relative_to(REPO)),
            "photo": photo_for(stem),
            "opacity": run.get("opacity"),
            "gaussians": run.get("n"),
            "secs": run.get("secs"),
            "mb": run.get("mb"),
        })

    if not entries:
        raise SystemExit(f"no *.ply under {d}")

    out.write_text(json.dumps(entries, indent=1))
    stems = sorted({e["stem"] for e in entries})
    tags = sorted({e["tag"] for e in entries})
    print(f"{len(entries)} entries · {len(stems)} photos · {len(tags)} variants")
    print(f"  {out}")
    print(f"  open: http://127.0.0.1:8899/site/ply-gallery.html?manifest=../{out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
