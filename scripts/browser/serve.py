"""Local dev server for the live splat viewer.

Serves three things over 127.0.0.1:

    /                live_viewer.html  - orbit the Gaussians with a mouse
    /models.json     manifest built by scanning a directory for *.ply
    /ply/<id>        the PLY itself
    /vendor/*        three.js + Spark, copied out of the browser harness
    /input/*         reference images (the cutout the model actually saw)

Nothing is uploaded and nothing is written outside the view directory: this is
the same PLY the Kaggle kernel dumped, read off disk and handed to the same
rasteriser the contact sheets used, except now you drive the camera.

Usage:
    python3 scripts/browser/serve.py                  # scans /tmp/flipview
    python3 scripts/browser/serve.py --ply-dir k2/out --port 8765
"""

import argparse
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shoot import ply_bounds  # noqa: E402  (pure function, no side effects)

DEFAULT_PLY_DIR = Path("/tmp/flipview")
DEFAULT_VENDOR = Path("/tmp/flipview/vendor")
VIEWER = HERE / "live_viewer.html"
BOUNDS_CACHE = Path("/tmp/flip-live").expanduser() / "bounds.json"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
SKIP_STEM = ("sheet", "pair", "contact", "probe")


# --------------------------------------------------------------------- manifest
def _cached_bounds(path: Path) -> tuple:
    """ply_bounds() rescanned on every start is 0.5 s per file; cache on mtime."""
    cache = {}
    if BOUNDS_CACHE.exists():
        try:
            cache = json.loads(BOUNDS_CACHE.read_text())
        except Exception:
            cache = {}
    key = str(path)
    stamp = path.stat().st_mtime
    hit = cache.get(key)
    if hit and abs(hit["mtime"] - stamp) < 1e-6:
        return hit["centre"], hit["radius"]
    centre, radius = ply_bounds(path)
    cache[key] = {"mtime": stamp, "centre": centre, "radius": radius}
    BOUNDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BOUNDS_CACHE.write_text(json.dumps(cache, indent=1))
    return centre, radius


def _split_stem(stem: str):
    m = re.match(r"^(?P<base>[A-Za-z_\-]+?)_?(?P<steps>\d+)$", stem)
    if m:
        return m.group("base"), int(m.group("steps"))
    return stem, None


def build_manifest(ply_dir: Path, vendor: Path) -> dict:
    if not ply_dir.is_dir():
        raise SystemExit(f"no such directory: {ply_dir}")
    if not (vendor / "spark.module.js").exists():
        raise SystemExit(
            f"missing vendored libs in {vendor}\n"
            "  three.module.js / three.core.js / spark.module.js are needed.\n"
            "  copy them from a working harness, or:\n"
            "    npm pack three @sparkjsdev/spark   # then unpack into that folder"
        )

    all_imgs = []
    for p in sorted(ply_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT and p.stat().st_size > 0:
            if not any(p.stem.startswith(s) for s in SKIP_STEM):
                all_imgs.append({"name": p.name, "url": "/input/" + p.name})

    models, seen = [], set()
    for p in sorted(ply_dir.glob("*.ply")):
        if p.stat().st_size < 1024:
            continue
        base, steps = _split_stem(p.stem)
        ident = p.stem
        if ident in seen:
            continue
        seen.add(ident)
        centre, radius = _cached_bounds(p)
        label = f"{base} · {steps} 步" if steps else base
        models.append({
            "id": ident,
            "label": label,
            "group": base,
            "steps": steps,
            "url": "/ply/" + ident,
            "path": str(p),
            "centre": [round(v, 6) for v in centre],
            "radius": round(radius, 6),
            "ply_mb": round(p.stat().st_size / 1e6, 2),
        })

    # best first: the subject with the most step-counts, high steps before low
    models.sort(key=lambda m: (m["group"] != "runner", m["group"], -(m["steps"] or 0)))

    # A reference image is the cutout the model actually saw, so attach only the
    # ones named after that subject (prepared_runner.png for "runner"). Match on
    # delimited segments -- "g" would otherwise hit every ".png".
    for m in models:
        seg = m["group"].lower()
        m["refs"] = [r for r in all_imgs
                     if seg in re.split(r"[_\-. ]+", Path(r["name"]).stem.lower())][:3]

    return {
        "ply_dir": str(ply_dir),
        "models": models,
        "refs": all_imgs,
    }


# ----------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    server_version = "flip-live/1.0"
    manifest: dict = {}
    ply_dir: Path = DEFAULT_PLY_DIR
    vendor: Path = DEFAULT_VENDOR

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _file(self, path: Path, ctype: str):
        if not path.is_file():
            return self._send(b"not found", "text/plain", 404)
        self._send(path.read_bytes(), ctype)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        route = urlparse(self.path).path
        name = unquote(route.rsplit("/", 1)[-1])

        if route in ("/", "/index.html"):
            return self._send(VIEWER.read_bytes(), "text/html; charset=utf-8")

        if route == "/models.json":
            return self._send(json.dumps(self.manifest).encode(), "application/json")

        if route.startswith("/ply/"):
            entry = next((m for m in self.manifest["models"] if m["id"] == name), None)
            if not entry:
                return self._send(b"no such model", "text/plain", 404)
            return self._file(Path(entry["path"]), "application/octet-stream")

        if route.startswith("/input/"):
            return self._file(self.ply_dir / name, "image/*")

        if route.startswith("/bench/"):
            # whitelist by basename so /bench/../../etc can't escape
            if name.endswith(".html") and Path(name).name == name and (HERE / name).is_file():
                return self._file(HERE / name, "text/html; charset=utf-8")
            return self._send(b"no such bench", "text/plain", 404)

        if route.startswith("/vendor/"):
            if name not in {"three.module.js", "three.core.js", "spark.module.js"}:
                return self._send(b"not vendored", "text/plain", 404)
            return self._file(self.vendor / name, "text/javascript")

        return self._send(b"not found", "text/plain", 404)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply-dir", default=str(DEFAULT_PLY_DIR))
    ap.add_argument("--vendor", default=str(DEFAULT_VENDOR))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="keep the default unless you really want it on the LAN")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    args = ap.parse_args(argv)

    ply_dir = Path(args.ply_dir).expanduser()
    vendor = Path(args.vendor).expanduser()
    manifest = build_manifest(ply_dir, vendor)

    if not manifest["models"]:
        raise SystemExit(f"no *.ply under {ply_dir}")
    if args.list:
        print(json.dumps(manifest, indent=2))
        return

    Handler.manifest = manifest
    Handler.ply_dir = ply_dir
    Handler.vendor = vendor

    httpd, port = None, args.port
    for p in range(args.port, args.port + 12):
        try:
            httpd = ThreadingHTTPServer((args.host, p), Handler)
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"no free port in {args.port}..{args.port + 11}")

    url = f"http://{args.host}:{port}/"
    print(f"flip live  ->  {url}")
    print(f"  ply dir  {ply_dir}")
    for m in manifest["models"]:
        print(f"  {m['label']:<22} {m['ply_mb']:>5} MB  r={m['radius']:.3f}")
    if manifest["refs"]:
        print("  refs     " + ", ".join(r["name"] for r in manifest["refs"]))
    print("  ctrl-c to stop")

    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
