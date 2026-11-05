#!/bin/zsh
# Serve the repo root and open the app. One command, Ctrl-C to stop.
#
#   ./scripts/browser/serve_site.sh            # port 8899, the app
#   ./scripts/browser/serve_site.sh 8900       # another port
#   NO_OPEN=1 ./scripts/browser/serve_site.sh  # don't launch a browser
#   GALLERY=out/teacher-2026-10-29 ./scripts/browser/serve_site.sh
#                                              # open the PLY gallery instead
#
# Why the repo root and not site/: the page reads ../assets/*.jpg, and with
# ?local=../w/ the weights and the ONNX Runtime dist. A server rooted at site/
# 404s all of that, which looks like a model bug and isn't one.
#
# The URL carries ?local=../w/&ort=/w/ort/ because this machine cannot reach
# huggingface.co or cdn.jsdelivr.net from a browser (see RESUME.md). Drop both
# if you are on a network where the page can fetch from HF and jsDelivr.
set -e
PORT=${1:-8899}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
URL="http://127.0.0.1:${PORT}/site/index.html?local=../w/&ort=/w/ort/"

if [ -n "$GALLERY" ]; then
  # GALLERY points at a directory holding *.ply + manifest.json
  # (scripts/make_ply_manifest.py writes the manifest).
  DIR=${GALLERY%/}
  [ -f "$ROOT/$DIR/manifest.json" ] || {
    echo "no $DIR/manifest.json — run: python3 scripts/make_ply_manifest.py $DIR"
    exit 1
  }
  URL="http://127.0.0.1:${PORT}/site/ply-gallery.html?manifest=../${DIR}/manifest.json"
fi

cd "$ROOT"
[ -d w ] || { echo "no w/ here — the weights are local-only; see RESUME.md"; exit 1; }
[ -d w/ort ] || echo "note: w/ort is missing, ONNX Runtime will fall back to jsDelivr"

python3 -m http.server "$PORT" --bind 127.0.0.1 >/tmp/flip-serve.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

sleep 1
CODE=$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "$URL")
echo "serve ${CODE}  ${URL}"
[ "$CODE" = "200" ] || { echo "server did not answer; log: /tmp/flip-serve.log"; exit 1; }
echo "  ctrl-c to stop   (requests: /tmp/flip-serve.log)"

[ -n "$NO_OPEN" ] || open "$URL"
wait $SRV
