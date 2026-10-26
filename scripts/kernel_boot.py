"""Tiny kernel bootstrap for Kaggle.

Paste this file into a Kaggle notebook cell with FLIP_SHA and FLIP_SCRIPT set,
and run it. It fetches `scripts/kaggle_kernel_body.py` at that commit and
executes it, so a whole run is identified by one SHA.

Why this exists: the body is ~150 lines that install packages and clone the
upstream repo, and pasting it by hand every run is where transcription bugs
come from. Fetching it by SHA gives the same integrity guarantee a manual copy
did - a mismatch is impossible, the bytes are pinned - without the paste.

raw.githubusercontent.com is blocked on the local machine but reachable from
Kaggle; api.github.com serves the same bytes and is the fallback.
"""

import base64
import json
import urllib.request

FLIP_REPO = "idefinity/research-webml-flip"
FLIP_SHA = "REPLACE_ME"
FLIP_SCRIPT = "scripts/k0f_which_op.py"


def _fetch(path):
    urls = [
        f"https://raw.githubusercontent.com/{FLIP_REPO}/{FLIP_SHA}/{path}",
        f"https://api.github.com/repos/{FLIP_REPO}/contents/{path}?ref={FLIP_SHA}",
    ]
    err = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flip-boot"})
            with urllib.request.urlopen(req, timeout=60) as fh:
                raw = fh.read()
            if "api.github.com" in url:
                raw = base64.b64decode(json.loads(raw)["content"])
            return raw.decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            err = exc
    raise RuntimeError(f"could not fetch {path} at {FLIP_SHA}: {err}")


print(f"FLIP_SHA={FLIP_SHA} FLIP_SCRIPT={FLIP_SCRIPT}", flush=True)
exec(compile(_fetch("scripts/kaggle_kernel_body.py"), "kaggle_kernel_body.py", "exec"))
