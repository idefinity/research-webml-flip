"""Upload a folder to the Hugging Face Hub with nothing but the standard library.

The machine this project was developed on cannot reach huggingface.co (the proxy
answers 502 on CONNECT, and direct connections time out), so huggingface_hub is
not available and pip cannot install it. This script talks to the two API
endpoints directly:

    POST /api/repos/create                          create the repo
    POST /api/models/{repo}/upload/main/{path}      upload one file (LFS handled
                                                    by the Hub, we just send bytes)

Usage:
    python3 scripts/hf_upload.py dist/flip-student-2step --repo-id <user>/flip-student-2step
    python3 scripts/hf_upload.py ... --dry-run       print the plan, send nothing

The token comes from $HF_TOKEN, or from ./key.secret.
"""
import argparse
import http.client
import json
import os
import mimetypes
import ssl
import sys
import uuid
from pathlib import Path

HOST = "huggingface.co"


def token(explicit):
    if explicit:
        return explicit.strip()
    env = os.environ.get("HF_TOKEN", "").strip()
    if env:
        return env
    p = Path("key.secret")
    if p.exists():
        return p.read_text().strip()
    sys.exit("no token: pass --token, set HF_TOKEN, or put it in key.secret")


def api(path, tok, method="POST", body=None, ctype="application/json"):
    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(HOST, timeout=120, context=ctx)
    headers = {"Authorization": f"Bearer {tok}", "User-Agent": "flip-upload/1"}
    if body is not None:
        headers["Content-Type"] = ctype
    conn.request(method, path, body=body, headers=headers)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, data


def upload_file(path, repo_id, rel, tok):
    size = path.stat().st_size
    boundary = "----flip" + uuid.uuid4().hex
    pre = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    post = f"\r\n--{boundary}--\r\n".encode()

    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(HOST, timeout=600, context=ctx)
    conn.putrequest("POST", f"/api/models/{repo_id}/upload/main/{rel}")
    conn.putheader("Authorization", f"Bearer {tok}")
    conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
    conn.putheader("Content-Length", str(len(pre) + size + len(post)))
    conn.endheaders()
    conn.send(pre)
    sent = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            conn.send(chunk)
            sent += len(chunk)
            print(f"\r  {sent / 1e6:6.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    conn.send(post)
    print()
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r.status, body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--repo-id", required=True, help="user/name")
    ap.add_argument("--token")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.folder)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"{root}: {len(files)} files, {total / 1e6:.1f} MB")
    for p in files:
        print(f"  {p.relative_to(root)}  {p.stat().st_size / 1e6:.2f} MB")
    if args.dry_run:
        return 0

    tok = token(args.token)
    name = args.repo_id.split("/")[-1]
    st, body = api("/api/repos/create", tok, body=json.dumps(
        {"name": name, "type": "model", "private": args.private}).encode())
    print(f"create repo: {st} {body[:160].decode(errors='replace')}")

    for p in files:
        rel = p.relative_to(root).as_posix()
        print(f"uploading {rel}")
        st, body = upload_file(p, args.repo_id, rel, tok)
        print(f"  -> {st} {body[:200].decode(errors='replace')}")
        if st >= 400:
            return 1
    print(f"done: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
