"""Repack k2m's flat int4.bin + manifest.json into a safetensors file.

The manifest is a private convention: it says where every tensor lives and how
many bits it uses, and nothing else on earth can read it. safetensors carries
the same information in its header - name, dtype, shape, byte offsets - so the
file becomes self-describing and any loader can at least list the tensors.

Layout decisions:

  int4 weight    -> U8,  shape [n, k // 2]     nibbles along K, even k low
  its scales     -> F16, shape [n, k // 32]    one per block of 32 along K
  fp16 leftover  -> F16, shape as recorded     norms, biases, embeddings

Names gain a suffix so the dtype is visible without opening the file:
`blocks.0.attn.qkv` becomes `blocks.0.attn.qkv.weight` + `.weight_scale`.

    python3 scripts/to_safetensors.py assets/k2n out.safetensors

Pure Python: no torch, no safetensors package, no numpy needed.
"""
import json
import struct
import sys
from pathlib import Path

DTYPE_SIZE = {"U8": 1, "F16": 2, "F32": 4}


def entries(manifest):
    """(name, dtype, shape, src_offset, nbytes) for every tensor, sorted."""
    out = []
    for t in manifest["tensors"]:
        if t["kind"] == "int4":
            n, k = t["n"], t["k"]
            out.append((t["name"] + ".weight", "U8", [n, k // 2],
                        t["offset_q"], t["bytes_q"]))
            out.append((t["name"] + ".weight_scale", "F16", [n, k // 32],
                        t["offset_s"], t["bytes_s"]))
        else:
            out.append((t["name"], "F16", list(t["shape"]), t["offset"], t["bytes"]))
    out.sort(key=lambda e: e[0])
    return out


def main(manifest_path=None, dst=None) -> int:
    """Callable two ways on purpose: the CLI takes a directory or a manifest,
    and the Kaggle upload kernel imports this and passes both paths."""
    if manifest_path is None:
        if len(sys.argv) != 3:
            print(__doc__)
            return 2
        manifest_path, dst = Path(sys.argv[1]), Path(sys.argv[2])
    manifest_path = Path(manifest_path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    src_bin = manifest_path.with_name("int4.bin")
    dst = Path(dst)

    manifest = json.loads(manifest_path.read_text())
    blob = src_bin.read_bytes()
    if len(blob) != manifest["bytes"]:
        print(f"size mismatch: {len(blob)} vs manifest {manifest['bytes']}")
        return 1

    items = entries(manifest)

    header = {"__metadata__": {
        "format": "flip-int4",
        "bits": "4",
        "block_size": str(manifest["block_size"]),
        "symmetric": str(manifest["symmetric"]).lower(),
        "zero_point": str(manifest["zero_point"]),
        "layout": manifest["layout"],
        "source": "triposplat student, k2n distill (fuliji 32 images, 2 steps)",
    }}
    cursor = 0
    for name, dtype, shape, off, nbytes in items:
        expect = 1
        for d in shape:
            expect *= d
        expect *= DTYPE_SIZE[dtype]
        if expect != nbytes:
            print(f"{name}: shape {shape} needs {expect} bytes, manifest says {nbytes}")
            return 1
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes

    head = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(head)) % 8
    head += b" " * pad

    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for name, dtype, shape, off, nbytes in items:
            f.write(blob[off:off + nbytes])

    total = 8 + len(head) + cursor
    print(f"wrote {dst}  {total} bytes  ({total / 1e6:.1f} MB)")
    print(f"  tensors {len(items)}  (int4 {(len(items) - sum(1 for t in manifest['tensors'] if t['kind'] != 'int4')) // 2}"
          f" pairs, fp16 {sum(1 for t in manifest['tensors'] if t['kind'] != 'int4')})")
    print(f"  header {len(head)} bytes, padded {pad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
