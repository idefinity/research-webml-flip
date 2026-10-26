"""K2d - record what the flow DiT actually does, step by step.

Why: the 4-step problem is not solvable by choosing a nicer number of steps.
The base model at 4 steps loses material (K2b, K2c): skin goes pale, black
shorts go pink, hair turns to flying threads. A 4-step *student* is a different
model - it is trained to land the same place in 4 steps that the teacher takes
20 for. That needs the teacher's velocity field as a function of (state, t),
which is what this script records, once, on Kaggle.

The trick that keeps this short: do not reimplement the sampler. Run the real
`pipe.sample_latent` and intercept the flow module's forward. Every call gives
(state, t, cond) -> velocity, for free, including both branches of CFG.

Write this file so it runs on two GPUs without a rewrite:

  * `FLIP_SHARD=i/n` traces only images i, i+n, i+2n ... and writes
    `trace_part<i>.json`. Nothing else changes, so a shard is a complete,
    self-contained run.
  * With no shard set and >= 2 visible GPUs, the script becomes a coordinator:
    it spawns one `FLIP_SHARD=i/n` child per GPU with CUDA_VISIBLE_DEVICES
    pinned, streams their logs, and merges the parts. Any shard that dies is
    re-run serially in the parent, so one bad card costs time, not the run.
  * `FLIP_FORK=0` forces the single-process path - the same script, same
    output, just slower. That is the fallback if the box is short on RAM.

Writes k2d/trace_XXXX.pt per image plus trace.json summary.

Env:
  FLIP_TRACE_N     images to trace (default 8; the smoke run)
  FLIP_TRACE_STEPS teacher steps (default 20)
  FLIP_TRACE_GAIN  guidance scale (default 3.0, the pipeline default)
  FLIP_TRACE_SHIFT flow schedule shift (default 3.0)
  FLIP_SHARD       "i/n" - trace one shard, see above
  FLIP_FORK        "auto" (default) | "0" - two-GPU coordination on or off
  FLIP_FORK_MIN_GB refuse to fork below this much free RAM (default 16)
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k2d"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_IMG = int(os.environ.get("FLIP_TRACE_N", "8"))
STEPS = int(os.environ.get("FLIP_TRACE_STEPS", "20"))
GAIN = float(os.environ.get("FLIP_TRACE_GAIN", "3.0"))
SHIFT = float(os.environ.get("FLIP_TRACE_SHIFT", "3.0"))
SHARD = os.environ.get("FLIP_SHARD", "").strip()
FORK = os.environ.get("FLIP_FORK", "auto").strip().lower()
FORK_MIN_GB = float(os.environ.get("FLIP_FORK_MIN_GB", "16"))
SEED = 42

# Tensors below this size stay inline; above it they get deduplicated. cond
# features are 4101x1280 and identical on every one of the 40 calls per image,
# so without this the trace is ~400 MB of the same tensor copied 40 times.
BIG = 4096

R = {"calls": [], "images": [], "notes": []}


def note(msg):
    line = str(msg)
    print(line, flush=True)
    R["notes"].append(line)


# ---------------------------------------------------------------------------
# what hardware am I on
# ---------------------------------------------------------------------------

def probe_gpus():
    """Record the facts the training plan depends on, instead of assuming them.

    Kaggle's 2xT4 is two separate cards. Everything downstream - fp16 vs bf16,
    whether DDP is worth it, whether a batch has to be split - follows from
    these numbers, and they are cheap to collect.
    """
    info = {"count": 0, "visible": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}
    if not torch.cuda.is_available():
        return info
    info["count"] = torch.cuda.device_count()
    try:
        info["bf16"] = bool(torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        info["bf16"] = None
    info["devices"] = []
    for i in range(info["count"]):
        p = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "i": i, "name": p.name, "sm": f"{p.major}.{p.minor}",
            "mem_gb": round(p.total_memory / 1e9, 1),
        })
    try:
        info["p2p"] = [[bool(torch.cuda.can_device_access_peer(i, j))
                        for j in range(info["count"])]
                       for i in range(info["count"])]
    except Exception as exc:  # noqa: BLE001
        info["p2p"] = f"unavailable: {type(exc).__name__}"
    return info


def mem_available_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1e6, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

def _hash(t):
    """Cheap content id for a tensor, so identical cond features are stored once."""
    if t is None:
        return "none"
    b = t.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha1(b).hexdigest()[:16]


class Recorder:
    """Wraps the flow module's forward and keeps every call.

    Patching the class, not the instance: `__call__` resolves on the type, so an
    instance attribute would be silently bypassed. K0e already burned me once on
    exactly this - a monkeypatch that never entered the graph.

    The patch is a plain function, so it binds like any other method and `self`
    is the flow module. Assigning a bound method instead would make the module
    itself land in `args` and get saved into the trace.
    """

    def __init__(self, flow):
        self.flow = flow
        self.calls = []
        self._cls = flow.__class__
        self._orig = self._cls.forward
        self._cls.forward = self._make()
        self.t0 = time.time()

    def _make(self):
        rec = self

        def patched(self, *a, **kw):
            # CUDA events, not wall clock: the kernel is async, so time.time()
            # around a launch measures the launch. The elapsed numbers also tell
            # us what a backward pass costs, which is 2x this.
            ev0 = ev1 = None
            try:
                if torch.cuda.is_available():
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
            except Exception:  # noqa: BLE001
                ev0 = ev1 = None
            out = rec._orig(self, *a, **kw)
            if self is rec.flow:
                try:
                    if ev1 is not None:
                        ev1.record()
                    rec._keep(a, kw, out, ev0, ev1)
                except Exception:  # noqa: BLE001 - never break the sampler
                    note("record failed: " + traceback.format_exc()[-500:])
            return out

        return patched

    def sync(self):
        """Resolve the pending CUDA events after the sampler finishes."""
        for c in self.calls:
            ev = c.pop("_ev", None)
            if ev is None:
                continue
            try:
                e0, e1 = ev
                e1.synchronize()
                c["ms"] = round(e0.elapsed_time(e1), 1)
            except Exception:  # noqa: BLE001
                c["ms"] = None

    def restore(self):
        self._cls.forward = self._orig

    def _keep(self, a, kw, out, ev0=None, ev1=None):
        """Store the call verbatim rather than guessing its signature.

        I know the module takes ({latent, camera}, t*1000, {feature1, feature2})
        from the export adapter, but I do not know how `sample_latent` actually
        calls it - positionally, by keyword, or through a wrapper. Guessing wrong
        produces a clean-looking trace with no states in it, so keep the raw
        shape and sort it out in the summary.
        """
        def small(x):
            if torch.is_tensor(x):
                return x.detach().float().cpu().half()
            if isinstance(x, dict):
                return {k: small(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [small(v) for v in x]
            if isinstance(x, nn.Module):
                return f"<module {type(x).__name__}>"
            return x

        self.calls.append({
            "args": [small(x) for x in a],
            "kwargs": {k: small(v) for k, v in kw.items()},
            "out": small(out),
            "_ev": (ev0, ev1) if ev0 is not None else None,
        })


def _walk(t, fn):
    """Apply fn to every tensor nested anywhere in args/kwargs/out."""
    if torch.is_tensor(t):
        fn(t)
    elif isinstance(t, dict):
        for v in t.values():
            _walk(v, fn)
    elif isinstance(t, (list, tuple)):
        for v in t:
            _walk(v, fn)


def compact(calls):
    """Replace repeated big tensors with a `$ref` into a shared table.

    The cond features do not change between steps - they are computed once per
    image - so 40 verbatim copies is 400 MB of duplicated bytes per image and
    it also makes the training loader re-hash them every epoch.
    """
    counts = {}

    def count(t):
        if t.numel() >= BIG:
            h = _hash(t)
            counts[h] = counts.get(h, 0) + 1

    for c in calls:
        _walk(c["args"], count)
        _walk(c["kwargs"], count)
        _walk(c["out"], count)

    shared, stored = {}, set()

    def repl(t):
        if torch.is_tensor(t):
            if t.numel() < BIG:
                return t
            h = _hash(t)
            if counts.get(h, 0) < 2:
                return t
            if h not in stored:
                stored.add(h)
                shared[h] = t
            return {"$ref": h}
        if isinstance(t, dict):
            return {k: repl(v) for k, v in t.items()}
        if isinstance(t, (list, tuple)):
            return [repl(v) for v in t]
        return t

    for c in calls:
        c["args"] = repl(c["args"])
        c["kwargs"] = repl(c["kwargs"])
        c["out"] = repl(c["out"])
    return shared


def load_trace(path):
    """Read a trace back, resolving `$ref` markers. Used by the student trainer."""
    p = torch.load(path, map_location="cpu", weights_only=False)

    def rehydrate(x):
        if isinstance(x, dict):
            if "$ref" in x and len(x) == 1:
                return p["shared"][x["$ref"]]
            return {k: rehydrate(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rehydrate(v) for v in x]
        return x

    p["calls"] = [{"args": rehydrate(c["args"]),
                   "kwargs": rehydrate(c["kwargs"]),
                   "out": rehydrate(c["out"]),
                   **{k: v for k, v in c.items() if k not in ("args", "kwargs", "out")}}
                  for c in p["calls"]]
    return p


def _shape(x):
    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, dict):
        if "$ref" in x and len(x) == 1:
            return f"$ref:{x['$ref']}"
        return {k: _shape(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_shape(v) for v in x]
    if isinstance(x, (int, float)):
        return f"scalar:{x}"
    return type(x).__name__


def unwrap(rec):
    """Summarise the recorded calls so the next script can be written against
    the real signature: shapes, scalar t values, which features repeat."""
    rows = []
    for i, c in enumerate(rec.calls):
        flat = []
        for v in list(c["args"]) + list(c["kwargs"].values()):
            if torch.is_tensor(v) and v.numel() == 1:
                flat.append(float(v.flatten()[0]))
        feats = []
        for v in list(c["args"]) + list(c["kwargs"].values()):
            if isinstance(v, dict):
                for k, t in v.items():
                    if torch.is_tensor(t):
                        feats.append(f"{k}:{_hash(t)}")
        rows.append({
            "i": i,
            "t": flat[0] if flat else None,
            "args": [_shape(v) for v in c["args"]],
            "kwargs": {k: _shape(v) for k, v in c["kwargs"].items()},
            "out": _shape(c["out"]),
            "feat": feats,
            "ms": c.get("ms"),
        })
    return rows


# ---------------------------------------------------------------------------
# the work
# ---------------------------------------------------------------------------

def pick_images():
    ex = REPO / "static" / "example_inputs"
    cands = sorted(p for p in ex.glob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not cands:
        cands = [p for p in sorted((REPO / "static").rglob("*"))
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    return cands[:N_IMG]


def build_pipeline():
    sys.path.insert(0, str(REPO))
    import triposplat as T  # noqa: PLC0415

    files = sorted(CKPT.rglob("*.safetensors"))

    def pick(*keys):
        for f in files:
            n = f.name.lower()
            if all(k in n for k in keys):
                return f
        return None

    dino_p, vae_p = pick("dino"), (pick("vae") or pick("flux"))
    rmbg_p = pick("biref") or pick("rmbg")
    dec_p = pick("decoder") or pick("octree")
    taken = {p for p in (dino_p, vae_p, rmbg_p, dec_p) if p}
    ck = [f for f in files if f not in taken][0]

    pipe = T.TripoSplatPipeline(
        ckpt_path=str(ck), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))
    mods = {type(v).__name__: v for v in vars(pipe).values() if isinstance(v, nn.Module)}
    flow = mods.get("LatentSeqMMFlowModel")
    if flow is None:
        raise RuntimeError(f"no LatentSeqMMFlowModel in pipeline; saw {list(mods)}")
    return pipe, flow


def trace_one(pipe, flow, idx, img_path):
    """Trace one image. Returns its summary row, or None on failure."""
    rec = Recorder(flow)
    try:
        image = Image.open(img_path).convert("RGB")
        prepared = pipe.preprocess_image(image)
        try:
            prepared.save(OUT / f"prepared_{idx:04d}.png")
        except Exception:  # noqa: BLE001
            pass
        gen = torch.Generator(device=DEV).manual_seed(SEED + idx)
        cond = pipe.encode_image(prepared, generator=gen)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        gen_i = torch.Generator(device=DEV).manual_seed(SEED + idx)
        lat = pipe.sample_latent(cond, steps=STEPS, guidance_scale=GAIN,
                                 shift=SHIFT, generator=gen_i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        rec.sync()

        rows = unwrap(rec)
        shared = compact(rec.calls)
        payload = {
            "image": img_path.name,
            "steps": STEPS,
            "guidance": GAIN,
            "shift": SHIFT,
            "seed": SEED + idx,
            "sample_s": round(dt, 2),
            "shared": shared,
            "calls": rec.calls,
            "final": {k: (v.detach().float().cpu().half() if torch.is_tensor(v) else v)
                      for k, v in lat.items()},
        }
        path = OUT / f"trace_{idx:04d}.pt"
        torch.save(payload, path)

        times = [r["ms"] for r in rows if r["ms"] is not None]
        return {
            "index": idx, "image": img_path.name, "file": path.name,
            "mb": round(path.stat().st_size / 1e6, 1), "sample_s": round(dt, 2),
            "n_calls": len(rows),
            "call_ms_mean": round(sum(times) / len(times), 1) if times else None,
            "t_values": sorted({round(r["t"], 4) for r in rows if r["t"] is not None}),
            "distinct_feat": len({tuple(r["feat"]) for r in rows}),
            "args_shape": rows[0]["args"] if rows else None,
            "kwargs_shape": rows[0]["kwargs"] if rows else None,
            "out_shape": rows[0]["out"] if rows else None,
        }
    finally:
        rec.restore()


def run_list(items, tag=""):
    """Trace the given (index, path) pairs in this process."""
    out = []
    if not items:
        return out
    note(f"{tag}loading pipeline -> {DEV}")
    probe = probe_gpus()
    note(f"{tag}gpu seen here: {[d['name'] for d in probe.get('devices', [])]}")
    pipe, flow = build_pipeline()
    note(f"{tag}flow params {sum(p.numel() for p in flow.parameters()) / 1e6:.1f} M  "
         f"dtype {next(flow.parameters()).dtype}")
    for idx, path in items:
        try:
            row = trace_one(pipe, flow, idx, path)
        except Exception:  # noqa: BLE001
            note(f"{tag}[{idx}] FAILED {path.name}")
            note(tag + traceback.format_exc()[-1200:])
            continue
        out.append(row)
        note(f"{tag}[{idx}] {path.name}  {row['sample_s']}s  calls={row['n_calls']}  "
             f"call={row['call_ms_mean']}ms  branches={row['distinct_feat']}  "
             f"{row['mb']} MB")
        note(f"{tag}    args {row['args_shape']}")
        note(f"{tag}    out  {row['out_shape']}")
    return out


# ---------------------------------------------------------------------------
# two-GPU coordination
# ---------------------------------------------------------------------------

def script_path():
    p = globals().get("__file__")
    if p and Path(p).exists():
        return Path(p)
    # the kernel body exec()s this file's source, so __file__ is not defined;
    # it also leaves a copy on disk, which is what a child can re-run.
    cand = WORK / "fetched_script.py"
    return cand if cand.exists() else None


def fork_workers(n):
    """One child per GPU, each pinned with CUDA_VISIBLE_DEVICES.

    Spawned as processes rather than threads: torch's CUDA context does not
    survive fork(), and a separate process also gives each card its own Python
    heap, which matters on a 30 GB box holding two copies of the pipeline.
    """
    exe = script_path()
    if exe is None:
        note("cannot fork: no on-disk copy of this script")
        return []
    procs = []
    for i in range(n):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        env["FLIP_SHARD"] = f"{i}/{n}"
        env["FLIP_FORK"] = "0"                     # a child never forks again
        env.pop("FLIP_FORK_REASON", None)
        log = open(OUT / f"worker_{i}.log", "w")   # noqa: SIM115
        p = subprocess.Popen([sys.executable, str(exe)], env=env,
                             stdout=log, stderr=subprocess.STDOUT, text=True)
        procs.append((i, p, log))
        note(f"forked shard {i}/{n} -> pid {p.pid} on cuda:{i}")
    for i, p, log in procs:
        rc = p.wait()
        log.close()
        note(f"shard {i} exited rc={rc}")
        try:
            tail = (OUT / f"worker_{i}.log").read_text().splitlines()[-25:]
            for line in tail:
                note(f"  [gpu{i}] {line}")
        except Exception:  # noqa: BLE001
            pass
    return [i for i, p, _ in procs if p.returncode == 0]


def merge_parts(parts):
    images, notes, probes = [], [], {}
    for i in parts:
        f = OUT / f"trace_part{i}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        images.extend(d.get("images", []))
        notes.extend(d.get("notes", []))
        if d.get("probe"):
            probes[str(i)] = d["probe"]
    images.sort(key=lambda r: r["index"])
    return images, notes, probes


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    note("=" * 72)
    note("K2d teacher trace")
    note("=" * 72)

    probe = probe_gpus()
    R["probe"] = probe
    note(f"cuda devices {probe['count']}  visible={probe['visible']}  "
         f"bf16={probe.get('bf16')}")
    for d in probe.get("devices", []):
        note(f"  cuda:{d['i']}  {d['name']}  sm{d['sm']}  {d['mem_gb']} GB")
    note(f"  peer access {probe.get('p2p')}")
    note(f"  host RAM available {mem_available_gb()} GB")

    cands = pick_images()
    note(f"{len(cands)} candidate inputs (asked for {N_IMG}): "
         f"{[p.name for p in cands][:8]}")
    all_items = list(enumerate(cands))

    # ---- shard mode: this process is one worker -------------------------
    if SHARD:
        i, n = (int(x) for x in SHARD.split("/"))
        mine = all_items[i::n]
        note(f"shard {i}/{n}: {len(mine)} of {len(all_items)} images")
        rows = run_list(mine, tag=f"[s{i}] ")
        part = {"images": rows, "notes": R["notes"], "probe": probe}
        (OUT / f"trace_part{i}.json").write_text(json.dumps(part, indent=2))
        note(f"wrote trace_part{i}.json ({len(rows)} images)")
        return

    # ---- coordinator mode ----------------------------------------------
    started = time.time()
    rows = []
    gpus = probe["count"]
    free_gb = mem_available_gb()
    reason = None
    if FORK == "0":
        reason = "FLIP_FORK=0"
    elif gpus < 2:
        reason = f"only {gpus} gpu(s)"
    elif len(all_items) < 4:
        reason = f"{len(all_items)} images - not worth splitting"
    elif free_gb is not None and free_gb < FORK_MIN_GB:
        reason = f"{free_gb} GB RAM free < {FORK_MIN_GB} GB"

    if reason is None:
        note(f"splitting {len(all_items)} images across {gpus} gpus")
        ok = fork_workers(gpus)
        rows, child_notes, child_probes = merge_parts(ok)
        R["notes"].extend(child_notes)
        R["probes"] = child_probes
        done = {r["index"] for r in rows}
        missing = [(i, p) for i, p in all_items if i not in done]
        if missing:
            note(f"{len(missing)} images missing after the shards: "
                 f"{[i for i, _ in missing]} - re-running serially")
            rows.extend(run_list(missing, tag="[retry] "))
    else:
        note(f"single process ({reason})")
        rows = run_list(all_items)

    rows.sort(key=lambda r: r["index"])

    if rows:
        first = rows[0]
        note("=" * 72)
        note(f"calls per image   {first['n_calls']}   "
             f"({first['call_ms_mean']} ms each)")
        note(f"distinct t         {len(first['t_values'])}  {first['t_values'][:8]}")
        note(f"cond branches      {first['distinct_feat']}")
        note("if calls == 2 * (distinct t) the sampler runs CFG as two forwards,")
        note("and the student can fold guidance into a single velocity target.")

    R["images"] = rows
    R["wall_s"] = round(time.time() - started, 1)
    note(f"wall {R['wall_s']} s")
    (OUT / "trace.json").write_text(
        json.dumps({k: v for k, v in R.items() if k != "calls"}, indent=2))
    note(f"wrote {OUT / 'trace.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        note(traceback.format_exc())
        raise
