"""K2l - two questions that decide whether a new dataset is worth anything.

Question 1: can this box actually pull images out of a HuggingFace dataset?
The last time we attached a Kaggle dataset through the API it was silently
ignored (/kaggle/input existed and was empty), so the run that was supposed to
trace 64 COCO images traced the same four repo examples again. That is why the
pipeline downloads photos over HTTP. If HF's dataset server works from here
then any image dataset becomes usable with one env var, and the question stops
being "which dataset can we get" and becomes "which dataset is right".

Question 2 - the one that matters: does the camera channel depend on the image?

If the camera path is (nearly) the same for every input, then it is a function
of the schedule, not of the content. It could then be fitted analytically and
dropped from the student's job entirely, which removes the exact thing that is
broken right now: K2j measured the student's camera rollout at 0.61-0.98 rel-l2
against the teacher's own 4-step 0.20-0.41. More data would not fix that, and
neither would a better dataset.

If the camera path does move with the image, then data choice matters and we
should pick images where the teacher is confident - which is a selection
criterion, not a dataset.

The test is deliberately cheap and refuses to need ground truth: sample N
images, record the camera output at every teacher call, and compare the spread
*between* images at a fixed step to the spread *within* one image's path. The
ratio is the whole answer.

Env:
  FLIP_PROBE_N      images to sample for the camera test (default 8)
  FLIP_PROBE_HF     "1" to run the HuggingFace part (default 1)
  FLIP_HF_DATASET   dataset repo to test the sampler on (default
                    DownFlow/fuliji, which the user asked about)
  FLIP_HF_ROWS      how many rows to pull (default 8)
  FLIP_HF_QUERIES   ";"-separated HF search terms, used to find real candidate
                    dataset names instead of guessing them

Writes k2l/report.json
"""

import json
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from PIL import Image

WORK = Path("/kaggle/working")
REPO = WORK / "TripoSplat"
CKPT = REPO / "ckpts"
OUT = WORK / "k2l"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

N_IMG = int(os.environ.get("FLIP_PROBE_N", "8"))
DO_HF = os.environ.get("FLIP_PROBE_HF", "1").strip() != "0"
HF_DATASET = os.environ.get("FLIP_HF_DATASET", "DownFlow/fuliji").strip()
HF_ROWS = int(os.environ.get("FLIP_HF_ROWS", "8"))
HF_SPLIT = os.environ.get("FLIP_HF_SPLIT", "train").strip()
HF_CONFIG = os.environ.get("FLIP_HF_CONFIG", "").strip()
HF_QUERIES = os.environ.get(
    "FLIP_HF_QUERIES",
    "scanned objects;objaverse;3d object render;product white background",
).split(";")

STEPS = 20
GAIN = 3.0
SHIFT = 3.0

R = {"notes": [], "hf": {}, "camera": {}}


def note(msg):
    line = str(msg)
    print(line, flush=True)
    R["notes"].append(line)


def save():
    (OUT / "report.json").write_text(json.dumps(R, indent=1))


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "flip-probe"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# question 1: can we read an HF dataset from here
# ---------------------------------------------------------------------------

def hf_search(term, limit=12):
    """Use HF's own search so we report dataset names that exist.

    Guessing repo names wastes a kernel run per guess; this costs one request.
    """
    url = ("https://huggingface.co/api/datasets?"
           + urllib.parse.urlencode({"search": term, "limit": limit,
                                     "sort": "downloads", "direction": "-1"}))
    try:
        hits = json.loads(_get(url, timeout=30))
    except Exception:  # noqa: BLE001
        return {"term": term, "error": traceback.format_exc()[-300:]}
    keep = []
    for h in hits:
        keep.append({
            "id": h.get("id"),
            "downloads": h.get("downloads"),
            "tags": [t for t in (h.get("tags") or []) if t in
                     ("image", "text-to-image", "image-classification",
                      "3d", "object-detection")],
        })
    return {"term": term, "hits": keep}


def hf_splits(dataset):
    """datasets-server wants a config, and config names are not guessable."""
    url = ("https://datasets-server.huggingface.co/splits?"
           + urllib.parse.urlencode({"dataset": dataset}))
    js = json.loads(_get(url, timeout=45))
    return js.get("splits") or []


def hf_rows(dataset, offset=0, length=8, config=None, split=None):
    """Pull rows through datasets-server.

    This is the endpoint that makes a 201 MB parquet unnecessary: it returns
    per-row image URLs, so a training run can sample 200 images from a 50 GB
    dataset without downloading it.
    """
    q = {"dataset": dataset, "split": split or HF_SPLIT, "offset": offset,
         "length": length}
    cfg = config or HF_CONFIG
    if cfg:
        q["config"] = cfg
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(q)
    return json.loads(_get(url, timeout=60))


def hf_probe():
    out = {"dataset": HF_DATASET, "queries": []}
    for term in HF_QUERIES:
        if term.strip():
            out["queries"].append(hf_search(term.strip()))
    save()

    t0 = time.time()
    cfg, split = HF_CONFIG or None, HF_SPLIT
    try:
        parts = hf_splits(HF_DATASET)
        out["splits"] = parts
        if parts:
            first = parts[0]
            cfg = cfg or first.get("config")
            split = split or first.get("split")
        note(f"hf splits: {[(p.get('config'), p.get('split')) for p in parts]}")
    except Exception:  # noqa: BLE001
        out["splits_error"] = traceback.format_exc()[-500:]
        note("hf splits lookup failed - trying the defaults")
    try:
        js = hf_rows(HF_DATASET, 0, min(HF_ROWS, 100), cfg, split)
    except Exception:  # noqa: BLE001
        out["rows_error"] = traceback.format_exc()[-900:]
        note(f"hf rows FAILED for {HF_DATASET}")
        R["hf"] = out
        return out
    rows = js.get("rows", [])
    out["row_count"] = len(rows)
    out["total_rows"] = (js.get("num_rows_total")
                         or (js.get("num_rows") or {}).get("train"))
    out["columns"] = [f.get("name") for f in js.get("features", []) or []]
    note(f"hf rows ok: {len(rows)} rows from {HF_DATASET} "
         f"({out['total_rows']} total, cols {out['columns']})")

    imgs = []
    for r in rows[:HF_ROWS]:
        row = r.get("row") or {}
        cell = row.get("image")
        src = cell.get("src") if isinstance(cell, dict) else None
        if not src:
            continue
        try:
            blob = _get(src, timeout=60)
            p = OUT / "inputs" / f"hf_{len(imgs):03d}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
            with Image.open(p) as im:
                im.load()
                imgs.append({"file": p.name, "bytes": len(blob),
                             "size": list(im.size), "mode": im.mode})
        except Exception:  # noqa: BLE001
            imgs.append({"src": src[:80], "error": traceback.format_exc()[-200:]})
    out["images"] = imgs
    out["ok"] = [i for i in imgs if "size" in i]
    note(f"  downloaded {len(out['ok'])}/{len(rows)} images "
         f"in {time.time()-t0:.1f}s")
    R["hf"] = out
    return out


# ---------------------------------------------------------------------------
# question 2: is the camera path a function of the image
# ---------------------------------------------------------------------------

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

    dino_p, vae_p = pick("dino"), pick("flux")
    dec_p = pick("decoder") or pick("octree")
    rmbg_p = pick("biref") or pick("rmbg")
    taken = {p for p in (dino_p, vae_p, rmbg_p, dec_p) if p}
    ck = [f for f in files if f not in taken][0]
    missing = [k for k, v in (("dino", dino_p), ("flux", vae_p),
                              ("decoder", dec_p), ("rmbg", rmbg_p)) if v is None]
    if missing:
        raise RuntimeError(f"missing ckpt parts {missing}")
    return T.TripoSplatPipeline(
        ckpt_path=str(ck), decoder_path=str(dec_p), dinov3_path=str(dino_p),
        flux2_vae_encoder_path=str(vae_p), rmbg_path=str(rmbg_p), device=str(DEV))


def as_map(state):
    if torch.is_tensor(state):
        return {"latent": state}
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


class CamRec:
    """Record the camera half of every flow call, and nothing else.

    Latent is 8192x16 - keeping it would make this a second copy of the trace
    run. Camera is 5 numbers. The whole test fits in a few kilobytes.
    """

    def __init__(self, mod):
        self.mod = mod
        self.orig = None
        self.calls = []

    def __enter__(self):
        self.orig = self.mod.forward
        rec = self

        def spy(state, t, cond, *a, **k):
            out = rec.orig(state, t, cond, *a, **k)
            m = as_map(out)
            cam = m.get("camera")
            if cam is not None:
                rec.calls.append({
                    "t": float(torch.as_tensor(t).flatten()[0].float()),
                    "camera": [round(float(x), 6)
                               for x in cam.detach().float().flatten().cpu()],
                })
            return out

        self.mod.forward = spy
        return rec

    def __exit__(self, *exc):
        self.mod.forward = self.orig
        return False


def pick_images():
    """Two groups on purpose.

    Group A: the repo's four examples - four different objects, the content
    class the model was built for. Group B: the downloaded dataset images - one
    narrow class. Comparing the two is what makes the camera test mean
    something: if the camera path is the same across groups as it is within a
    group, the image genuinely does not steer it. Testing on one group alone
    cannot tell "the image does not matter" from "these images are alike".
    """
    exts = (".png", ".jpg", ".jpeg", ".webp")
    # the upstream examples live in static/example_inputs, not examples/ -
    # getting this wrong silently empties the control group, and a camera test
    # with only one content class cannot distinguish "the image does not
    # matter" from "these images are alike"
    ex = REPO / "static" / "example_inputs"
    grp_a = [p for p in sorted(ex.glob("*")) if p.suffix.lower() in exts] \
        if ex.is_dir() else []
    if not grp_a and (REPO / "static").is_dir():
        grp_a = [p for p in sorted((REPO / "static").rglob("*"))
                 if p.suffix.lower() in exts]
    in_dir = OUT / "inputs"
    grp_b = [p for p in sorted(in_dir.glob("*")) if p.suffix.lower() in exts] \
        if in_dir.exists() else []
    half = max(1, N_IMG // 2)
    out = [("objects", p) for p in grp_a[:half]]
    out += [("dataset", p) for p in grp_b[:N_IMG - len(out)]]
    return out


def dist(a, b):
    """Mean relative distance between two camera vectors."""
    va, vb = torch.tensor(a), torch.tensor(b)
    denom = max(float(torch.cat([va.abs(), vb.abs()]).max()), 1e-8)
    return float((va - vb).abs().mean() / denom)


def analyse_camera(items, pipe):
    per_image = []
    with torch.inference_mode():
        for group, p in items:
            try:
                prepared = pipe.preprocess_image(Image.open(p).convert("RGB"))
                cond = pipe.encode_image(prepared)
                gen = torch.Generator(device=DEV).manual_seed(SEED)
                with CamRec(pipe.flow_model) as rec:
                    pipe.sample_latent(cond, steps=STEPS, guidance_scale=GAIN,
                                       shift=SHIFT, generator=gen)
                # CFG gives two calls per step; combine them the way the sampler
                # does. K2j verified the order (3a-2b) by checking that one
                # Euler step reproduces the teacher's next state.
                by_t = {}
                for c in rec.calls:
                    by_t.setdefault(c["t"], []).append(c["camera"])
                path = []
                for t in sorted(by_t, reverse=True):
                    pair = by_t[t]
                    if len(pair) >= 2:
                        v = [GAIN * a - (GAIN - 1) * b
                             for a, b in zip(pair[0], pair[1])]
                    else:
                        v = pair[0]
                    path.append({"t": round(t, 3), "camera": [round(x, 6) for x in v]})
                per_image.append({"image": p.name, "group": group,
                                  "steps": len(path), "path": path})
                note(f"  [{group}] {p.name}: {len(path)} steps, "
                     f"final camera {path[-1]['camera'] if path else '--'}")
            except Exception:  # noqa: BLE001
                per_image.append({"image": p.name, "group": group,
                                  "error": traceback.format_exc()[-700:]})
                note(f"  [{group}] {p.name} FAILED: {traceback.format_exc()[-200:]}")
            save()
            torch.cuda.empty_cache()
    return per_image


def _mean_pairs(groups):
    """Mean pairwise relative distance between all vectors in the given groups."""
    flat = [v for g in groups for v in g]
    if len(flat) < 2:
        return None
    ds = [dist(flat[i], flat[j])
          for i in range(len(flat)) for j in range(i + 1, len(flat))]
    return sum(ds) / len(ds)


def camera_verdict(per_image):
    ok = [r for r in per_image if "path" in r and r["path"]]
    if len(ok) < 2:
        return {"call": "not enough images", "n": len(ok)}

    n_steps = min(len(r["path"]) for r in ok)
    ok = [dict(r, path=r["path"][:n_steps]) for r in ok]

    names = sorted({r["group"] for r in ok})
    # between-image spread at each step, split by whether the two images are
    # from the same content class or different ones
    same_step, cross_step = [], []
    for s in range(n_steps):
        vec = {r["image"]: r["path"][s]["camera"] for r in ok}
        same, cross = [], []
        for i, a in enumerate(ok):
            for b in ok[i + 1:]:
                d = dist(vec[a["image"]], vec[b["image"]])
                (same if a["group"] == b["group"] else cross).append(d)
        same_step.append(sum(same) / len(same) if same else None)
        cross_step.append(sum(cross) / len(cross) if cross else None)

    def clean(xs):
        return [x for x in xs if x is not None]

    within = {}
    for g in names:
        rows = [r for r in ok if r["group"] == g]
        start = rows[0]["path"][0]["camera"]
        within[g] = round(max(dist(start, r["path"][s]["camera"])
                              for r in rows for s in range(n_steps)), 5)

    same = sum(clean(same_step)) / max(len(clean(same_step)), 1)
    cross = (sum(clean(cross_step)) / len(clean(cross_step))
             if clean(cross_step) else None)
    travel = sum(within.values()) / len(within)

    out = {
        "n_images": len(ok), "n_steps": n_steps, "groups": names,
        "same_group_mean": round(same, 5),
        "cross_group_mean": None if cross is None else round(cross, 5),
        "within_path_travel": {g: within[g] for g in names},
        "within_path_travel_mean": round(travel, 5),
        "same_per_step": [None if x is None else round(x, 5) for x in same_step],
        "cross_per_step": [None if x is None else round(x, 5) for x in cross_step],
        "start_cameras": {r["image"]: r["path"][0]["camera"] for r in ok},
        "end_cameras": {r["image"]: r["path"][-1]["camera"] for r in ok},
    }
    if travel < 1e-6:
        out["call"] = "camera never moves - it is a constant, drop it entirely"
        note(out["call"])
        return out

    ratio_same = same / travel
    out["ratio_same_over_travel"] = round(ratio_same, 4)
    if cross is not None:
        out["ratio_cross_over_same"] = round(cross / max(same, 1e-9), 3)

    note("")
    note(f"camera spread between images, same class  {same:.5f}")
    if cross is not None:
        note(f"camera spread between images, other class {cross:.5f}")
    note(f"how far one image's own path travels      {travel:.5f}")
    note(f"ratio (same-class spread / travel)        {ratio_same:.4f}")
    if cross is not None:
        note(f"ratio (cross-class / same-class)          {out['ratio_cross_over_same']}")

    # two numbers decide it: the path has to move, and it has to move with the
    # image. If images sit on the same line, the path is the schedule's.
    if ratio_same < 0.15:
        out["call"] = ("camera path is essentially the same for every image - "
                       "fit it analytically, take it out of the student's task")
    elif cross is not None and out["ratio_cross_over_same"] < 1.5:
        out["call"] = ("camera varies image to image, but not by content class - "
                       "a shared path plus a small per-image correction fits")
    else:
        out["call"] = ("camera moves with content - the dataset choice matters; "
                       "pick images where the teacher is confident")
    note(out["call"])
    return out


if __name__ == "__main__":
    note(f"K2l probe | DEV={DEV} | pid={os.getpid()}")
    if DO_HF:
        try:
            hf_probe()
        except Exception:  # noqa: BLE001
            note("hf probe crashed")
            note(traceback.format_exc()[-1200:])
            R["hf"] = {"error": traceback.format_exc()[-1200:]}
    save()

    try:
        pipe = build_pipeline()
        items = pick_images()
        note(f"camera test on {len(items)} images: "
             f"{[(g, p.name) for g, p in items]}")
        R["camera"]["images"] = analyse_camera(items, pipe)
        R["camera"].update(camera_verdict(R["camera"]["images"]))
    except Exception:  # noqa: BLE001
        note("camera test crashed")
        note(traceback.format_exc()[-1500:])
        R["camera"]["error"] = traceback.format_exc()[-1500:]
    save()
    note("k2l done")
