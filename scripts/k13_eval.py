"""k13 eval: decode the holdout only; gate t20 / t04 / g04-native / s04-eNNN.

k12_eval scanned every trace, which was fine for four photos and is not for
forty-eight: each decode is ~3 s and the student is only judged on the six
images it never saw. FLIP_EVAL_N picks the prefix of sorted traces, which is
the same set k2e held out (it holds out traces[:HOLDOUT]).
"""

import copy
import gc
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path

import torch

WORK = Path(os.environ.get("FLIP_WORK", "/kaggle/working"))
OUT = WORK / "k13_eval"
OUT.mkdir(parents=True, exist_ok=True)
TRACES = WORK / "k2d"
CKPT_DIR = WORK / "k2e"
N_GAUSS = 65536
EVAL_N = int(os.environ.get("FLIP_EVAL_N", "6"))

spec = importlib.util.spec_from_file_location("k2e", str(WORK / "k2e_distill.py"))
k2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k2e)

R = {"eval_n": EVAL_N, "notes": [], "rows": []}


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    R["notes"].append(s)


def dump():
    (OUT / "report.json").write_text(json.dumps(R, indent=2, default=float))


def gstats(g):
    # get_opacity/get_scaling are activated; the PLY stores their inverse
    # (logit/log), so the k4 gate's sigmoid/exp applies to the file, not here.
    op = g.get_opacity.detach().float().flatten()
    sc = g.get_scaling.detach().float()
    xyz = g.get_xyz.detach().float()
    d = (xyz - xyz.mean(0)).norm(dim=1)
    return {"n": int(xyz.shape[0]),
            "opacity": round(float(op.mean()), 4),
            "shell": round(float(((sc.amax(1) < 0.02) & (op > 0.1)).float().mean()), 4),
            "r95": round(float(torch.quantile(d, 0.95)), 4),
            "r50": round(float(torch.quantile(d, 0.5)), 4),
            "mean_scale": round(float(sc.amax(1).mean()), 5)}


def decode(pipe, lat, path):
    t0 = time.time()
    g = pipe.decode_latent(lat.half(), num_gaussians=N_GAUSS)
    s = gstats(g)
    g.save_ply(str(path))
    s.update(ply=path.name, mb=round(path.stat().st_size / 1e6, 2),
             decode_s=round(time.time() - t0, 2))
    del g
    return s


def cos_vs(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return {"cos": round(float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 4),
            "rel_l2": round(float((a - b).norm() / (b.norm() + 1e-12)), 4)}


def bucket_of(name):
    parts = Path(name).stem.split("_")
    return parts[1] if len(parts) > 1 else "?"


def main():
    files = sorted(TRACES.glob("trace_*.pt"))[:EVAL_N]
    grid = json.loads((CKPT_DIR / "grid.json").read_text())
    ckpts = sorted(CKPT_DIR.glob("student_e*.pt"))
    p("holdout", [f.name for f in files])
    p("grid", grid)
    p("checkpoints", [c.name for c in ckpts])
    if not ckpts:
        raise RuntimeError("no student_e*.pt to evaluate")

    pipe, flow = k2e.build_pipeline()
    dev = k2e.DEV
    for name in ("dinov3", "vae_encoder", "rmbg"):
        mod = getattr(pipe, name, None)
        if mod is not None:
            mod.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    p("front end on cpu:", round(torch.cuda.memory_allocated() / 1e9, 2), "GB live")

    for f in files:
        tr = k2e.load_trace(f)
        idx = int(f.stem.split("_")[-1])
        states, ts, final = k2e.trajectory(tr)
        cond = k2e.split_call(tr["calls"][0])[2]
        cond = {k: (v.to(dev).float() if torch.is_tensor(v) else v) for k, v in cond.items()}
        x0 = {k: v.to(dev).float() for k, v in states[0].items()}
        seed = int(tr.get("seed", 42 + idx))
        stem = Path(str(tr["image"])).stem
        row = {"image": tr["image"], "index": idx, "seed": seed,
               "bucket": bucket_of(tr["image"]),
               "t_first": round(ts[0], 2), "t_last": round(ts[-1], 2)}
        p(f"[{idx}] {tr['image']} seed {seed} states {len(states)} "
          f"t {ts[0]:.1f}..{ts[-1]:.1f}")

        lats = {"t20": {k: v.to(dev).float() for k, v in final.items()}}
        ref = lats["t20"]["latent"]

        pipe.flow_model = flow
        gen = torch.Generator(device=dev).manual_seed(seed)
        with torch.no_grad(), k2e.autocast():
            lats["t04"] = pipe.sample_latent(cond, steps=4, guidance_scale=3.0,
                                             shift=3.0, generator=gen)
        with torch.no_grad(), k2e.autocast():
            lats["g04"] = k2e.integrate(flow, x0, grid["t"], grid["dt"], cond)

        for ck in ckpts:
            tag = "s04_" + ck.stem.split("_")[-1]
            blob = torch.load(ck, map_location="cpu", weights_only=True)
            student = copy.deepcopy(flow).float()
            student.load_state_dict(blob["model"])
            student.eval()
            with torch.no_grad(), k2e.autocast():
                lats[tag] = k2e.integrate(student, x0, grid["t"], grid["dt"], cond)
            del student, blob
            gc.collect()
            torch.cuda.empty_cache()
            p(f"  integrated {tag}")

        for tag, lat in lats.items():
            s = decode(pipe, lat["latent"], OUT / f"{stem}_{tag}.ply")
            s["latent"] = cos_vs(lat["latent"], ref)
            row[tag] = s
            p(f"  {stem} {tag:9s} op {s['opacity']:.3f} shell {s['shell']:.3f} "
              f"r95 {s['r95']:.3f} lat {s['latent']}")
        rr = row["t20"]["r95"] or 1e-9
        for tag in lats:
            if tag == "t20":
                continue
            e = row[tag]
            e["r95_ratio"] = round(e["r95"] / rr, 3)
            e["gate"] = bool(e["opacity"] >= 0.60 and e["shell"] >= 0.70
                             and e["r95_ratio"] <= 1.15)
        R["rows"].append(row)
        dump()
        del lats
        gc.collect()
        torch.cuda.empty_cache()

    # --- summaries: overall and per view, because a student that only learned
    # fronts should not be able to hide behind a good average.
    tags = sorted({t for r in R["rows"] for t in r if t.startswith(("s04_", "t04", "g04"))})
    summary = {}
    for tag in tags:
        for group in ("all", "front", "side", "back"):
            rs = [r[tag] for r in R["rows"]
                  if tag in r and (group == "all" or r["bucket"] == group)]
            if not rs:
                continue
            key = tag if group == "all" else f"{tag}/{group}"
            summary[key] = {
                "n": len(rs),
                "opacity": round(sum(x["opacity"] for x in rs) / len(rs), 3),
                "shell": round(sum(x["shell"] for x in rs) / len(rs), 3),
                "r95_ratio": round(sum(x["r95_ratio"] for x in rs) / len(rs), 3),
                "gate": f"{sum(1 for x in rs if x['gate'])}/{len(rs)}"}
    R["summary"] = summary
    dump()
    p("summary\n" + json.dumps(summary, indent=2))


try:
    main()
except Exception:
    p(traceback.format_exc()[-3000:])
    dump()
    raise
