"""Render the turbo student's 4 steps against the teacher's 20, on held-out photos.

The comparison has to start from the trace, not from a fresh sampler call. The
student was distilled on the state the teacher's sampler actually started from,
and that state only exists in the recorded first forward, so each eval photo is
traced by k2d (teacher cond, 20 steps) and then integrated from `states[0]` with
the frozen grid in grid.json.

Three latents are decoded for every photo, all through the same decoder and the
same PLY writer, so the two effects stay separable at the render:

  b04  the distilled student, 4 steps on the frozen grid
  g04  the teacher's own weights, 4 steps on that same grid
  t20  the teacher's endpoint, recorded in the trace

If b04 lands near t20 while g04 does not, the gain is the distillation and not
the grid. If g04 is already fine, the grid was the whole story.

env:
  FLIP_WORK      TripoSplat clone parent (default /kaggle/working)
  FLIP_K2E       k2e_distill.py on disk (default WORK/code/k2e_distill.py)
  FLIP_TRACE_DIR k2d output for the eval photos (default WORK/k2d_eval)
  FLIP_CKPT      student .pt written by k2e (default WORK/k2e/student_best.pt)
  FLIP_OUT       where the PLYs go (default WORK/k7b)
  FLIP_GAUSS     num_gaussians for the decoder (default 65536)
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
K2E = Path(os.environ.get("FLIP_K2E", WORK / "code" / "k2e_distill.py"))
TRACES = Path(os.environ.get("FLIP_TRACE_DIR", WORK / "k2d_eval"))
CKPT = Path(os.environ.get("FLIP_CKPT", WORK / "k2e" / "student_best.pt"))
OUT = Path(os.environ.get("FLIP_OUT", WORK / "k7b"))
N_GAUSS = int(os.environ.get("FLIP_GAUSS", "65536"))
OUT.mkdir(parents=True, exist_ok=True)

R = {"ckpt": str(CKPT), "traces": [], "notes": []}


def p(*a):
    line = " ".join(str(x) for x in a)
    print(f"[{time.strftime('%H:%M:%S')}]", line, flush=True)
    R["notes"].append(line)


def dump():
    (OUT / "report.json").write_text(json.dumps(R, indent=2, default=float))


def load_k2e():
    spec = importlib.util.spec_from_file_location("k2e", str(K2E))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stats(g):
    return {"n": int(g.get_xyz.shape[0]),
            "opacity": round(float(g.get_opacity.detach().float().mean()), 4),
            "scale": round(float(g.get_scaling.detach().float().mean()), 5),
            "x": [round(float(g.get_xyz[:, 0].min()), 3),
                  round(float(g.get_xyz[:, 0].max()), 3)]}


def main():
    k2e = load_k2e()
    files = sorted(TRACES.glob("trace_*.pt"))
    if not files:
        raise SystemExit(f"no traces in {TRACES}")
    p(f"{len(files)} eval traces, ckpt {CKPT.name}")

    blob = torch.load(CKPT, map_location="cpu", weights_only=True)
    grid = blob.get("grid") or json.loads((k2e.OUT / "grid.json").read_text())
    p("grid t", grid["t"], "dt", grid["dt"], "s_steps", blob.get("s_steps"))
    if int(blob.get("s_steps") or 0) != len(grid["t"]):
        raise SystemExit(f"ckpt says {blob.get('s_steps')} steps, grid has "
                         f"{len(grid['t'])} - the ckpt and grid.json are not the same run")

    pipe, teacher = k2e.build_pipeline()
    dev = k2e.DEV
    student = copy.deepcopy(teacher).float()
    student.load_state_dict(blob["model"])
    student.eval()
    p("student params", round(sum(q.numel() for q in student.parameters()) / 1e6, 2), "M")
    # the pipeline is only here to build and run the DiT; point it at the student
    # so decode_latent keeps its own front end and decoder
    teacher.to(dev)
    pipe.flow_model = student

    del blob
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    for f in files:
        row = {"trace": f.name}
        try:
            tr = k2e.load_trace(f)
            row["image"] = tr.get("image")
            row["cond"] = str(tr.get("cond"))
            states, ts, final = k2e.trajectory(tr)
            cond = k2e.split_call(tr["calls"][0])[2]
            cond = {k: (v.to(dev).float() if torch.is_tensor(v) else v)
                    for k, v in cond.items()}
            x0 = {k: v.to(dev).float() for k, v in states[0].items()}
            row["grid_match"] = (len(grid["t"]) == 4)

            lat = {}
            with torch.no_grad(), k2e.autocast():
                pipe.flow_model = student
                xs = k2e.integrate(student, x0, grid["t"], grid["dt"], cond)
                lat["b04"] = xs["latent"]
                # same start, same grid, the weights the grid was cut from
                xs_t = k2e.integrate(teacher, x0, grid["t"], grid["dt"], cond)
                lat["g04"] = xs_t["latent"]
            lat["t20"] = final["latent"].to(dev)

            ref = lat["t20"].float().flatten()
            for tag in ("b04", "g04", "t20"):
                q = lat[tag].float().flatten()
                t0 = time.time()
                # the decoder holds fp16 weights, and integrate hands back fp32
                g = pipe.decode_latent(lat[tag].half(), num_gaussians=N_GAUSS)
                ply = OUT / f"{Path(str(row['image'])).stem}_{tag}.ply"
                g.save_ply(str(ply))
                row[tag] = {**stats(g), "ply": ply.name,
                            "mb": round(ply.stat().st_size / 1e6, 2),
                            "decode_s": round(time.time() - t0, 2),
                            "latent_cos": round(float(
                                torch.nn.functional.cosine_similarity(q, ref, dim=0)), 4),
                            "latent_rel_l2": round(float(
                                (q - ref).norm() / (ref.norm() + 1e-12)), 4)}
                p(f"  {row['image']} {tag} op {row[tag]['opacity']} "
                  f"sc {row[tag]['scale']} cos {row[tag]['latent_cos']} "
                  f"rel {row[tag]['latent_rel_l2']}")
                del g
            del states, final, cond, x0, lat
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            row["error"] = traceback.format_exc()[-1500:]
            p(f.name, "FAILED", row["error"][-300:])
        R["traces"].append(row)
        dump()

    agg = {}
    for tag in ("b04", "g04", "t20"):
        rs = [r[tag] for r in R["traces"] if tag in r]
        if rs:
            agg[tag] = {"n": len(rs),
                        "opacity": round(sum(r["opacity"] for r in rs) / len(rs), 4),
                        "scale": round(sum(r["scale"] for r in rs) / len(rs), 5),
                        "latent_cos": round(sum(r["latent_cos"] for r in rs) / len(rs), 4),
                        "latent_rel_l2": round(sum(r["latent_rel_l2"] for r in rs) / len(rs), 4)}
    R["summary"] = agg
    dump()
    p("summary", json.dumps(agg))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        p(traceback.format_exc()[-2000:])
        dump()
        raise
