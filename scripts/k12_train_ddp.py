"""k12 DDP: k2e's loop on two T4s, one rank per GPU (NCCL allreduce)."""

import copy
import gc
import importlib.util
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

WORK = Path("/kaggle/working")
os.environ.setdefault("FLIP_GRID", "index")
os.environ.setdefault("FLIP_EPOCHS", "6")
os.environ.setdefault("FLIP_HOLDOUT", "2")
os.environ.setdefault("FLIP_STEPS", "4")
os.environ.setdefault("FLIP_TRACE_DIR", str(WORK / "k2d"))

spec = importlib.util.spec_from_file_location("k2e", str(WORK / "k2e_distill.py"))
k2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k2e)

KEEP = {2, 4, 6}


def main():
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    rank, world = dist.get_rank(), dist.get_world_size()
    k2e.DEV = torch.device(f"cuda:{rank}")
    note = k2e.note
    note(f"rank {rank}/{world} {torch.cuda.get_device_name(rank)}")

    files = sorted(k2e.TRACE_DIR.glob("trace_*.pt"))
    traces = [k2e.load_trace(f) for f in files]
    note(f"{len(traces)} traces {[t['image'] for t in traces]}")

    fit = k2e.fit_dt_scale(traces[0]) or k2e.fit_dt_scale(traces[-1])
    if fit is None:
        k2e.DT_SCALE = 1.0
    else:
        name, med, spread = fit
        k2e.DT_SCALE = med
        note(f"dt scale {med:.6g} ({name}, spread {spread:.3f})")

    pairs, _ = k2e.build_pairs(traces)
    held = {tr["image"] for tr in traces[:k2e.HOLDOUT]}
    pairs = [p for p in pairs if p["image"] not in held]
    note(f"{len(pairs)} pairs, held out {sorted(held)}")
    if len(pairs) < world:
        raise RuntimeError(f"{len(pairs)} pairs cannot feed {world} ranks")

    dt_by_k, t_by_k = {}, {}
    for p in pairs:
        dt_by_k.setdefault(p["k"], []).append(p["dt"])
        t_by_k.setdefault(p["k"], []).append(p["t"])
    grid_json = {
        "student_steps": k2e.S_STEPS,
        "t": [round(sum(t_by_k[k]) / len(t_by_k[k]), 4) for k in sorted(t_by_k)],
        "dt": [round(sum(dt_by_k[k]) / len(dt_by_k[k]), 6) for k in sorted(dt_by_k)],
        "note": "frozen: native 4-step grid, k12 (t20 indices 0/5/10/15/20)",
    }
    note(f"grid t {grid_json['t']} dt {grid_json['dt']}")
    if rank == 0:
        (k2e.OUT / "grid.json").write_text(json.dumps(grid_json, indent=2))

    pipe, teacher = k2e.build_pipeline()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    student = copy.deepcopy(teacher).float()
    teacher.to("cpu")
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    student.train()
    if k2e.GRAD_CKPT:
        note(f"ckpt {k2e.enable_ckpt(student)}")
    student = torch.nn.parallel.DistributedDataParallel(
        student, device_ids=[rank], output_device=rank,
        find_unused_parameters=True)
    opt = torch.optim.AdamW(student.parameters(), lr=k2e.LR,
                            betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler(device="cuda", enabled=True)

    def dump(tag):
        if rank == 0:
            torch.save({"model": student.module.state_dict(), "grid": grid_json,
                        "s_steps": k2e.S_STEPS}, k2e.OUT / tag)
            note(f"  saved {tag}")

    def eval_shard(tag):
        student.eval()
        with torch.no_grad():
            k2e.evaluate(student, traces[rank::world], grid_json, tag)
        student.train()

    eval_shard("before")
    dist.barrier()

    t0 = time.time()
    for epoch in range(k2e.EPOCHS):
        gen = torch.Generator().manual_seed(1000 + epoch)
        perm = torch.randperm(len(pairs), generator=gen).tolist()
        n = len(pairs) // world
        mine = perm[: n * world][rank::world]
        run_loss, run_n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for idx in mine:
            p = pairs[idx]
            state = k2e.to_dev(p["state"])
            tgt = {k: v.to(k2e.DEV) for k, v in p["target"].items()}
            cond = {k: (v.to(k2e.DEV).float() if torch.is_tensor(v) else v)
                    for k, v in p["cond"].items()}
            with k2e.autocast():
                v = k2e.forward_velocity(student, state, p["t"], cond)
                terms = {k: torch.nn.functional.mse_loss(v[k].float(), tgt[k])
                         for k in tgt if k in v}
                loss = sum(terms[k] * (k2e.W_CAM if k == "camera" else 1.0)
                           for k in terms)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            run_loss += loss.item()
            run_n += 1
        stat = torch.tensor([run_loss, float(run_n)], device=k2e.DEV)
        dist.all_reduce(stat)
        mean = stat[0].item() / max(stat[1].item(), 1.0)
        note(f"epoch {epoch:3d}  loss {mean:.6f}  "
             f"{(time.time() - t0) / 60:.1f} min  world {world}")
        if rank == 0:
            k2e.R["training"].append({"epoch": epoch, "loss": mean})
            k2e.save()
        if (epoch + 1) in KEEP:
            eval_shard(f"e{epoch + 1}")
            dump(f"student_e{epoch + 1:03d}.pt")
        dist.barrier()

    eval_shard("final")
    if rank == 0:
        k2e.save()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
