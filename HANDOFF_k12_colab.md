# k12 Colab test — handoff

Written 2026-11-01 for a fresh Qoder session. **Status: nothing has been
executed on Colab.** The driver scaffolding exists, no notebook cell has run,
no model has been downloaded.

## Mission

Test whether the k12 direction in [DISTILL.md](DISTILL.md) §6 is right, on
Colab, with a few epochs and a visual comparison the user can judge. The
question is narrow: **does training on the teacher's native 4-step grid remove
the soft-figure / blob failure of k11?**

- Same 370.53 M DiT, initialised from teacher weights.
- Teacher front end (DINOv3 ViT-H + Flux2 VAE + BiRefNet). Not optional.
- 4 DiT forwards at inference instead of 40.
- Minimal scale: 4 photos, 4–6 epochs, then decode PLYs and render
  `photo | t20 | t04 | g04-native | s04-native` into cell output.

## Why this grid (read DISTILL.md §3.1 before changing it)

k11 trained and sampled with `steps=4, shift=1`, whose σ boundaries are
1, .75, .5, .25, 0. That is **not** the teacher's native 4-step schedule. The
teacher's 20-step shift=3 states are σ = 1, .9, .75, .5, 0 at indices
**0/5/10/15/20**. k11's `pick_grid` in sigma mode selected the nearest recorded
states to the sigma-uniform grid and got indices **0/10/15/18/20**, i.e.
segments of 10/5/3/2 teacher steps. Its first student step was an averaged
velocity over half the trajectory — a low-pass filter. That is the blur.

The k12 grid is exact, no interpolation:

```
states s0 .. s20 from a 20-step shift=3 CFG=3 trace
grid   = s[0], s[5], s[10], s[15], s[20]
t      = 1.0, 0.9, 0.75, 0.5, 0.0
dt     = s - s_prev = 0.1, 0.15, 0.25, 0.5
target v_k = (s[5(k+1)] - s[5k]) / dt_k
```

Segment displacements telescope to the t20 endpoint, so the 4-step student can
hit the teacher's destination exactly.

## Training recipe for this test

1. **Trace**: `scripts/k2d_teacher_trace.py`, 20 steps, `guidance_scale=3`,
   `shift=3`, teacher cond. It records the CFG-deduped states (40 forwards).
2. **Pairs**: the 4 segments above, `GRID_MODE=index` semantics. Do **not**
   use sigma-nearest.
3. **Student**: `copy.deepcopy(teacher)`.
4. **Offline epochs 1–2**: L2 on teacher states.
5. **On-policy epochs 3–4** (if the offline run is not already conclusive):
   roll the student from x = s0 with no grad, keep visited states, then
   `loss = ||v_student - v_k||^2 + 0.25 * ||v_student - v_local||^2`
   where `v_local` is the teacher CFG velocity (2 forwards) at the visited
   state. **Check the sign convention first**: the sampler integrates
   `x = x - v*dt`; confirm `k2e_distill.py`'s `forward_velocity` convention
   before using teacher velocities as targets.
6. **Eval**, 4 photos, render everything:
   - `t20` teacher, 20 steps, CFG 3 (ceiling, decode only)
   - `t04` teacher, 4 steps, CFG 3 (the real bar: 8/12 gate passes)
   - `g04-native` teacher weights, native 4-step grid, no CFG (control)
   - `s04-native` student, native grid, no CFG (the candidate)
   The interesting comparison is `s04-native` vs `g04-native`: it isolates what
   training adds on the *correct* grid.
7. **Judge by render and gate, never by latent rel-L2.** Gate: mean opacity
   ≥ 0.60, thin-shell share ≥ 0.70 (max Gaussian scale < 2 cm), r95 ≤ 1.15×
   that photo's own t20. k11 proved endpoint rel-L2 ranks a blob above a figure.

## Scale and time on one T4

- Setup: clone TripoSplat `d8db9e0`, pip extras, snapshot `VAST-AI/TripoSplat`
  (4.2 GB) + three front-end checkpoints from HF — about 8–12 min.
- Trace 4 photos: ~4 min.
- Train 4–6 epochs × 16 pairs: ~5–8 min.
- Decode + render 4 variants × 4 photos: ~6–10 min.
- Total ~30–35 min after the download.

## Colab access

Route A — Qoder's own tools. The MCP server now finally exposes the 7 notebook
tools (`add_code_cell`, `add_text_cell`, `delete_cell`, `get_cells`,
`move_cell`, `run_code_cell`, `update_cell`). In a new session, call `get_cells`
first. If only `open_colab_browser_connection` is listed, call it, then the
seven should appear (that is what happened at the end of the broken session).
Do not use browser-use tools.

Route B — the stdio driver, proven on 2026-10-30. Qoder's registry never
surface the proxy tools then, so a file-driven client was used:

```
cd /tmp/colab_k12
nohup /Users/zziang/.cache/uv/archive-v0/waPfh-xYM8E_RsdJ/bin/python \
    client.py >> driver.out 2>&1 &
# wait for {"ev": "ready"} in out.log, then
/Users/zziang/.cache/uv/archive-v0/waPfh-xYM8E_RsdJ/bin/python \
    send.py call c1 open_colab_browser_connection '{}' 120
python3 show.py c1            # expect "true"
```

`open_colab_browser_connection` opens a Colab scratch tab; its frontend
connects to the driver's websocket. Only one frontend connection is allowed per
server. Then:

```
python3 send.py add A1 cell.py 0      # add_code_cell
python3 show.py A1                    # get the new cell id
python3 send.py run A2 <cellId> 1800  # run_code_cell, long timeout
python3 show.py A2                    # stdout / outputs
```

Caveat: the tab is a fresh runtime. **If `torch.cuda.is_available()` is False,
stop and ask the user to set Runtime → Change runtime type → T4 GPU.** Do not
attempt the trace on CPU.

Old working material from the k4 Colab run is still in `/private/tmp/colab_proxy/`:
`setup_cell.py` (verified download/install cell), `photos_cell.py` (12 input
photos embedded as a tar.gz, writes `/content/input_example`), `k4_cell.py`
(how the pipeline is loaded and driven), `send.py`/`show.py`. Reuse rather than
rewrite. The idle colab-mcp processes from Oct 30 can be left alone.

## Inputs that exist

- Repo: `VAST-AI-Research/TripoSplat` at `d8db9e0`.
- Teacher checkpoints: HF `VAST-AI/TripoSplat`, snapshot to
  `/content/TripoSplat/ckpts` (4.2 GB). Never set `HF_ENDPOINT` to a mirror for
  `snapshot_download`.
- Front end: public HF `idefinity/flip-student-2step` →
  `encoder/checkpoints/{dino,vae,matte}_model.pt` (the working `setup_cell.py`
  fetches exactly these).
- Distillation scripts: `scripts/k2d_teacher_trace.py`,
  `scripts/k2e_distill.py`. They are small; embed them in cells (base64 or text)
  rather than relying on signed URLs.

## Guardrails

- No Kaggle run without explicit approval; remaining quota is ~11.5 T4-hours
  until 2026-11-07. This test is Colab only.
- Never read, print, or use HF/Kaggle tokens from disk.
- Do not commit. Do not touch the browser directly.
- Save checkpoints and PLYs after every epoch; Colab disconnects.
- If the visual is still a soft figure, that is a real result: report it and
  stop rather than extending the run.

## Context sources

- Full plan and failure analysis: [DISTILL.md](DISTILL.md) — §1 ladder, §3.1
  grid, §6 k12 recipe, §8 traps.
- Run history: [RESUME.md](RESUME.md), k11 section.
- Raw transcript of the broken session:
  `/Users/zziang/.qoder-cn/projects/-Users-zziang-Documents-projects-flip/e1f499ae-deeb-4bd5-a253-f47d90d0c986.jsonl`

## Paste into the new session

> Continue the k12 Colab test. Read HANDOFF_k12_colab.md and DISTILL.md §3.1/§6
> first. Use the Colab MCP notebook tools (no browser-use, no Kaggle). Check
> GPU first, run the 4-photo native-grid test, and show me the
> photo/t20/t04/g04-native/s04-native render grid in cell output. The previous
> session's driver fallback is at /tmp/colab_k12 if the tools are not exposed.
