# TripoSplat, WebGPU edition

One photo in, 65,536 gaussians out, in the browser, under 60 s, with under 600 MB
of first load. Upstream [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)
is PyTorch + CUDA; this repo is the distillation, ONNX packing and JS runtime that
turn it into something `onnxruntime-web` can run on WebGPU — and the scripts that
measured every step of it.

## Where it stands (2026-11-05)

| piece | state | evidence |
|---|---|---|
| Front end: matte / DINO / VAE students | shipped | DINO cos 0.811, VAE cos 0.967, matte bce 0.363; ONNX 77.6 / 31.3 / 11.1 MB, parity 1.000000 |
| DiT, 2-step int4 | shipped | 294 MB `MultiHeadAttention` graph, session 978 ms; latent cos 0.9933 vs fixture |
| DiT, 4-step native grid (k13) | best checkpoint not yet packaged | holdout gate 5/6 vs teacher 4-step 4/6, naive grid 1/6 |
| Whole chain in a browser | 56.3 s on an M-series Mac | 1.5 load + 25.5 encode + 23.9 sample + 5.5 decode; 526 MB first load |
| QAT + ONNX export of the 4-step student | next | k13 `student_e001.pt` (1.48 GB) is in the Kaggle version output, not on HF yet |

The 56.3 s number is the **2-step int4** DiT plus the shipped encoders. The 4-step
student is better-looking but is not packaged or timed yet.

## Layout

- `scripts/k*.py` — the pipeline, numbered in the order it was built:
  `k2d` traces the teacher's 20-step trajectory, `k2e` distils it, `k3` trains the
  three front-end students, `k4` compares student vs teacher conditions,
  `k2v`–`k2z` fuse attention / pack / export / quantise the ONNX graphs.
  `k12_*` and `k13_*` are the run drivers, eval and rasteriser for the 4-step runs.
- `site/` — the browser runtime: ONNX Runtime Web + WebGPU, `pipeline.js` is the
  five-stage chain, `viewer.js` renders the gaussians.
- `webgpu/` — probe pages kept because they are the measurements (`budget.html`,
  `block0.html`, `i4mma.html`, and the per-graph tests under `onnx/`). Their
  `.onnx` / `.npy` fixtures are not in git.
- `assets/` — reports and contact sheets, one per experiment. PLYs are not in git.
- `dist/` — the model card for the published [HF repo](https://huggingface.co/idefinity/flip-student-2step).
- `data/man48.txt` — the 48-photo jandan manifest (URLs + view labels, no bytes).

## Weights (never in git)

- Teacher: `VAST-AI/TripoSplat` on Hugging Face, 4.46 GB, no token needed.
- Students and ONNX graphs: `idefinity/flip-student-2step`.
- Upstream source is pinned at commit `d8db9e0` by every driver.

## Environment

Python 3.10+; PyTorch with CUDA 12 on a T4 (sm75, fp16 + `GradScaler`), or ROCm on
the AMD W7900. `torch.cuda.is_available()` is true under ROCm, so the scripts run
unchanged; `FLIP_CKPT=0` drops gradient checkpointing on 48 GB. ONNX Runtime on
ROCm needs the ROCm build (`ROCMExecutionProvider` / `MIGraphXExecutionProvider`);
the plain `onnxruntime` wheel is CPU-only. Extra packages: `huggingface_hub`,
`safetensors`, `torchvision`, `tqdm`, `pillow`, `onnx`.

## Reproducing a run

```
FLIP_IMAGE_DIR=photos FLIP_TRACE_N=48 FLIP_TRACE_COND=teacher python3 scripts/k2d_teacher_trace.py
FLIP_TRACE_DIR=k2d FLIP_GRID=index FLIP_STEPS=4 FLIP_EPOCHS=2 FLIP_HOLDOUT=6 \
  FLIP_SAVE_EVERY=1 FLIP_CKPT=1 python3 scripts/k2e_distill.py
FLIP_EVAL_N=6 python3 scripts/k13_eval.py
```

The Kaggle drivers (`scripts/k12_kaggle_driver.py`, `scripts/k13_kaggle_driver.py`)
fetch the trace/distill scripts by SHA over two hosts and check their hashes, so a
run is identified by one commit. They stop on the first failed stage and keep the
log; there is no fallback path anywhere in this repo, by design.

## Results that define the work

k13, six holdout photos (two per view), gate = opacity >= 0.60, thin-shell >= 0.70,
r95 <= 1.15x the same photo's 20-step teacher:

| variant | gate | front / side / back |
|---|---|---|
| teacher, 4 steps | 4/6 | 2/2 · 2/2 · 0/2 |
| naive grid | 1/6 | 1/2 · 0/2 · 0/2 |
| student, epoch 1 | **5/6** | 2/2 · 2/2 · 1/2 |
| student, epoch 2 | 3/6 | 2/2 · 1/2 · 0/2 |

Epoch 2 overfits; epoch 1 is the checkpoint. Sheets and the full report:
`assets/k13/`.

## Docs

- `PUBLISH.md` — the release plan and the measured state of every stage (Chinese).
- `DISTILL.md` — why k9/k11 failed, what the literature does, the k12/k13 recipe.
- `RESUME.md` — the handoff: Kaggle API gotchas, contracts, browser runbook.
- `HANDOFF_k12_colab.md` — the original k12 grid derivation.
