# RESUME — the encoder is finished; what is left is the eyeball

Last updated 2026-11-05. The matte finished at 20 epochs, the ONNX were re-exported and
republished, e2e ran, and the whole chain ran in a real browser at 56.3 s.
HF is still the source of truth: `idefinity/flip-student-2step`.

## k13 (2026-11-05): 48 photos, holdout, one epoch, 5/6

The k12 recipe (native index grid, teacher conditions, offline L2) on 48 balanced
jandan photos - 16 front / 16 side / 16 back, 42 train, 6 holdout (two per view,
the first six traces). Kaggle `kkonda/flip-k13` v3, kernel 142712057, one T4.
27 min/epoch at 12.25 GB peak with checkpointing; both epochs saved.

Gate on the six holdout photos (opacity >= 0.60, shell >= 0.70, r95 <= 1.15x t20):

| variant | opacity | shell | r95x | gate | F / S / B |
|---|---|---|---|---|---|
| t04 teacher | 0.779 | 0.861 | 1.056 | 4/6 | 2/2 - 2/2 - 0/2 |
| g04 naive grid | 0.361 | 0.413 | 1.111 | 1/6 | 1/2 - 0/2 - 0/2 |
| **s04_e001** | 0.786 | 0.899 | 1.020 | **5/6** | 2/2 - 2/2 - 1/2 |
| s04_e002 | 0.792 | 0.918 | 1.187 | 3/6 | 2/2 - 1/2 - 0/2 |

- e001 beats the teacher's own 4-step sampler on photos it never saw, and fixes the
  teacher's back-view collapse (p002_back opacity 0.795 vs t04's 0.544). Its one
  miss is p005_back on the outlier radius (1.244 vs 1.15), which t04 also misses
  (1.177).
- Epoch 2 overfits: gate 3/6 and latent cos vs teacher drops 0.50 -> 0.39. Ship
  `student_e001.pt`, not `student.pt`.
- This is still the offline-L2 recipe, not the distribution term in DISTILL.md §6.
  One seed, six photos: it is the first honest holdout number, not a converged recipe.
- Checkpoint `k2e/student_e001.pt` (1.48 GB) lives in the kernel version output and
  is not on HF yet. Next: QAT + ONNX export on the W7900 (48 GB removes the memory
  ceiling), then time the 4-step graph in the site.
- Sheets: `assets/k13/sheet_main.jpg` (all six holdout rows, prepared | t20 | t04 |
  g04 | e001 | e002) and `sheet_first.jpg` at 512.

## What this run closed

| step | result |
|---|---|
| Kaggle T4, six stages in one submission | 57 min total: restore → setup → prep → matte → export → e2e |
| matte 7 → 20 | acc 0.5299 → **0.6372** (bce 0.41 → 0.363), 13 × 74 s |
| export | 77,557,789 / 31,328,848 / 11,064,381 B, parity **1.000000** |
| e2e (teacher DiT + teacher decoder, student conditions) | lat cos **0.800–0.884**; DINO cos 0.797–0.859, VAE cos 0.953–0.978 |
| republish | `matte_student.onnx` sha256 `e2842c50…`; dino/vae byte-identical to what was live |
| browser, real encoder, `site/index.html?local=../w/&ort=/w/ort/` | **56.3 s** = 1.5 load + 25.5 encode + 23.9 sample + 5.5 decode; 65,536 gaussians |

The 60 s acceptance line is met with 3.7 s of slack on an M-series Mac.

## Where the quality actually stands

`lat_cos_full` 0.80–0.88 is the honest number. Two things show it is DINO, not VAE:
`lat_cos_dino` (DINO student's condition + teacher VAE) is within 0.01 of `lat_cos_full`,
and VAE feature cos is 0.95–0.98 while DINO is 0.81.

The browser screenshot is consistent with that: the splat renders as a dim, mushy cloud
rather than a figure. Nobody has yet put it side by side with the teacher's 20-step PLY,
which is the acceptance criterion that has not been checked.

## If you want more quality — in cost order

1. **Look at it first.** Render the student PLY next to the teacher's
   (`assets/k2g/building_stone_house_t20.ply`) and decide whether it is presentation
   (opacity bias, lighting, camera) or geometry. Cheap, and it decides everything below.
2. fp16 encoder (`keep_io_types=True`): ~60 MB instead of 120 MB, and it should shave
   part of the 25.5 s encode. No quality change.
3. Bigger DINO conv: `FLIP_DINO_DIM=768 FLIP_DINO_DEPTH=10/12`, 40 ep. Maybe 0.811 →
   0.85. ~1.5 h on T4, plus prep is already cached if you keep `/kaggle/working`.
4. Fine-tune the 2-step DiT on **student** conditions. This is the real fix: it removes
   the train/test domain gap instead of chasing the teacher's features. Bigger job,
   `scripts/k2e_distill.py` is the base.
5. ViT DINO student (was 0.707 at ep5 vs conv 0.563, 94 s/epoch). Untested to convergence.

## In flight (2026-10-31): the 4-step turbo DiT

The distilled front end is out of scope - `u20`/`s20` are unusable. What is being built is
a **turbo teacher**: the teacher architecture (370.53 M) and teacher encoders, distilled so
4 denoising steps land on the t20 latent.

**k9 (finished, ERROR at t=9,507 s).** `out/k9/k9_turbo_distill.py`, kernel
`kkonda/flip-k9-turbo-distill` v1 id 141878119. The hash gate passed (15,632 chars, byte
exact) and the run traced 31 of 36 photos, then trained 8 epochs on **76 pairs** and died
in the eval cell on `import k2e` - the artefact is placed as `k2e_distill.py`. Training
survived and its log is the verdict: holdout latent rel-l2 vs the t20 endpoint was
0.687-0.818 with the **teacher's own weights on the frozen 4-step grid** (that row is the
g04 control), 0.84 at epoch 7, 0.95-1.02 at epoch 8. The student never beat the weights it
started from: train loss fell 3x while holdout error rose, i.e. 76 pairs x 370 M params is
memorisation. 31 traces also cost the trace stage: shard 0 hit a transient datasets-server
error at 145 s, and the coordinator's in-parent retry at 82.6 s/image ran into my 3,000 s
cap.

**k11 (finished, rc 0, 31,910.6 s wall).** `out/k11/k11_turbo_distill.py`,
`out/k11/check_k11.py --bake`, kernel `kkonda/flip-k11-turbo-distill` id 141929953, 2xT4.
108 photos traced in 5,400 s -> 432 pairs, 384 trained (12 photos held out), 6 epochs,
train loss 0.2299 -> 0.1164 at ~69 min/epoch. Two cells existed because of how k9 failed: a
4-photo PILOT (gates at >=2 traces) and a REHEARSAL that ran the whole eval body - pipeline,
integrate, decode, save_ply, 1.48 GB checkpoint round trip - against teacher weights, so a
mistake cost 5 minutes instead of 10 hours.
**No checkpoint beat the control.** Latent rel-l2 / cos to the t20 endpoint, all 12 holdout
photos: student_best 0.8453/0.5842, e006 0.8560/0.5902, e004 0.8788/0.5686, **e002
0.8370/0.5827 (chosen)**, student.pt 0.8560/0.5902, and **g04 - teacher weights on the same
frozen grid - 0.7321/0.6776**. Usability gate (opacity >= 0.60, thin-shell >= 0.70,
r95 <= 1.15x that photo's own t20): t20 0.859/0.950/1.00, 12/12; g04 0.351/0.326/0.96,
0/12; b04 0.375/0.436/0.99, 0/12. The renders invert the ranking, though: b04 is a
recognisable but soft figure (hair, face, jacket stripes), g04 is a featureless blob - so
endpoint rel-l2 is not a usable selector in this regime. **Careful, the schedule is not
isolated**: t04 differs from g04 in two ways - native u-uniform sigma boundaries
(1/0.9/0.75/0.5/0) versus uniform (1/0.75/0.5/0.25/0), **and** guidance (CFG, 8 DiT
forwards, versus 4 with guidance folded into the student's target). So the 8/12 vs 0/12
gap cannot be attributed to the grid alone. What is established: distillation beat its own
init on the frozen path (g04 blob -> b04 figure; opacity 0.351 -> 0.375, shell 0.326 ->
0.436) and still missed the gate; five times k9's data and a 5x lower train loss did not
move the holdout latent gap, so more of the same is not the answer.
Recomputing k4's PLYs with the same gate gives the real 4-step sampler t04 mass 0.660 /
shell 0.769 / r95 1.04, 8/12 pass (t08 10/12, t02 4/12). Next measurement before any
retrain: a 12-photo 2x2 on teacher weights - uniform grid + CFG (8 forwards) and native
4-step grid + no CFG (4 forwards); g04 and t04 are the other two corners. That isolates
grid from guidance for ~20 min of GPU. If uniform+CFG recovers, the grid is fine and the
failure is in what the student was asked to learn; if it stays a blob, the next
distillation must follow t04's own 4-call trajectory, starting from weights that already
pass 8/12 on it. Outputs: `/tmp/flipview/k11/k11/*.ply` (36 PLYs),
`/tmp/flipview/k11/k11/{sweep,eval}.json`, page `k11-turbo.html`.

**The z-image reading (2026-11-01).** `DISTILL.md` has the literature review, the
failure analysis and the next recipe. The one-line diagnosis of k11 that was not
visible before: the student was trained and inferred on `steps=4, shift=1` (sigma
1/0.75/0.5/0.25/0), while the teacher's native 4-step grid (`t04`) is
`steps=4, shift=3` (sigma 1/0.9/0.75/0.5/0) - and the latter is *exactly* the
t20 states at indices 0/5/10/15/20, so the shift=3 segment averages are exact
with no interpolation. k11's `GRID_MODE=sigma` chose the shift=1 grid, whose own
teacher control (`g04`) is a blob. The next run (k12) is the same traces and the
same target construction on the native grid, plus on-policy student states and
gate-based selection; Z-Image-Turbo's Decoupled-DMD + DMDR is the stage after
that, and it costs ~100-1000x the remaining quota. Do not start k12 without
explicit approval; the budget is 41,547 GPU-s (~11.5 h) until 2026-11-07.

## Kaggle API facts that cost time

- A **new** kernel needs `slug` fully qualified: `"kkonda/flip-k9-turbo-distill"`. A bare
  slug is rejected with `Invalid slug: '...'` for every value, which looks like a naming
  rule and is not. Re-running an existing kernel takes `id` and the qualified slug.
- `machineShape: "NvidiaTeslaT4x2"` with `enableGpu: true` is accepted; script form is
  `language: "python"` + `kernelType: "script"` + `# %%` cells.
- `get_notebook_session_status` answers with `userName` + `kernelSlug` only; pass
  `versionLabel` and it says the version is not found, even for the version that is running.
- Nothing is downloadable until the session finishes: `download_notebook_output` mints a
  signed URL for any path it is asked for, including `script.log`, and that URL 404s while
  the run is live. So mid-run progress is only observable as `RUNNING` plus the eventual
  files - budget the kernel accordingly instead of watching it.
- Typing 15k characters of readable Python into `text` arrives intact far more often than
  typed base64 does; the hash gate is what makes a bad transcription cost 30 seconds instead
  of 3 hours of GPU. 23k works too (k11 v1: `chars 23313, ok true`).
- **`kaggleusercontent.com/kf/...` signed URLs expire.** The two artefact URLs minted from
  k9's output listing at completion were 403 by that evening, so k11 v1 died at t=1.3 s on
  `FETCH FAIL k2d HTTP 403`. Mint fresh ones with `download_notebook_output` per path
  (ownerSlug + kernelSlug + versionNumber + filePath) and push within the hour; verify them
  locally with curl + sha256 first, which this machine can still do for kaggleusercontent.

## Kaggle runbook (what was actually done)

The notebook is five lines: download `encoder/code/k3_kaggle_driver.py` from HF, run it.
The driver does restore → setup/prep → matte → export → e2e and prints everything.
Kernel: `kkonda/flip-k3-encoder`, version 3, T4, internet on, private.

Two things that cost time:

- **Do not set `HF_ENDPOINT=https://hf-mirror.com` on Kaggle.** The mirror drops
  `x-repo-commit`, `snapshot_download` fails with "Distant resource does not seem to be
  on huggingface.co". Kaggle reaches huggingface.co directly (0.2 s). The driver pops the
  variable for that reason; the mirror is only this Mac's problem.
- The notebook API wants a real `.ipynb` JSON for `text`, not a Python script. A script
  gets a `NotJSONError` from papermill and the run ends in 9 s.

## Local browser run (this Mac)

Neither huggingface.co nor cdn.jsdelivr.net is reachable, and hf-mirror answers
browser-shaped requests with an HTML challenge (no CORS headers → "Failed to fetch").
So the local run needs the bytes on disk:

```sh
# once: the 4 ONNX into w/, ONNX Runtime 1.31.1 dist into w/ort/
python3 -m http.server 8899 --bind 127.0.0.1
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9333 --no-sandbox --disable-gpu-sandbox \
  --enable-unsafe-webgpu --use-angle=metal --user-data-dir=/tmp/chrome-flip about:blank
node scripts/browser/cdp_run.mjs "http://127.0.0.1:8899/site/enc-test.html?local=../w/&ort=/w/ort/" 600000 9333
```

`--no-sandbox` is required (Chrome's sandbox cannot start here); `?ort=` is new, it points
ONNX Runtime at a local `dist/` instead of jsDelivr. `site/enc-test.html` now prints the
per-stage split and the splat stats, which is where the 56.3 s breakdown comes from.

## Contracts (unchanged, still gotchas)

- features: `feature1 [1,4101,1280]` (DINO, LN over 1280), `feature2 [1,4096,128]`
  (VAE mean, BN-standardised). The DiT ONNX pads 5 zero tokens internally; do not pre-pad.
- inputs: DINO `rgb/255`; VAE `rgb*2-1`; matte `rgb/255`, sigmoid output, dynamic h/w (/32).
- preprocessing: shortest side 1024, round /32; alpha → erode 1 px → 1.2× square crop →
  black pad 1024. Python `make_sample` is the reference; JS `loadCondition` mirrors it.
- `ENCODER_FILES` bytes in `site/src/pipeline.js` are still correct (nothing changed size).
- Never `scripts/hf_upload.py` (dead endpoint); use `HfApi.upload_file`.

## Remaining checklist

- [x] Kaggle setup + 4.46 GB teacher ckpts
- [x] restore inputs/checkpoints; rebuild cache (33 min @ 9.83 s/img)
- [x] matte 7→20
- [x] export (parity 1.000000) + e2e (`encoder/e2e.json`)
- [x] overwrite HF root ONNX + meta; verify 200/size/sha
- [x] update `PUBLISH.md` status/budget/e2e; `site/README.md` numbers
- [x] `ENCODER_FILES` bytes — checked, unchanged
- [x] browser end-to-end + timing (56.3 s, real encoder)
- [x] dev pages kept (`site/enc-test.html` with stage split, `site/ui-test.html`)
- [ ] visual acceptance: student PLY vs teacher 20-step PLY, side by side
- [ ] fp16 encoder (~60 MB) if the 120 MB lazy load ever bothers anyone
- [ ] do not commit unless asked; `key.secret` stays ignored
