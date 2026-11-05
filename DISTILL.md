# Few-step distillation of the TripoSplat DiT

What we tried, why it failed, what the literature actually does, and the one
experiment worth running next. Written 2026-11-01, after k11.

The goal has not changed: **the same 370.53 M DiT, teacher-initialised, teacher
front end, 4 DiT forwards instead of 40, at t20 quality.** No model-size
reduction. This document is about the training recipe, not the architecture.

## The short answer

Z-Image-Turbo is not "a student trained to copy the teacher's output." It is a
teacher-initialised, same-size model trained on **its own samples** against the
teacher's **distribution**, with CFG folded in and guidance *augmented* during
training (Decoupled-DMD), then pushed with RL (DMDR). The practice that ships:
teacher init; fold guidance; on-policy states; CFG augmentation as the quality
engine; distribution matching or an adversarial term as the regulariser; then a
reward stage. The practice that fails: offline L2 regression onto teacher states
on a frozen grid, with a smoothed average-velocity target and a latent-space
selector. k9 and k11 did the second thing almost exactly, and produced a soft
figure and a blob. k12 corrects the two mechanical errors (the grid and the
off-policy states); k13 adds the distribution term that the literature says is
what actually restores detail.

---

## 1. The quality ladder we are trying to move

All numbers below are the usability gate on rendered PLYs: mean opacity >= 0.60,
thin-shell share >= 0.70 (max Gaussian scale < 2 cm), r95 <= 1.15x that photo's
own t20. The gate was set before any of these runs and has not been tuned since.

| tag | what runs | steps | CFG | DiT forwards | opacity | shell | r95/t20 | pass |
|---|---|---|---|---|---|---|---|---|
| t20 | teacher, stock sampler | 20 | 3 | 40 | 0.858 | 0.941 | 1.00 | **12/12** |
| t08 | teacher, stock sampler | 8 | 3 | 16 | 0.785 | 0.883 | 1.03 | 10/12 |
| t04 | teacher, stock sampler | 4 | 3 | 8 | 0.660 | 0.769 | 1.04 | **8/12** |
| t02 | teacher, stock sampler | 2 | 3 | 4 | 0.488 | 0.608 | 1.03 | 4/12 |
| g04 | teacher weights, student grid (shift=1), no CFG | 4 | 1 | 4 | 0.351 | 0.326 | 0.96 | 0/12 |
| b04 | k11 student, student grid (shift=1) | 4 | folded | 4 | 0.375 | 0.436 | 0.99 | 0/12 |
| u20 | **student front end** + teacher DiT | 20 | 3 | 40 | 0.280 | 0.404 | 1.12 | 0/4 |
| u04 | **student front end** + teacher DiT | 4 | 3 | 8 | 0.280 | 0.377 | 1.10 | 0/4 |
| s20 | old student front end + old student DiT | 20 | 3 | 40 | 0.445 | 0.708 | 0.97 | 1/12 |
| s02 | old student front end + old student DiT | 2 | 1 | 2 | 0.182 | 0.223 | 0.90 | 0/12 |

Read the first four rows as the teacher's quality/speed curve: the stock 4-step
sampler is already usable on 8 of 12 photos. That is the bar a 4-forward student
has to clear, and it is the bar k11's b04 did not clear (0/12).

`u20`/`u04` are not about distillation at all. They are the old student front
end feeding the teacher's DiT, and they are unusable. The teacher front end
(DINOv3 ViT-H + Flux2 VAE) is not optional; every run after k4 uses it.

---

## 2. What we tried, and what it cost

### k4 — the corner matrix (done, 12 photos, ~4 h)

The full variant matrix, teacher and student front ends, 2xT4 sharded. The
useful part is the teacher's own row: 20/8/4/2 steps with CFG gives 12/12,
10/12, 8/12, 4/12. The old 2-step student (`s02`) was 0/12 and the old front end
with the teacher's DiT (`u20`) was 0/4. This closed the "shrink the model"
direction for good: step count is the only axis left.

### k7b — the eval harness (done)

Turns a checkpoint into PLYs and a gate table. Its lesson is procedural and
still binding: **a checkpoint is judged by decoded Gaussians, never by latent
distance.** k11 later proved why.

### k9 — 76 pairs, 8 epochs (ERROR at 9,507 s)

31 of 36 photos traced, 19 of them trainable: 19 x 4 segments = 76 pairs for a
370 M model. Training loss fell ~3x, holdout latent rel-l2 against the t20
endpoint went **0.687-0.818 (teacher weights on the frozen grid) -> 0.84 at
epoch 7 -> 0.95-1.02 at epoch 8**. The run then died in the eval cell on
`import k2e` because the artefact is placed as `k2e_distill.py`.

Two findings survived the crash:

1. 76 pairs cannot teach 370 M parameters; the loss fall was memorisation.
2. Even the teacher's own weights, run on the student's frozen 4-step grid with
   guidance off (`g04`), were 0.687-0.818 away from the t20 endpoint. The
   "before" row was already bad, which should have been a warning about the
   grid, not just about the student.

### k11 — 384 pairs, 6 epochs (finished, 31,910.6 s)

108 photos traced in 5,400 s, 12 held out, 432 pairs (384 trained), six epochs,
train loss 0.2299 -> 0.1164 at ~69 min/epoch. Five candidate checkpoints, every
one scored on all 12 holdout photos.

| checkpoint | latent rel-l2 | cos |
|---|---|---|
| g04 — teacher weights, same grid, guidance off | **0.7321** | 0.6776 |
| e002 — chosen | 0.8370 | 0.5827 |
| student_best | 0.8453 | 0.5842 |
| e006 | 0.8560 | 0.5902 |
| e004 | 0.8788 | 0.5686 |

No checkpoint beat the weights it started from on latent rel-l2. On the gate,
t20 was 0.859/0.950/1.00, 12/12; g04 was 0.351/0.326/0.96, 0/12; b04 was
0.375/0.436/0.99, 0/12.

Then the renders inverted the ranking. b04 is a recognisable but soft figure —
hair, face, jacket stripes. g04 is a featureless blob. **The metric that was
used to choose the checkpoint ranked the blob above the figure.** Latent
rel-l2 to the endpoint is not a quality metric in this regime and is now banned
as a selector.

---

## 3. Why k11 failed — five mechanisms, in order of evidence

### 3.1 It trained and inferred on a schedule the teacher never uses

This is the concrete, provable one. The sampler is

```
t_seq = shift * linspace(1, 0, steps+1) / (1 + (shift-1) * linspace(1, 0, steps+1))
dt    = t - t_prev
x     = x - v * dt
```

k11's student grid was sigma = 1.0, 0.75, 0.5, 0.25, 0 — that is `steps=4,
shift=1.0`. The teacher's stock 4-step grid (`t04`) is `steps=4, shift=3.0`,
which gives sigma = 1.0, 0.9, 0.75, 0.5, 0. It is not the same path.

Worse, the shift=3 grid is exactly a subset of the 20-step grid: the 20-step
states at u = 1, 0.75, 0.5, 0.25, 0 are sigma = 1, 0.9, 0.75, 0.5, 0, i.e.
teacher states **0, 5, 10, 15, 20**. That is five teacher steps per student
segment. k11's `pick_grid` in "sigma" mode instead asked for sigma = 1, 0.75,
0.5, 0.25 and picked the nearest recorded states, which are teacher indices
**0, 10, 15, 18, 20** — segments of **10, 5, 3 and 2** teacher steps. The first
student step is a chord over half the teacher's trajectory, through the most
curved part of the path; the last is a two-step chord. The docstring justified
this by Euler truncation (`dt^2` is smaller on a sigma-even grid), but that
argument assumes the target is a tangent. It is not: it is the chord, and the
chord over ten teacher steps is exactly the low-pass filter that produced the
soft figure.

The empirical signature is in the controls: g04 — teacher weights on that same
shift=1 grid — is a blob. t04 — the teacher on its native grid with CFG — passes
8/12. The student was chasing a path whose own teacher control fails.

### 3.2 Off-policy only: the student's own states were never supervised

Every target was computed at the teacher's states. At inference the student
starts from the same noise, but its first step is not exactly the teacher's, so
from step 2 onward it is off the manifold the targets describe. With 4 huge
steps there is no chance to recover. k11 trained on 384 teacher states and
never once trained on a state the student itself would visit.

This is exposure bias, and it is the reason the whole on-policy distillation
family exists (Section 4). k11's student is a recognisable figure because the
averaged targets did carry some guidance, but it is soft because it is being
asked to extrapolate from a thin shell of teacher states.

### 3.3 The objective was a low-pass filter

The target for each segment is the mean velocity over the segment. For a large
segment, that is a heavily smoothed version of the teacher's field: high-
frequency state dependence is averaged away. Four such segments, plus a 370 M
network trained on 384 examples, produce exactly what the renders show — the
low frequencies (the figure) survive, the high frequencies (surface, thin
structure) do not.

This is the documented failure of pure trajectory regression; DMD/ADD/LADD
papers all say the same thing (Section 4).

### 3.4 The selector chose the wrong model

Latent rel-l2 to the endpoint ranked g04 (blob) above b04 (figure). It also
made the "best" checkpoint e002, the earliest saved one. The selection problem
is not cosmetic: with a wrong selector, a longer run or a better recipe can look
worse than its own starting point, which is exactly what k9 and k11 reported.

The gate is the only selector that has ever agreed with the renders (it ranks
t20 > t04 > t02 and rejects both g04 and b04). Use it, with a render sheet for
every final candidate.

### 3.5 Data scale

96 train photos x 4 segments = 384 pairs. k9 showed 76 pairs memorise. k11
showed 384 pairs still overfit: train loss 0.2299 -> 0.1164 while the holdout
latent gap did not move. The student is not underfit; it is being asked to
generalise a 370 M field from a few hundred teacher states. More pairs per
photo (more noise seeds) and more photos are both needed, and neither is
free.

---

## 4. What the literature actually does

### 4.1 The direct analogue: Z-Image / Z-Image-Turbo

Z-Image is a 6 B single-stream S3-DiT. The base pipeline (SFT + CFG) needs about
50 steps x 2 branches = ~100 NFEs. Z-Image-Turbo is the **same 6 B model** at
8 NFEs, with `guidance_scale=0.0`, sub-second on an H800. The full training
workflow is reported at **314 K H800 GPU-hours** (pretraining + distillation +
reward post-training); the distillation/post-training part is a large fraction
of that.

The recipe has two stages, both published:

- **Decoupled-DMD** (arXiv 2511.22677): DMD's success comes from two separable
  mechanisms. **CFG augmentation (CA)** is the engine — it is what teaches the
  few-step model to be sharp; **distribution matching (DM)** is the regulariser
  — it keeps the student near the real distribution and removes artifacts. They
  give CA and DM *different noise schedules* (tau_CA > t, tau_DM in [0,1])
  instead of training one blurred objective. The paper's thesis is literally
  "CFG augmentation as the spear, distribution matching as the shield."
- **DMDR** (arXiv 2511.13649): DMD and RL at the same time. DMD regularises
  against reward hacking; RL pushes past the teacher. The abstract claims
  few-step generation that "surpasses the multi-step teacher." Dynamic
  distribution guidance (a small LoRA) and dynamic renoise sampling (high noise
  early) are the two mechanisms.

The important part for us is not the 6 B scale. It is the shape: **teacher
init, guidance folded in, on-policy, distribution-level objective, then reward.**
No stage is "regress the student onto the teacher's states."

### 4.2 DMD / DMD2

DMD (arXiv 2311.18828) and DMD2 (arXiv 2405.14867) minimise the reverse-KL
between the student's output distribution and the real distribution using two
score models: a frozen real score (the teacher) and a **fake score trained on
the student's own samples**. The generator is updated on its own outputs
(on-policy by construction). DMD2's changes matter:

- no regression loss at all;
- the fake score is updated **5x per generator step**;
- a GAN loss is attached at the fake-score bottleneck, which restores
  high-frequency detail;
- backward simulation: train on states the sampler will actually visit, not
  just the clean endpoint.

Cost: **~3,840 A100 GPU-hours** for the main experiments; HyperSD is ~800.
Even the cheap variant is hundreds of GPU-hours, i.e. ~100x our remaining
weekly quota.

### 4.3 Adversarial distillation: ADD / LADD / SD3.5 Large Turbo

ADD (Sauer et al., arXiv 2311.17042) combines a score-distillation term with a
discriminator on noisy samples. LADD and the SD3.5 Large Turbo (8 B, 4 steps)
scale the same idea to a large rectified-flow backbone, initialised from the
teacher, with a latent-space discriminator. The lesson: the adversarial term is
what puts back the high frequencies that an L2 objective averages away. The
cost is a discriminator, a second backward, and careful balancing.

### 4.4 The cheap, on-policy family: pi-Flow

**pi-Flow** (ICLR 2026, Stanford + Adobe) is the closest thing to what we can
afford. The student's output head predicts a network-free *policy*; the policy
produces velocities at substeps without extra network calls. Training is plain
L2 flow matching, but **the target is the teacher's velocity along the policy's
own trajectory** — on-policy imitation, no fake model, no discriminator, no
JVP. It reports 4 NFE on FLUX.1-12B and Qwen-Image-20B at teacher-level quality
and better diversity, and 1-NFE FID 2.85 on ImageNet 256.

The paper's own framing is the one that matters here: the format mismatch
between "student predicts a shortcut to the endpoint" and "teacher predicts a
velocity" is what makes previous methods complicated. Matching velocities along
the student's own path removes the mismatch. Our k2e objective is already a
velocity objective; it is the *along the student's own path* part that is
missing.

### 4.5 3D-specific: MDT-dist

**MDT-dist** (arXiv 2509.04406, "Few-step Flow for 3D Generation via
Marginal-Data Transport Distillation") compresses a 3D flow model from tens of
steps to 1-2. Its two terms are **Velocity Matching (VM)**, which stabilises the
trajectory, and **Velocity Distillation (VD)**, which removes artifacts and
fills structure. Both are needed; either alone is worse. It beats consistency
models on 3D benchmarks. The caveat is data: it trains on abundant 3D geometry,
which we do not have.

### 4.6 Few-shot / low-compute: Uni-DAD

**Uni-DAD** (CVPR 2026, Bahram & Granger, ETS Montreal) does distillation and
adaptation in one stage for few-shot domains: **NFE=3, ~2.5 H100 GPU-hours**,
dual-domain DMD plus a multi-head GAN, with Source Teacher / Target Teacher /
Fake Teacher / Student Generator roles. It matches or beats non-distilled
fine-tuning on Babies, Sunglasses, MetFaces and Cats at 25x fewer NFEs. The
limits are explicit: it needs a pretrained source-domain teacher, several
discriminators, and it is only validated on face/object few-shot benchmarks.

This is the only published recipe in our compute range. It is also the proof
that 3 NFE is possible with a few-shot dataset — but with a GAN and a source
teacher, neither of which we have.

### 4.7 The classical path, and why it is not enough

Progressive distillation (Salimans & Ho), guided distillation (Meng et al.),
LCM and consistency models all work by halving the step count or enforcing
self-consistency along the teacher's ODE. Guided distillation is where "fold
CFG into the student" comes from, and it is the right *first* stage. But below
~4 steps these methods trade quality for speed; TwinFlow (arXiv 2512.05150)
notes that progressive/consistency approaches degrade at <4 NFE and adds a
self-adversarial term. TripoSG's real-time "Tripo Doodle" uses CFG distillation
for 3D — evidence that the 3D field does use guidance folding, not that it is
sufficient.

### 4.8 One line from industry practice

A Runway write-up on real-time video generation puts it directly: off-policy
distillation is a strong starting point, but **most of the actual gains come
from on-policy distillation** — rollouts inside training, each generated latent
becoming the next state. That is the single biggest thing k11 did not do.

---

## 5. The synthesis — nine rules the failures map onto

1. **Initialise from the teacher.** We do. Keep doing it.
2. **Fold guidance into the student; do not hope it is inferred.** k11's target
   did carry guidance, but the 4-forward control (g04) ran with guidance off,
   so the comparison could not attribute the gap. The inference procedure must
   be exactly the training procedure: 4 forwards, no CFG, same grid, same dt.
3. **Train on the student's own states.** Off-policy teacher states are a
   starting point, not the dataset. This is the pi-Flow/Runway/DMD lesson and
   the most likely reason k11's figure is soft.
4. **Match displacements/distributions, not smoothed teacher velocities.** A
   segment-average is a low-pass filter. If the objective is only that, expect
   blur. The fix is either a distribution term (DMD) or an adversarial term
   (ADD/LADD), or both.
5. **Keep a local velocity term when using an endpoint term.** MDT-dist's VM+VD:
   the endpoint target fills structure, the local velocity keeps the path
   stable. One alone is worse.
6. **Use the teacher's schedule.** The native 4-step shift=3 boundaries are
   teacher states 0/5/10/15/20; the target becomes exact with no interpolation.
   Do not invent a uniform grid because Euler truncation looks smaller on paper.
7. **Select on renders/gate, never latent distance.** k11 selected a blob over a
   figure. Latent rel-l2 is a diagnostic, not a selector.
8. **More data is not optional.** 384 pairs for 370 M is memorisation territory.
   More noise seeds per photo is the cheapest multiplier for the local-velocity
   term; more photos is the only multiplier for the endpoint term.
9. **Budget honestly.** Z-Image's workflow is 314 K H800-hours. DMD2 is 3,840
   A100-hours. Uni-DAD, the cheapest published few-shot recipe, is 2.5 H100-hours
   with a GAN. Our remaining quota is ~11.5 T4-hours. This is a pilot, not a
   reproduction.

---

## 6. What to run next: k12

### 6.1 The hypothesis

k11 had the right endpoint target and the wrong three things: the shift=1
grid, no on-policy states, and a selector that could not see blur. k12 changes
exactly those three, in that order, and keeps everything else — teacher init,
same 370.53 M, teacher front end, 4 forwards, guidance off at inference.

The native-grid target is exact. With the 20-step trace states s_0..s_20:

```
boundaries   s_0, s_5, s_10, s_15, s_20     (sigma 1, 0.9, 0.75, 0.5, 0)
displacement d_k = s_{5(k+1)} - s_{5k}
target       v_k = d_k / dt_k,  dt = (0.1, 0.15, 0.25, 0.5)
```

The displacements telescope, so a student that outputs v_k exactly lands on
s_20 in four Euler steps. No interpolation, no nearest-state fudge.

### 6.2 The loop

```
for each photo:
    x1, cond = trace[photo]                      # same start as the teacher
    # offline phase (epochs 1-2): teacher states
    for k in 0..3:
        loss += || student(s_k, t_k, cond) - v_k ||^2
    # on-policy phase (epochs 3..):
    x = x1
    for k in 0..3:
        x_visited[k] = x
        x = x + student(x, t_k, cond) * dt_k     # no grad
    for k in 0..3:
        v_local = teacher_cfg(x_visited[k], t_k, cond)   # 2 forwards, GPU1
        loss += || student(x_visited[k], t_k, cond) - v_k ||^2
        loss += lambda * || student(x_visited[k], t_k, cond) - v_local ||^2
```

`teacher_cfg` is called directly on the module, so its sign convention must be
checked against k2e's target convention by replaying the teacher's own t20 path
from `x1` and reproducing the trace. Do that before the first training step, not
after the run.

- The teacher stays on `cuda:1`, fp16, `no_grad`; the student and Adam stay on
  `cuda:0`. This is the memory answer to k2e's "drop the teacher" problem.
- `lambda` starts at 0.25. The endpoint term is the point; the local term is
  the regulariser that keeps the student on the teacher's manifold.
- Checkpoint every epoch; gate-evaluate the 12 holdout photos every epoch;
  choose by gate pass count, then by opacity+shell, never by latent distance.

### 6.3 The controls, at the same 12 photos

| control | forwards | what it isolates |
|---|---|---|
| t20 | 40 | the target |
| t04 | 8 | teacher, native grid, CFG — the bar to beat at 4 student forwards |
| g04-native | 4 | teacher weights, native grid, no CFG — the "before" for the training |
| b04-native | 4 | the student — the "after" |

k11 had g04 on the shift=1 grid; k12's g04-native is the missing control. If
g04-native is close to t04, the schedule was the killer. If b04-native is far
above g04-native but below t04, the remaining gap is the distribution/adversarial
term, not the grid.

### 6.4 Budget (2xT4, one session)

| stage | estimate |
|---|---|
| setup + front end + hash gates | 0.3 h |
| 108-photo t20 trace (k2d, measured 5,400 s in k11) | 1.5 h |
| offline, 2 epochs x 384 pairs at 9.7 s/pair | 2.1 h |
| on-policy, 2 epochs x 96 photos at ~75 s/photo | 4.0 h |
| gate eval after each epoch (12 photos, decode only) | 0.6 h |
| final decode: b04 + t04 + g04-native + t20, 12 photos | 1.4 h |
| **total** | **9.9 h** |

Remaining quota at the time of writing: **41,547 s (~11.5 h)**; refresh
2026-11-07T00:00:00Z. The run fits with ~1.6 h of slack. The script should
check `BUDGET - elapsed` before the on-policy stage and skip to the final
decode if it is short, rather than being killed mid-write like k9.

### 6.5 Go / no-go

- **Pass at 8/12 or better**: the 4-forward student matches the teacher's own
  4-step sampler at half the forwards (2x vs t04, 10x vs t20). Ship it as the
  turbo DiT, then decide whether to spend the next quota on the DMD stage to
  close the remaining t20 gap.
- **Between 1/12 and 7/12**: the grid+on-policy fixed part of it. The residual
  is the missing distribution term; k13 adds Decoupled-DMD-lite (CA + DM) with
  a LoRA fake score, not a second full model.
- **0/12 again**: stop retraining trajectory objectives. The next run must be a
  distribution-matching run with a discriminator or a fake score, or the
  project needs a larger data/compute budget. No more "same recipe, more
  epochs."

### 6.6 What k12 is not

It is not a Z-Image-Turbo reproduction. It is a ~10 T4-hour pilot of the same
*mechanism* (teacher init, folded guidance, on-policy, endpoint-matched) at
~1/1000 of the cheapest published few-shot distillation cost (Uni-DAD, 2.5
H100-hours) and ~1/30,000 of Z-Image's workflow. The realistic best case is
t04-level quality at 4 forwards. A true t20-level 4-forward model needs the
DMD/CA + reward stage, more photos, and more noise seeds — a multi-session
campaign, not one kernel.

---

## 7. The next stage, if k12 is not enough (k13)

Ordered by expected gain per GPU-hour:

1. **Decoupled-DMD-lite.** Freeze the teacher as the real score on GPU1; add a
   LoRA fake score on top of a frozen copy (DynaDG-style) so it fits; update
   the fake score 5x per generator step; CA term on the real score with a
   higher noise level; DM term as the regulariser. This is the Z-Image engine,
   shrunk to one 370 M backbone and two GPUs. Expect 20-30 h for a meaningful
   run, i.e. the whole next quota.
2. **A small latent discriminator.** LADD-style, on the decoded 3DGS latent or
   on rendered views. This is the cheapest way to put high frequencies back;
   a 3-layer MLP on 8192x16 tokens may be enough. Balance with a hinge loss and
   a slow discriminator LR.
3. **More noise seeds per photo.** For the local-velocity term only: the
   teacher field is defined for any state, so a noise pool from the 108 initial
   states gives free on-policy diversity. The endpoint term still needs one
   trace per seed.
4. **More photos.** Rows 108-221 of the DownFlow/fuliji set are untraced. 114
   more photos is another ~1.5 h of tracing and ~1.5x the endpoint data.
5. **Reward / preference fine-tuning.** Only after the 4-forward model is
   stable; it cannot fix a broken sampler.

---

## 8. Traps checklist (read before writing the next kernel)

1. Never select a checkpoint by latent rel-l2. Decode and gate.
2. Never compare variants that differ in more than one factor (k11's t04 vs g04
   differed in grid *and* guidance).
3. Match training and inference exactly: grid, dt, t values, guidance scale,
   initial noise distribution, camera convention.
4. Validate the teacher replay before training: integrate the teacher's field
   on the chosen grid and reproduce the trace's endpoint. If it does not, the
   target construction is wrong.
5. Keep the teacher on the second GPU rather than dropping it; do not let the
   memory budget force off-policy.
6. Gradient checkpointing on; first-sample peak printed; skip-on-OOM with a
   reason.
7. Rehearse the whole eval body (pipeline, integrate, decode, save_ply, 1.48 GB
   checkpoint round trip) against teacher weights before the real run.
8. Save every epoch; evaluate every epoch; stop early if the gate does not move.
9. Mint fresh Kaggle signed URLs immediately before pushing; verify with curl +
   sha256 locally. Expired URLs killed k11 v1 at 1.3 s.
10. Never set `HF_ENDPOINT` to hf-mirror on Kaggle; it breaks snapshot_download.
11. A new kernel needs a fully qualified slug (`owner/slug`); a bare slug is
    rejected for every value.
12. Type readable Python, not base64; the hash gate makes a bad transcription
    cost 30 seconds instead of 3 hours.
13. Do not let one bad import kill a finished run: check the artefact name
    (`k2e_distill.py`, not `k2e.py`) in the eval cell.
14. A 2-image eval is not a selector; k11's "best" was chosen on 2 photos.
15. If train loss falls while holdout rises, stop. That is memorisation, not
    distillation.

---

## 9. References

- Z-Image / Z-Image-Turbo — arXiv 2511.22699. 6 B S3-DiT, 8-step Turbo,
  guidance 0, Decoupled-DMD + DMDR, 314 K H800 GPU-h full workflow.
- Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the
  Shield — arXiv 2511.22677.
- DMDR: Distribution Matching Distillation Meets Reinforcement Learning —
  arXiv 2511.13649.
- DMD2 — arXiv 2405.14867; DMD — arXiv 2311.18828.
- ADD: Adversarial Diffusion Distillation — arXiv 2311.17042; LADD / SD3.5
  Large Turbo (4-step 8 B, latent adversarial + score distillation).
- Guided distillation — Meng et al., "On Distillation of Guided Diffusion
  Models" (CVPR 2023); progressive distillation — Salimans & Ho (2022).
- pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation — ICLR
  2026, Stanford + Adobe.
- TwinFlow — arXiv 2512.05150.
- Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step
  Few-shot Image Generation — CVPR 2026, Bahram & Granger, ETS Montreal.
- MDT-dist: Few-step Flow for 3D Generation via Marginal-Data Transport
  Distillation — arXiv 2509.04406.
- Phased DMD — arXiv 2510.27684.
- TripoSG / Tripo Doodle — VAST, CFG distillation for real-time 3D.
