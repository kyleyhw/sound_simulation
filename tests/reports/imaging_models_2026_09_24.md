# Learned imaging models, pose error, passive and 3D sensing (Phase 6: 6.3, 6.6) — 2026-09-24

**Headline.** A compact U-Net fed the four grid-aligned physics images and
the prior reaches held-out IoU **0.362** at K = 4, against **0.189** for the
logistic fusion of the same images and **0.101** for the no-audio prior.
The paired difference against the logistic fusion is **+0.173 ± 0.008
(z = 22.0)**, and against the prior **+0.261 ± 0.009 (z = 28.2)**. That is
500 held-out rooms, with nothing tuned on them. With all 8 poses the U-Net
reaches 0.450. A network that migrates the recovered impulse responses
itself does as well (0.370). The same impulse responses fed to a global,
non-aligned encoder stay at the prior (0.099). This is the Phase 2 plateau,
reproduced with better inputs. **The ≈ 0.10 ceiling of the Phase 2 CNNs
came from the architecture discarding geometry, not from the data.**

Robustness is the weak point. Half a cell of pose error per coordinate
costs a third of the U-Net's IoU, and σ = 2 cells brings it back to the
prior. Refining the poses against the empty-box model recovers 65-78 % of
that loss. Passive imaging (unknown source) barely beats the prior. A
small 3D demonstration beats a 3D prior with z = 14.

Figures and numbers: `imaging_models_2026_09_24_artifacts/`
(`results.json`, `examples.png`, `k_sweep.png`, `reliability.png`,
`pose_robustness.png`, `room3d.png`). Maths: `docs/imaging.md` §8.

## What was built

New files only. No existing file was modified except an append to
`docs/imaging.md`.

| file | plan | content |
|---|---|---|
| `src/acoustic_system/imaging/pose_images.py` | 6.3, 6.6.1 | per-pose physics images (their sum equals `compute_room_images` exactly), grid geometry channels, pose perturbation |
| `src/acoustic_system/imaging/models.py` | 6.3.1-6.3.4 | `AlignedUNet`, `IRMigrationNet` (differentiable delay-and-sum), `IRGlobalNet` (control), `PoseSetNet` (attention/DeepSets), D4 augmentation, training loop, ensembles/MC dropout, temperature, reliability |
| `src/acoustic_system/imaging/pose_refine.py` | 6.6.2 | joint exhaustive least-squares pose fit to the empty-box model; alternating search via reciprocity with a quiet-start cost |
| `src/acoustic_system/imaging/passive.py` | 6.6.4 | GCC-PHAT interferometric imaging (source signal unused), empty-box GCC removal |
| `src/acoustic_system/imaging/room3d.py` | 6.6.5 | 3D rooms, 4-mic arrays, 3D back-projection and carving |
| `scripts/eval_imaging_models.py` | all | reproduces every number here |
| `tests/imaging/test_{models,pose_images,pose_refine,passive_room3d}.py` | — | 31 new tests (~4 s) |

## Protocol (no held-out tuning)

- **Data.** The same archives as the physics report: training
  `active_sensing_v2_train_10kx4.hdf5` and held-out
  `active_sensing_v2_heldout_500x8.hdf5` (64², p = 0, 2 mics, chirp).
- **Rooms.** Training rooms 0-499 are the *validation* rooms. The logistic
  "all physics" fusion is fitted there exactly as in the physics report
  (it reproduces 0.1894 / 0.2444 / 0.0560). Every learned model takes its
  IoU threshold and temperature there. Training rooms **500-3999 (3500
  rooms, K = 4 poses each) train the networks**. Their per-pose images are
  cached outside the repo (`/tmp/acoustic_imaging_models_cache`, 4.5 GB).
  The priors are the physics report's.
- **Held-out.** All 500 held-out rooms, first K poses: K = 4 for the main
  table, and K = 1, 2, 8 for the pose sweep (the illuminated target only at
  K = 4, since its prior is a 4-pose prior). Metrics per room: IoU at the
  validation threshold, AP, boundary F (1-cell tolerance), and information
  gain over the prior in bits. Every comparison is the paired per-room
  difference, mean ± SE, with z = mean/SE.
- **Leakage check.** No held-out mask occurs among training rooms 0-3999
  (0 exact duplicates of 500). The median IoU between a held-out mask and
  its nearest training mask is 0.41 (maximum 0.80). A memoriser could reach
  that only if it knew which layout to recall.
- **Training.** AdamW (lr 2e-3, wd 1e-4), cosine schedule, batch 16, BCE
  summed over the two heads, a random D4 symmetry per batch, and one torch
  thread. The U-Net trained 20 epochs, the IR nets 10, and the set net 6
  (the shared machine was loaded 3-4× over; see Runtime). A U-Net trained
  for 6 epochs is included as the budget-matched control for the set net.
  No early stopping and no hyperparameter search: the only choices are
  these defaults.

## Main table

Filled obstacle mask (the Phase 2 target) and illuminated boundary (6.1.1).
Mean ± SE over held-out rooms. Δ columns are paired, and z is in brackets.

| target | K | method | IoU | AP | BF | info gain (bits/room) | ΔIoU vs prior (z) | ΔIoU vs logistic (z) | ΔAP vs logistic (z) |
|---|---|---|---|---|---|---|---|---|---|
| filled | 4 | no-audio prior | 0.101 ± 0.003 | 0.128 | 0.119 | 0 | — | — | — |
| filled | 4 | logistic all physics | 0.189 ± 0.005 | 0.344 | 0.250 | 227 ± 8 | +0.088 ± 0.005 (19.3) | — | — |
| filled | 4 | **aligned U-Net (6.3.1)** | **0.362 ± 0.008** | 0.578 | 0.457 | 480 ± 10 | +0.261 ± 0.009 (28.2) | **+0.173 ± 0.008 (22.0)** | +0.234 ± 0.010 (22.9) |
| filled | 4 | IR migration net (6.3.2) | **0.370 ± 0.007** | **0.598** | **0.465** | **509 ± 10** | +0.268 ± 0.008 (32.1) | +0.180 ± 0.007 (24.4) | +0.254 ± 0.010 (25.2) |
| filled | 4 | IR global encoder (6.3.2, no alignment) | 0.099 ± 0.003 | 0.157 | 0.094 | 65 ± 4 | −0.002 ± 0.001 (−1.7) | −0.090 ± 0.004 (−21.0) | −0.187 ± 0.010 (−19.6) |
| filled | 4 | pose-set net (6.3.3, 6 epochs) | 0.346 ± 0.008 | 0.552 | 0.425 | 447 ± 9 | +0.244 ± 0.009 (28.1) | +0.156 ± 0.007 (21.3) | +0.207 ± 0.010 (20.8) |
| filled | 4 | U-Net, 6 epochs (control) | 0.338 ± 0.008 | 0.544 | 0.428 | 440 ± 10 | +0.236 ± 0.009 (26.9) | +0.148 ± 0.007 (19.9) | +0.200 ± 0.010 (20.0) |
| filled | 4 | U-Net + T (6.3.4) | 0.362 ± 0.008 | 0.578 | 0.457 | 480 ± 10 | +0.261 ± 0.009 (28.2) | +0.173 ± 0.008 (22.0) | +0.234 ± 0.010 (22.9) |
| filled | 4 | U-Net MC dropout ×16 + T | 0.362 ± 0.008 | 0.579 | 0.455 | 480 ± 10 | +0.260 ± 0.009 (28.5) | +0.172 ± 0.008 (22.2) | +0.234 ± 0.010 (23.0) |
| filled | 4 | U-Net ensemble ×2 + T | 0.368 ± 0.008 | 0.586 | 0.464 | 488 ± 10 | +0.266 ± 0.010 (28.0) | +0.178 ± 0.008 (22.1) | +0.242 ± 0.010 (23.8) |
| filled | 4 | passive logistic (6.6.4) | 0.105 ± 0.003 | 0.146 | 0.134 | 6 ± 3 | +0.004 ± 0.002 (1.8) | −0.084 ± 0.004 (−20.1) | −0.198 ± 0.009 (−22.9) |
| filled | 4 | passive U-Net (6.6.4) | 0.110 ± 0.004 | 0.154 | 0.135 | 22 ± 3 | +0.009 ± 0.002 (4.1) | −0.079 ± 0.004 (−17.7) | −0.190 ± 0.009 (−20.6) |
| filled | 1 | logistic | 0.089 ± 0.004 | 0.191 | 0.147 | 46 ± 7 | −0.012 ± 0.005 (−2.6) | — | — |
| filled | 1 | U-Net | 0.182 ± 0.006 | 0.328 | 0.249 | 145 ± 10 | +0.080 ± 0.006 (13.0) | +0.092 ± 0.005 (18.4) | +0.137 ± 0.009 (14.8) |
| filled | 1 | IR migration | 0.152 ± 0.004 | 0.252 | 0.237 | 10 ± 11 | +0.050 ± 0.004 (11.7) | +0.063 ± 0.004 (15.1) | +0.062 ± 0.007 (8.5) |
| filled | 1 | pose-set net | 0.176 ± 0.005 | 0.312 | 0.235 | 140 ± 9 | +0.074 ± 0.005 (14.8) | +0.086 ± 0.004 (20.8) | +0.122 ± 0.009 (13.9) |
| filled | 2 | logistic | 0.140 ± 0.005 | 0.259 | 0.204 | 131 ± 8 | +0.038 ± 0.005 (8.0) | — | — |
| filled | 2 | U-Net | 0.261 ± 0.007 | 0.451 | 0.336 | 312 ± 11 | +0.159 ± 0.008 (21.1) | +0.121 ± 0.006 (19.3) | +0.192 ± 0.010 (18.5) |
| filled | 2 | IR migration | 0.249 ± 0.006 | 0.439 | 0.338 | 274 ± 11 | +0.148 ± 0.006 (23.0) | +0.109 ± 0.006 (18.7) | +0.181 ± 0.011 (17.1) |
| filled | 2 | pose-set net | 0.252 ± 0.006 | 0.436 | 0.312 | 303 ± 9 | +0.151 ± 0.007 (21.4) | +0.112 ± 0.006 (20.0) | +0.178 ± 0.010 (18.2) |
| filled | 8 | logistic | 0.244 ± 0.005 | 0.459 | 0.291 | 337 ± 8 | +0.143 ± 0.004 (34.1) | — | — |
| filled | 8 | **U-Net** | **0.450 ± 0.008** | **0.684** | **0.557** | **621 ± 10** | +0.349 ± 0.010 (35.9) | +0.206 ± 0.009 (22.8) | +0.225 ± 0.009 (24.6) |
| filled | 8 | IR migration | 0.422 ± 0.008 | 0.681 | 0.531 | 560 ± 9 | +0.320 ± 0.010 (32.8) | +0.177 ± 0.010 (18.2) | +0.222 ± 0.009 (23.9) |
| filled | 8 | pose-set net | 0.419 ± 0.009 | 0.657 | 0.524 | 534 ± 10 | +0.318 ± 0.010 (31.5) | +0.175 ± 0.009 (18.8) | +0.198 ± 0.009 (22.3) |
| filled | 8 | passive U-Net | 0.115 ± 0.003 | 0.169 | 0.130 | 28 ± 3 | +0.014 ± 0.002 (9.0) | −0.129 ± 0.004 (−31.7) | −0.290 ± 0.010 (−28.0) |
| illum. | 4 | no-audio prior | 0.027 ± 0.001 | 0.032 | 0.155 | 0 | — | — | — |
| illum. | 4 | logistic | 0.056 ± 0.001 | 0.093 | 0.224 | 54 ± 2 | +0.029 ± 0.001 (24.4) | — | — |
| illum. | 4 | U-Net | 0.163 ± 0.004 | 0.285 | **0.495** | 119 ± 3 | +0.136 ± 0.005 (29.6) | +0.107 ± 0.004 (25.7) | +0.192 ± 0.007 (27.4) |
| illum. | 4 | IR migration | 0.170 ± 0.004 | 0.291 | 0.483 | 125 ± 2 | +0.143 ± 0.004 (34.3) | +0.114 ± 0.004 (30.2) | +0.198 ± 0.006 (32.8) |
| illum. | 4 | IR global encoder | 0.026 ± 0.001 | 0.037 | 0.116 | 10 ± 1 | −0.001 ± 0.000 (−2.3) | −0.030 ± 0.001 (−24.8) | −0.056 ± 0.003 (−20.9) |
| illum. | 4 | pose-set net | 0.133 ± 0.003 | 0.229 | 0.435 | 100 ± 2 | +0.106 ± 0.004 (28.0) | +0.077 ± 0.003 (22.6) | +0.136 ± 0.005 (25.7) |
| illum. | 4 | U-Net ensemble ×2 + T | **0.172 ± 0.005** | **0.298** | 0.486 | 123 ± 3 | +0.145 ± 0.005 (28.6) | +0.115 ± 0.005 (24.9) | +0.205 ± 0.007 (28.3) |
| illum. | 4 | passive U-Net | 0.030 ± 0.001 | 0.035 | 0.169 | 2 ± 0 | +0.003 ± 0.000 (5.1) | −0.026 ± 0.001 (−21.3) | −0.058 ± 0.003 (−21.5) |

Paired differences between learned models (`results.json` → `pairs`):

| comparison | K | ΔIoU (z) | ΔAP (z) | Δ info gain, bits (z) |
|---|---|---|---|---|
| IR migration − U-Net | 4 | +0.007 ± 0.005 (1.4) | +0.020 ± 0.006 (3.3) | +30 ± 7 (4.4) |
| IR migration − U-Net | 1 / 8 | −0.030 (−6.1) / −0.029 (−5.0) | −0.075 (−9.5) / −0.003 (−0.5) | −135 (−16.3) / −61 (−7.7) |
| set net − U-Net (20 epochs) | 4 | −0.016 ± 0.004 (−3.9) | −0.027 ± 0.005 (−5.4) | −33 ± 5 (−7.0) |
| set net − U-Net (6 epochs, budget-matched) | 1 / 2 / 4 / 8 | +0.005 (1.4) / +0.006 (1.5) / +0.008 (2.0) / −0.004 (−1.0) | +0.006 / +0.008 / +0.008 / +0.006 | +11 (2.5) / +15 (3.6) / +7 (1.6) / −46 (−7.8) |
| ensemble ×2 + T − U-Net | 4 | +0.006 ± 0.002 (3.6) | +0.008 ± 0.001 (5.2) | +8 ± 1 (5.8) |
| MC dropout + T − U-Net | 4 | −0.001 ± 0.000 (−1.1) | +0.000 (2.1) | +0.4 ± 0.3 (1.3) |
| IR global encoder − prior | 4 | −0.002 ± 0.001 (−1.7) | +0.030 ± 0.004 (8.0) | +65 ± 4 (15.0) |

## Plan items

### 6.3.1 Aligned images → network: **done**

The key experiment. The compact U-Net (119 k parameters, 21 min of
training on one shared core) beats the logistic fusion at every K:
+0.092 (K = 1), +0.121 (K = 2), +0.173 (K = 4) and +0.206 (K = 8) IoU.
Every z is at least 18. It also gives +253 bits per room of information
over the logistic fusion at K = 4 (z = 35.8). It was trained only at
K = 4, on per-room standardised images, and still generalises to K = 1, 2
and 8. On the illuminated boundary it triples the logistic IoU (0.056 →
0.163) and reaches a boundary F of 0.495 against 0.224. The examples
(`examples.png`) show why. The logistic fusion adds smooth evidence
blobs. The U-Net draws thin walls, closes outlines, and clears shadowed
space that carving leaves ambiguous. Caveat: the network learns this
room generator's shape vocabulary (walls, L-shapes, discs). The leakage
check rules out memorised rooms, but not the benefit of a matched shape
prior. The Phase 2 CNNs also had that benefit and did not use it.

### 6.3.2 Impulse responses as inputs: **done** (IR + learned migration wins at K = 4; a non-aligned IR encoder stays at the prior)

Two versions, honestly split:

- **IR global encoder** (the Phase 2 layout with IRs replacing the
  spectrograms): IoU 0.099, statistically at the prior (z = −1.7). It is
  not empty: AP +0.030 (z = 8) and +65 bits/room. The information is
  there, but a vector bottleneck followed by a decoder cannot place it.
  This reproduces the Phase 2 result with a cleaner input, so the input
  representation was not the problem.
- **IR migration net**: a learned 1D filter bank on the (IR, envelope)
  traces, then a fixed, differentiable delay-and-sum onto the grid, then
  the U-Net. At K = 4 it is the best single model: IoU 0.370, AP 0.598,
  509 bits. That is +0.007 ± 0.005 IoU (z = 1.4) and +0.020 AP (z = 3.3)
  over the U-Net on hand-made images, from 10 epochs instead of 20. It
  generalises worse to other pose counts (−0.030 at K = 1, −0.029 at
  K = 8). Its migration averages over poses, so the evidence scale
  changes with K. The hand-made images are standardised per room.

### 6.3.3 Pose-aware set model: **done; negative result**

`PoseSetNet` encodes each pose's four images plus three geometry channels
with a shared encoder. It pools them with per-pixel attention, mean and
max, so it is invariant to pose order and defined for any K. It was
trained on random subsets of 1-4 poses. It works at every K (K = 1:
0.176, 2: 0.252, 4: 0.346, 8: 0.419) but does not beat summing the images
first. It loses to the 20-epoch U-Net at every K (−0.016 at K = 4,
z = −3.9). Against the budget-matched 6-epoch U-Net it is level: +0.005
to +0.008 IoU at K ≤ 4 (z ≤ 2), −0.004 at K = 8, and −46 bits at K = 8.
It is slightly overconfident there, since K = 8 is outside its training
range. Pose-aware fusion adds nothing measurable over the aligned sum. The
geometry is already in the migration, and sums are the right fusion for
evidence that adds. Caveat: the set net had 6 epochs (training cost ∝ K),
and a longer run could change the sign at K ≤ 4. It is unlikely to change
the conclusion.

### 6.3.4 Uncertainty: **done** (calibrated probability maps); a generative model of whole maps is **not done**

- The single U-Net is already calibrated. The validation temperature is
  T = 0.994 (filled) and 1.007 (boundary). The expected calibration error
  (15 equal-mass bins over all held-out pixels) is 0.0020 → 0.0018,
  against 0.0068 for the logistic fusion (`reliability.png`). The
  information gain over the prior, the proper score, is 480 bits/room
  against 227 for the logistic.
- MC dropout (16 draws, p = 0.1) changes nothing measurable: +0.4 ± 0.3
  bits, T = 0.99.
- A two-member deep ensemble + T gives a small, significant gain:
  +0.006 IoU (z = 3.6), +8 bits (z = 5.8), and on the boundary +0.008 IoU
  (z = 6.7). A third member was dropped for compute.
- The uncertainty tracks the errors. The per-pixel predictive entropy
  correlates with |truth − q| at r = 0.72 (filled) and 0.48 (boundary).
  The entropy map (`examples.png`, last column) lights up the unobserved
  regions and the edges of uncertain objects.

These are calibrated *marginal* per-pixel maps. The plan item's
"generative model" that samples whole plausible rooms was not built.

### 6.6.1 Robustness to pose error: **done**

Every device (the source and each mic) is displaced independently by
round(N(0, σ²)) per coordinate. The background simulation and all travel
times use the displaced cells, and the recordings stay true. There are 500
held-out rooms, K = 4, and the models were trained on exact poses. The
best method (U-Net) is shown, with the logistic fusion alongside
(`pose_robustness.png`):

| σ (cells) | mean device error | U-Net IoU (exact 0.362) | Δ vs exact (z) | vs prior (z) | logistic IoU (exact 0.189) |
|---|---|---|---|---|---|
| 0.5 | 0.59 | 0.235 | −0.127 ± 0.007 (−18) | +0.134 (16.9) | 0.138 |
| 1 | 1.26 | 0.144 | −0.218 ± 0.009 (−25) | +0.042 (8.4) | 0.094 |
| 2 | 2.50 | 0.105 | −0.257 ± 0.009 (−28) | +0.004 (1.0) | 0.063 |
| 3 | 3.76 | 0.089 | −0.273 ± 0.009 (−30) | −0.012 (−3.9) | 0.048 |

This is severe. Half a cell of error per coordinate (54 % of devices off
by at least one cell) costs 35 % of the IoU. At σ = 2 cells the U-Net is
back at the prior, and the logistic fusion falls below it from σ = 1. Two
things break together. The background subtraction leaves a direct-path
error that starts at the direct arrival, and carving keys on exactly that
onset. The travel times are also wrong. At the v2 scale (λ_min ≈ 2.2
cells) one cell is half a wavelength.

### 6.6.2 Joint pose-and-map estimation: **done** (recovers 65-78 % of the loss; poses are only partly recovered)

Refinement fits the empty-box model to the recordings within a window of
radius ⌈2σ⌉ around the assumed cells, then images with the refined cells
and their incident fields:

| σ | U-Net perturbed → alternating → **joint** | recovered fraction (alt. / joint) | joint Δ vs exact (z) | device error after joint (exact fraction) | logistic perturbed → joint |
|---|---|---|---|---|---|
| 0.5 | 0.235 → 0.319 → **0.334** | 0.66 / **0.78** | −0.028 ± 0.003 (−9) | 0.55 (58 %) | 0.138 → 0.179 |
| 1 | 0.144 → 0.275 → **0.314** | 0.60 / **0.78** | −0.048 ± 0.004 (−12) | 1.07 (47 %) | 0.094 → 0.169 |
| 2 | 0.105 → 0.223 → **0.293** | 0.46 / **0.73** | −0.069 ± 0.005 (−14) | 2.18 (38 %) | 0.063 → 0.157 |
| 3 | 0.089 → 0.177 → **0.266** | 0.32 / **0.65** | −0.096 ± 0.006 (−16) | 3.49 (34 %) | 0.048 → 0.146 |

The joint least-squares fit, exhaustive over (2⌈2σ⌉+1)² source cells
with the best mic cells for each, recovers 65-78 % of the IoU lost to pose
error. It also restores the logistic fusion to 77-95 % of its exact-pose
value. It does *not* recover the poses themselves well. The mean device
error barely drops (3.76 → 3.49 cells at σ = 3), and only 34-58 % of
devices land on the exact cell. It finds cells whose empty-box response
matches the data, which is what the background subtraction needs. The
cells can be wrong but equivalent: before the first wall echo, the direct
wave fixes only the source-mic distance.

Checks on the validation rooms (training archive; `results.json` →
`pose.dev_checks`) explain why exact poses are out of reach. The first
scattered arrival (the carving onset detector) trails the direct arrival
by a median of only 4 lags, and precedes it in 39 % of traces (the FDTD
precursor, or an obstacle near the direct path). So the pre-scatter window
holds almost no pose information. At σ = 1 on 10 rooms, the joint fit
puts 36 % of devices on their exact cell with the empty box (12.5 %
before refinement), and 93 % with the *true* obstacle map in the model.
So a full pose-and-map alternation could pin the poses, but only with a
far better map than these imagers give. Two development notes on 16
training rooms are not reproduced by the script. First, a coordinate-wise
least-squares search stalled in chirp local minima even with the true
map. Second, putting the estimated logistic map (IoU ≈ 0.14 under pose
error) into the model gave worse images than the empty box (IoU 0.17
against 0.23 at σ = 1). The refinement here is therefore pose given
background, then map, not a full alternation. The alternating reciprocity
search is 5-20× cheaper ((M + 1) simulations per iteration against
(2R + 1)²) and recovers 32-66 %.

### 6.6.3 Next-best pose: **done; heuristics beat random only marginally**

Starting from the first two of the eight held-out poses, the next pose is
chosen greedily from the remaining six, twice (to K = 4). The fused map is
the logistic fusion, so it is calibrated and needs no retraining for
pose subsets. Random is scored exactly, as the mean over all choices (6
at K = 3 and 15 pairs at K = 4). The hindsight oracle is a ceiling. All
500 held-out rooms are used, with nothing tuned on them. The sensitivity
profile v(d) was learned on validation rooms.

| strategy | IoU K = 3 | ΔIoU vs random (z) | info gain K = 3 | IoU K = 4 | ΔIoU vs random (z) | info gain K = 4 | Δ bits vs random (z) |
|---|---|---|---|---|---|---|---|
| (K = 2 start) | 0.140 | | 131 | | | | |
| random (expected) | 0.170 | — | 190 | 0.193 | — | 234 | — |
| (a) entropy within 12 cells of the pose centroid (web panel) | 0.173 | +0.002 ± 0.002 (0.9) | 190 | 0.198 | +0.006 ± 0.002 (2.3) | 239 | +4.8 ± 3.8 (1.3) |
| (b) sensitivity-weighted entropy Σ H(q)·v(d) | 0.175 | +0.005 ± 0.002 (2.1) | 198 | 0.199 | +0.006 ± 0.003 (2.5) | 240 | +5.7 ± 3.7 (1.5) |
| hindsight oracle (best IoU) | 0.234 | +0.064 ± 0.002 (28.8) | 268 | 0.268 | +0.076 ± 0.002 (34.2) | 313 | +78.8 ± 3.4 (23.4) |

Verdict: both heuristics are only just better than random. They gain
about 0.006 IoU at K = 4 (z ≈ 2.3-2.5, weak given the several
comparisons) and have no significant information gain. The oracle shows
that pose choice matters a lot (+0.076). An entropy map around the
devices does not predict which pose will pay off. The payoff depends on
the unknown geometry, such as which surfaces a pose illuminates and
which shadows it clears. A real expected-information-gain computation,
simulating candidate poses on posterior samples, would need the
generative model that 6.3.4 did not build. Criterion (b) is the
cheapest justified proxy (first order in the expected logit change) and
is no better than (a).

### 6.6.4 Passive sensing: **done; negative (barely above the prior)**

The source signal is never used, and the speaker and mic positions are
assumed known. GCC-PHAT of the two mics is exactly independent of the
source spectrum (unit-tested). It is migrated along the direct-scattered
delays, with and without the empty-box GCC removed (computed from an
impulse probe). Fused with the prior by logistic regression: +0.004 IoU
(z = 1.8) and +0.018 AP (z = 3.6) at K = 4, and +0.009 (z = 4.6) at K = 8.
A U-Net on the two passive images, trained like 6.3.1 for 6 epochs:
+0.009 IoU (z = 4.1), +22 bits at K = 4, and +0.014 IoU (z = 9.0) at K = 8.
It is significant but tiny, and 0.08 IoU below the active logistic
fusion. The passive image's fitted weight is *negative*: GCC energy marks
free paths more than scatterers. With a single two-mic pair per pose and a
400-sample recording, the cross-correlation is dominated by the
direct-direct peak and the box. Removing the box's GCC did not help
(0.106 vs 0.106).

### 6.6.5 3D demonstration: **done (small)**

32³ p = 0 boxes, 2-4 random box obstacles (held-out voxel occupancy
2.4 %), and a
source with four mics on a ±3-cell square. The drive is a known Ricker
(f₀ = 0.12), with 240 steps and K = 4 random poses per room. There are 200
training rooms (fit and threshold) and 100 held-out rooms. The 3D prior is
the mean of 2000 generator masks. Both imagers were fused with the prior
by logistic regression (smoothing chosen on training rooms):

| method (fused with the 3D prior) | K | voxel IoU | AP | info gain (bits) | ΔIoU vs prior (z) | ΔAP vs prior (z) |
|---|---|---|---|---|---|---|
| 3D prior | — | 0.070 | 0.086 | 0 | — | — |
| back-projection | 4 | 0.077 | 0.095 | 58 | +0.007 ± 0.003 (2.5) | +0.009 ± 0.006 (1.6) |
| carving | 4 | 0.132 | 0.177 | 673 | +0.062 ± 0.004 (15.8) | +0.091 ± 0.006 (15.5) |
| **both** | 4 | **0.139** | **0.199** | **685** | **+0.069 ± 0.005 (14.3)** | **+0.113 ± 0.008 (13.4)** |
| both | 2 | 0.091 | 0.136 | 350 | +0.021 ± 0.006 (3.6) | +0.050 ± 0.006 (9.1) |
| both | 1 | 0.041 | 0.106 | 94 | −0.029 ± 0.006 (−4.7) | +0.020 ± 0.003 (6.0) |

3D works as it does in 2D. Carving (negative evidence) carries it, and
back-projection alone is weak with a 6-cell aperture. At K = 1 the IoU
falls below the prior while AP and information gain rise. The fit and
threshold were chosen at K = 4, and one pose carves too little for that
operating point (`room3d.png`). No learned model was trained in 3D.

## Limitations

1. **Simulation, noise-free, exact background.** Everything inherits the
   physics report's caveats: the same simulator generated and explains the
   data, and the empty outer box is known. The learned models were not
   tested under noise (the physics report showed the imagers survive
   20 dB). 6.6.1 shows the chain is fragile to pose error, the first
   realism knob tried.
2. **Matched generator.** Training and held-out rooms come from the same
   generator. The networks learn its shape vocabulary. A real room will
   not share it.
3. **Compute-limited training.** The epochs (20 / 10 / 6) were set by the
   shared machine, not tuned, and the ensemble has only two members. The
   set-net comparison is budget-matched for exactly this reason.
4. **Multiple comparisons.** About 60 paired tests are reported. The
   headline z-values (≥ 18) survive any correction. The next-best-pose
   gains (z ≈ 2.3-2.5), the set-net-vs-control differences (|z| ≤ 2), and
   the passive logistic (z = 1.8) do not.
5. **6.6.2 is not a full joint estimate.** The refinement uses the known
   empty box, not the evolving map. Pose-map alternation with a better map
   (FWI) is the obvious next step. The oracle-map check says it would pin
   the poses exactly.

## Runtime (one torch/numba thread per process; machine shared, load 7-16 on 4 cores)

| stage | wall time |
|---|---|
| per-pose images, 4000 training rooms + 500 held-out (K = 8) | ~11 min |
| U-Net ×2 seeds (20 epochs each) | 21 min each |
| IR migration (10 epochs) / IR global (10 epochs) | 18 / 4 min |
| set net (6 epochs) / U-Net control (6 epochs) / passive U-Net (6 epochs) | 13 / 4 / 4 min |
| pose images (500 rooms × 3 variants), σ = 0.5 / 1 / 2 / 3 | 5 / 8 / 19 / 43 min |
| passive images (4500 rooms) | ~3 min |
| 3D rooms (300) | 1 min |
| scoring and figures from caches (the final run) | 6 min |

Two or three processes ran concurrently. In total the run took about
2 h 10 min of wall time.

## Reproduce

```
NUMBA_NUM_THREADS=1 uv run python scripts/eval_imaging_models.py \
    --out-dir tests/reports/imaging_models_2026_09_24_artifacts
NUMBA_NUM_THREADS=1 uv run pytest tests/imaging     # 57 tests, ~9 s
```

The stages cache under `--cache-dir` (default
`$TMPDIR/acoustic_imaging_models_cache`), so a second run only re-scores
(6 min). `--stages` runs a subset, for example
`--stages poseimg --pose-sigmas 3` to split the slow pose-image stage
across processes.
