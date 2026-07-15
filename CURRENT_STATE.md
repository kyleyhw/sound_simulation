# Project Status as of 2026-07-15

## 0. Sensing v2 (Task 2.3, added 2026-07-15)

The five sensing-quality improvements (a–e) from the demo-8 diagnosis
are built, trained, and evaluated (`docs/learning.md` §8, report
`tests/reports/sensing_v2_2026_07_15.md`): inter-channel phase
channels, chirp band to the spatial-Nyquist limit + doubled recording,
shape-diverse rooms, a multi-scale skip decoder (`SkipSensingCNN`,
`checkpoints/skip_v2` — now the demo/UI default), and calibrated
fusion with a validation-selected operating point.

**New best: held-out IoU 0.100 at K=4** (calibrated Bayes @ τ=0.12,
equal to the oracle-threshold ceiling, no leakage) vs 0.094 for the
v1 recipe on the same archive. Major methodological finding: the
fixed 0.5 threshold inflated all earlier fusion gains (oracle K=1
already reaches ~0.093; genuine pose-evidence accumulation is ≈ +8 %,
saturating at K≈4) — scalar calibration is affine-monotone in the
fused logit, so its IoU value is exactly the operating-point choice.

Outstanding: the architecture-vs-data attribution run
(`checkpoints/joint_v2`, v1 architecture on v2 data) is **suspended at
~epoch 40/60 by explicit user pause** (PIDs 40532/7824 frozen in
place; resume → ~35 min → eval → append to the report).

## 1. Demonstrations (added 2026-07-12)

Both phases are now demonstrable end to end (`docs/demos.md`):
`scripts/demo_room_mapping.py` maps a fresh room from the command line
(figure: truth + fused map sharpening over K=1..8 poses), and the web
UI gained an *Acoustic sensing* panel — draw a room, press **Sense
room**, get the Bayes-fused obstacle map in ~0.3 s while the live
simulation keeps streaming (`sense_room`/`sense_result` events, E2E
Playwright-verified). Shared engine: `learning/sensing.py`. Report:
`tests/reports/demos_2026_07_12.md`.

## 2. Resolution (Phase 1 + 2 close, 2026-07-11)

**Phases 1 and 2 are complete.** Phase 1 closed with the GPU backend
(Task 1.5); Phase 2 closed with the multi-pose active-sensing campaign
(Task 2.1.4) and the passive-sensing comparison (Task 2.2).

### The Phase 2 result in one table

Held-out obstacle-mask IoU @ 0.5 (500 rooms, 2-mic stereo, 64×64 grid):

| recipe | K=1 | K=8 |
| --- | --- | --- |
| single-pose CNN (2.1.3 baseline) | 0.037 | — |
| single-pose CNN + Bayes fusion (2.1.4b) | 0.038 | 0.092 |
| joint-pose CNN, native mean-pool (2.1.4c) | 0.056 | 0.039 |
| **joint-pose CNN per pose + Bayes fusion** | 0.056 | **0.0924** |
| passive CNN, unknown source (2.2) | 0.030 | — |

**Best recipe: run the joint-trained encoder on each pose separately
and fuse with the Bayes product rule**
$\sigma(\sum_k \ell_k - (K-1)\,\mathrm{logit}\,\hat\pi)$ — 2.4× the
single-pose baseline, unsaturated in K. Physical picture: carry the
laptop to K spots in the room, chirp at each, accumulate evidence.
This is the sensing recipe to integrate in Phase 4.

Key findings behind it (reports in `tests/reports/`):
- Single-pose 2-mic data is information-limited; regularisation makes
  it worse (2026-05-14 runs). Multi-pose is the lever — confirmed
  quantitatively by fusion-at-inference (`multipose_2026_07_10.md`).
- Joint multi-pose *training* regularises the encoder (better even as
  a single-pose predictor: 0.056 vs 0.038), but mean-pool fusion
  degrades with K (1/K latent-variance shrinkage pushes the decoder
  toward the prior). Explicit Bayes fusion beats the learned mean and
  stacks with the better encoder (`joint_pose_2026_07_11.md`).
- Source knowledge is worth little in the single-pose regime: passive
  (unknown source) loses only 19 % held-out (`passive_2026_07_10.md`).
  Passive multi-pose is untested — natural next experiment.

### Phase 1 close-out (Task 1.5)

`Simulate(backend="gpu")`: CuPy `RawKernel` twins of the fused 2D/3D
kernels, device-resident buffers, zero step-path transfers
(`p_host()` readback, `set_obstacle_mask()` bulk upload). Gates green
at ~1e-6 relative L2 vs the CPU truth chain; RTX 2070 SUPER speedups:
13.2× at 2048² (2D), 16.7× at 200³ (3D), crossover ≈512². CPU path
untouched (evolve-harness gates pass with identical error values).
See `docs/gpu.md` + `tests/reports/gpu_backend_2026_07_10.md`.

## 2. Assets

- Datasets (`data/training_data/`, gitignored, regenerable by seed;
  commands in `docs/learning.md`): single-pose train/held-out
  (1234/999), multi-pose 10k×4 train (31415) and 500×8 held-out
  (424242), passive randomized-source train/held-out (5678/8765).
- Checkpoints (`checkpoints/`, gitignored): `long_baseline` (verified
  reproduction of the 2026-05-14 run), `passive_baseline`,
  `joint_baseline` (best_iou.pt = epoch 36, the evaluated one).
- Models: `DualInputCNN` / `PassiveCNN` / `JointPoseCNN` behind
  `build_model`; checkpoints carry a `model_type` tag.

## 3. Operational notes

- Artifact convention: datasets → `data/training_data/`, checkpoints →
  `checkpoints/`, never `/tmp` (a temp cleanup destroyed the
  2026-05-14 artifacts; datasets were regenerated from seeds).
- Long training runs should be launched **detached**
  (`Start-Process`, logs to files): one 100-epoch joint run was lost
  at epoch 36 when a paused harness background task was reaped and the
  kill took the process tree. Seeded CPU reruns replay near-identically
  (the restart reproduced val IoU 0.0464 at epoch 36 exactly).

## 4. Open follow-ups (Phase 3+)

- Phase 3 (beamforming) is next per `PROJECT_PLAN.md`; the engine
  already supports multiple independent drivers.
- Sensing-side upgrades, if needed later: variance-normalised or
  log-sum-exp pose pooling / variable-K training; GCC-PHAT (TDOA)
  input channel for passive; passive multi-pose.
- 3D obstacle drawing in the web UI remains deferred.
