# Project Status as of 2026-09-24

## Where things stand after executing the plan (read first)

The live site is at https://kyleyhw.github.io/sound_simulation/. It
covers the sandbox, gallery, explainers, research write-ups, the
closed-loop demo, the room-measurement lab and the docs. Everything runs
in the browser. `PROJECT_PLAN.md` has the per-item status. The summary
by phase:

| phase | status | headline |
|---|---|---|
| 3 Debug audit | in progress | 18 engine, 12 backend and 10 learning findings fixed or retired (`tests/reports/debug_audit_2026_09_24.md`). The retrained leak-free skip_v2 still sits on the no-audio baseline (IoU 0.104 vs 0.101 at K = 4, z = 1.1; debug audit §3). **Open:** remote data hosting (3.5.3, needs a storage target), GPU run (3.2.3, needs hardware). |
| 4 UI overhaul | done | Browser app with no server. 27 Playwright end-to-end tests and 46 unit tests, run in CI and against the live site. |
| 5 Physics | done except the GPU run | Verification report `docs/physics.md`: modes to 1e-5 % of theory, second-order convergence, 1e-7 energy drift, CPML at −49 dB at 62°, T60 between Sabine and Eyring. |
| 6 Sensing, second attempt | done | Physics imagers beat the no-audio baseline: IoU 0.189 vs 0.101, z = 19.3 (K = 4); 0.244 at K = 8 (`tests/reports/imaging_2026_09_24.md`). **The Phase 2 CNNs sat at the baseline, so the "information ceiling" reading below is wrong.** A U-Net on the aligned physics images reaches **0.362** (K = 4) and 0.450 (K = 8), while a non-aligned encoder of the same impulse responses stays at the baseline (`tests/reports/imaging_models_2026_09_24.md`). Weak points: pose error (back to the baseline at 2 cells), passive sensing, and next-pose choice. A generative sampler (6.3.4) gives coherent room hypotheses but does not beat independent draws from the calibrated U-Net (mean-map IoU 0.311 vs 0.362; `tests/reports/imaging_generative_2026_09_25.md`). |
| 7 Control | done | ACC 25.5 dB broadband, measured in the time domain. Crosstalk cancellation 17.7 dB broadband in an absorbing room (84 % of the band per frequency). 15 dB needs walls known to 1.5–4 cm (`tests/reports/control_2026_09_24.md`). |
| 8 Closed loop | done | `#/loop`: sense → twin → ACC → measure, through a moving listener, a moving obstacle and a new partition. The monitor-guarded loop keeps 16–41 dB; the loop period is about 5 s. |
| 9 Real hardware | tools done, measurements pending | The Lab page covers sweep IRs, device calibration, echoes, T60, the room twin, crosstalk cancellation with head tracking, and phone orientation. It is tested with a fake device. Real-room numbers need the user (§4 stop point). |
| 10 Showcase | mostly done | Pages deploy, WebGPU engine (parity 1e-6), in-browser CNN (parity 1e-5), 15-scene gallery, 6 explainers, docs site, GIF renderer, benchmark, 4 write-ups. **Open:** dataset hosting (needs a storage target) and the real-hardware write-up (needs measurements). |

**New findings recorded during execution:**
- The skip model's phase channels are rounding noise in 57–64 % of the
  time-frequency bins (near-silent), and the model is sensitive to that
  noise.
- Numba's `prange` collapses under CPU oversubscription. Dataset
  generation now scales with `--workers` instead.

Everything below this line is the historical record from before the plan
was executed.

---

## Audit (2026-09-24) — read first

A full plan audit (`docs/plan_audit.md`) produced a revised
`PROJECT_PLAN.md` with ten sequential phases. The next two phases come
before any new feature work:

- **Phase 3:** a full codebase debug audit.
- **Phase 4:** a complete overhaul of the web UI.

**The key finding changes how to read the Phase 2 results below.**
Predictors that ignore the audio match the sensing headline on the
same per-room IoU metric (`scripts/eval_no_audio_baseline.py`, 500
held-out rooms, ± SE):

| no-audio predictor | v2 mixed | v1 rect |
| --- | --- | --- |
| prior map, best τ | 0.104 ± 0.003 | 0.090 ± 0.002 |
| fixed interior box | 0.103 ± 0.003 | 0.091 ± 0.002 |

For comparison, the v2 calibrated K=4 result is 0.100 and the v1 best
(K=8) is 0.092. The "information ceiling ≈ 0.10" below is therefore
better explained as convergence to the prior. It is not yet evidence
that the audio carries geometry the model extracts. Three more audit
findings matter:

- $p = 0$ walls are *pressure-release*, not rigid.
- The rooms are lossless.
- The simulation has no physical units.

Re-scoring against the exact archives and adding threshold-free metrics
is plan Task 3.4. Rigid walls and the verification suite are Phase 5.


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

Attribution (ablation completed 2026-07-17): the v1 architecture
retrained on v2 data reaches the same calibrated fused ceiling as the
skip model (0.101 vs 0.100 @ K=8). Decomposition: (e) ≈ +6 % and
equalises architectures; (b+c) ≈ +5 %; (a+d) ≈ 0 at the ceiling.
Two different architectures converging on ≈ 0.10 marks the
**information ceiling of the current acquisition** (2 mics, 64² grid,
λ ≥ 2.2 cells, K ≤ 8 poses) — further sensing gains require changing
the acquisition physics, not the model. Task 2.3 closed.

## 1. Demonstrations (added 2026-07-12)

Both phases are now demonstrable end to end (`docs/demos.md`):
`scripts/demo_room_mapping.py` maps a fresh room from the command line
(figure: truth + fused map sharpening over K=1..8 poses), and the web
UI gained an *Acoustic sensing* panel — draw a room, press **Sense
room**, get the Bayes-fused obstacle map in ~0.3 s while the live
simulation keeps streaming (`sense_room`/`sense_result` events, E2E
Playwright-verified). Shared engine: `learning/sensing.py`. Report:
`tests/reports/demos_2026_07_12.md`.

## 2. Resolution (Phase 1 + 2 close, 2026-07-11) — superseded in part by the 2026-09-24 audit

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

## 3. Assets

- Datasets (`data/training_data/`, gitignored, regenerable by seed;
  commands in `docs/learning.md`): single-pose train/held-out
  (1234/999), multi-pose 10k×4 train (31415) and 500×8 held-out
  (424242), passive randomized-source train/held-out (5678/8765).
- Checkpoints (`checkpoints/`, gitignored): `long_baseline` (verified
  reproduction of the 2026-05-14 run), `passive_baseline`,
  `joint_baseline` (best_iou.pt = epoch 36, the evaluated one).
- Models: `DualInputCNN` / `PassiveCNN` / `JointPoseCNN` behind
  `build_model`; checkpoints carry a `model_type` tag.

## 4. Operational notes

- Artifact convention: datasets → `data/training_data/`, checkpoints →
  `checkpoints/`, never `/tmp` (a temp cleanup destroyed the
  2026-05-14 artifacts; datasets were regenerated from seeds).
- Long training runs should be launched **detached**
  (`Start-Process`, logs to files): one 100-epoch joint run was lost
  at epoch 36 when a paused harness background task was reaped and the
  kill took the process tree. Seeded CPU reruns replay near-identically
  (the restart reproduced val IoU 0.0464 at epoch 36 exactly).

## 5. Open follow-ups

- Superseded by the 2026-09-24 plan: the Phase 3 debug audit comes
  next, then the Phase 4 UI overhaul. Beamforming is now Phase 7 and waits on the
  Phase 5 physics fixes (absorbing boundaries, materials, verification).
- Sensing-side upgrades, if needed later: variance-normalised or
  log-sum-exp pose pooling / variable-K training; GCC-PHAT (TDOA)
  input channel for passive; passive multi-pose.
- 3D obstacle drawing in the web UI remains deferred.
