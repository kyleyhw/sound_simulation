# Project Status as of 2026-07-09

## 1. Resolution

Phase 2 / Task 2.1.3 is **closed with a decisive negative result**, and
the repo is prepped for Task 2.1.4 (multi-pose aggregation).

Two training runs of the `DualInputCNN` completed on 2026-05-14:

| run | in-dist IoU | held-out IoU | report |
| --- | --- | --- | --- |
| baseline (10k samples, 100 epochs) | 0.134 | 0.037 | `tests/reports/training_2026_05_14.md` |
| + dropout 0.1 + sensor augmentation (60 epochs) | 0.088 | 0.030 | `tests/reports/training_2026_05_14_aug.md` |

The baseline learned a marginal prior over obstacle locations rather
than the conditional map; regularisation removed that strategy and
left nothing in its place (every metric got worse). Conclusion:
**single-pose 2-mic recordings are information-limited** — the failure
is data under-determination, not model capacity. The lever is
multi-pose aggregation (Task 2.1.4), consistent with the laptop-only
hardware constraint (move the laptop, don't add mics).

## 2. What changed this session (2026-07-09 loose-end sweep)

- **Artifact loss discovered and mostly repaired.** The 2026-05-14
  session wrote its datasets and checkpoints to `/tmp` (the Windows
  temp dir), and a temp cleanup destroyed them. The datasets were
  regenerated deterministically from their recorded seeds into
  `data/training_data/` (`active_sensing_train_10k.hdf5`, seed 1234;
  `active_sensing_heldout_500.hdf5`, seed 999 — exact flags recovered
  from the session transcript and recorded in `docs/learning.md`).
  The **trained checkpoints are not recoverable** and require a ~2 h
  CPU retrain (exact command in `docs/learning.md`) before the
  Bayesian-aggregation quick test can run.
- **New convention:** datasets → `data/training_data/`, checkpoints →
  `checkpoints/` (both gitignored; `*.pt` + `checkpoints/` rules
  added). Never write run artifacts to `/tmp`.
- **`docs/learning.md` written** — the learning package and the two
  active-sensing scripts previously had no `/docs` entry. Covers the
  inverse-problem formulation, dataset generation parameters and
  their rationale, the architecture, BCE+Dice/IoU math, training
  configuration, and results to date. Indexed from `docs/index.md`
  and `README.md`.
- **`PROJECT_PLAN.md` refreshed**: 2.1.3 marked completed with the
  negative result; 2.1.4 (multi-pose) added with sub-tasks — it was
  referenced by both training reports but never defined in the plan;
  bare `[ ]` tags normalised to `[pending]`.
- **README staleness fixed**: the "Phase 2 (in progress)" paragraph
  claimed the CNN was unimplemented; `environment.yml` (removed long
  ago) was still in the directory tree.
- **Gates re-verified green**: `check_simulate.py` (2D, max_abs
  7.8e-07), `check_simulate_3d.py` (max_abs 1.0e-07), and
  `tests/learning/test_model.py` all pass.

## 3. Open follow-ups

- **Retrain the baseline checkpoint** (blocks 2.1.4's quick test;
  needs explicit go-ahead per the plan's task-commencement rule):

  ```bash
  uv run python -m acoustic_system.learning.train \
      --dataset data/training_data/active_sensing_train_10k.hdf5 \
      --epochs 100 --batch-size 32 --lr 1e-3 --weight-decay 1e-4 \
      --scheduler cosine --target-size 64 \
      --ckpt-dir checkpoints/long_baseline --log-every 5 --seed 42
  ```

  Note: CPU-side nondeterminism means the retrained checkpoint will
  reproduce the baseline's behaviour statistically, not bit-exactly;
  re-verify held-out IoU ≈ 0.037 with `scripts/eval_and_report.py`
  before using it as the 2.1.4 reference point.
- **2.1.4 step 1**: extend `scripts/generate_active_sensing.py` to
  K poses per room; then the Bayesian (geometric-mean) aggregation
  test against the 0.037 held-out baseline. Details in
  `PROJECT_PLAN.md` and the end of `docs/learning.md`.
- The 3D web UI still lacks 3D obstacle drawing — only the generator
  script can author 3D scenes. Deferred until the ML side stabilises.
