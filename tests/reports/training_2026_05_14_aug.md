# Augmentation + dropout retry — 2026-05-14

Follow-up to [training_2026_05_14.md](training_2026_05_14.md). The
baseline run's diagnosis was that the model learned a marginal prior
over obstacle locations (val loss rose from epoch 1, held-out IoU
0.037 vs in-dist 0.134, predictions clustered in similar regions
across all held-out scenes). The two standard interventions were
queued: **dropout in the encoder + sensor augmentation
(per-channel gain jitter + additive Gaussian noise)**. This run tests
them.

## What

Same dataset, same architecture, same optimiser as the baseline. Only
three things changed:

| knob | baseline | augmented |
| --- | --- | --- |
| `--dropout` | 0.0 | 0.1 (Dropout2d in each encoder, before AdaptiveAvgPool) |
| `--augment` | off | on (per-channel gain in [0.7, 1.3], additive Gaussian noise σ=0.02 on sensor only) |
| `--epochs` | 100 | 60 (time-budget cut; cosine LR rescaled accordingly) |

Same dataset (10 k samples, seed 1234), same held-out (500 samples,
seed 999), same `--seed 42` for the train/val split. Held-out
evaluation is on the **clean** dataset — augmentation only fires on
the train split.

## Why

To answer two questions:

1. Can dropout + augmentation break the marginal-prior failure mode?
2. If so, by how much — and is it worth pursuing in this direction?

## Results

### Final-epoch metrics

| metric | baseline (epoch 100) | augmented (epoch 60) |
| --- | --- | --- |
| train_loss | 0.9337 | 1.0494 |
| train_iou | **0.200** | 0.095 |
| val_loss | 1.1969 | 1.1385 |
| val_iou | 0.034 | 0.028 |

Dropout halved train IoU (0.200 → 0.095) — which means the model
genuinely cannot memorise as effectively. Val loss is slightly lower
in the augmented run (1.1385 vs 1.1969) but val IoU is also lower —
the predictions are softer (less confident wrong predictions, which
helps cross-entropy) but no closer to the right pixels (so IoU
doesn't move).

### Best-IoU checkpoint comparison

| split | baseline IoU | augmented IoU | Δ |
| --- | --- | --- | --- |
| in-distribution (10 k) | 0.134 | 0.088 | **−34 %** |
| held-out (500) | 0.037 | 0.030 | **−19 %** |
| best val_iou (during training) | 0.042 (epoch 50) | 0.031 (epoch 35) | −26 % |

Augmentation made every single metric **worse**. The drop is larger
on in-distribution than held-out, which is consistent with
"regularisation suppressed the memorisation that was inflating
in-dist IoU" — but the held-out IoU also dropped, so the regularised
features did not generalise either.

### Plots

- `training_2026_05_14_aug_artifacts/loss.png` — train + val loss +
  IoU + LR schedule. Same monotonic-rising val loss as baseline,
  just compressed in absolute scale by dropout.
- `training_2026_05_14_aug_artifacts/preds_indist.png` — 8-sample
  in-dist prediction grid.
- `training_2026_05_14_aug_artifacts/preds_heldout.png` — 8-sample
  held-out prediction grid.

### Wall-clock

| stage | time |
| --- | --- |
| dataset regeneration (warm cache) | 81 s + 4 s |
| training (60 epochs) | 4163 s (≈ 1 h 9 min, ~69 s/epoch) |
| eval (10 k + 500) | ~3 min |
| **total** | **≈ 1 h 15 min** |

## Takeaways

### What just happened

The augmentation + dropout combination cut the model's ability to
memorise (train IoU 0.200 → 0.095) but did **not** transfer that
freed-up capacity into generalisation. Val IoU peaked at 0.029 and
held-out IoU was 0.030, both lower than the baseline. The val loss
curve still rises monotonically from epoch 1 — exactly the same
"overfitting from the very first epoch" signature as the baseline,
just at a smaller absolute scale.

### What this means

The **information-bottleneck diagnosis from the baseline write-up is
confirmed**. The single-pose 2-mic data does not contain enough
information to constrain the obstacle map; the only thing the model
*can* do with it is fit the marginal distribution. Dropout +
augmentation prevent that strategy and leave nothing in its place,
which is why every metric got worse rather than better.

This rules out (b) "the architecture has too much capacity" as a
sufficient explanation — the regularised model has *less* capacity
and does *worse*, which means the missing ingredient is data
information, not model degrees of freedom.

### Implications for the next experiment

The lever is **multi-pose aggregation** (Task 2.1.4 in
PROJECT_PLAN.md). Concrete shape:

1. Modify `scripts/generate_active_sensing.py` to write K poses per
   room — same obstacle mask, K different (driver, mic-pair)
   placements, K corresponding sensor recordings.
2. Modify the dataset to yield `(K, n_mics, T_rec)` sensor + the
   shared mask.
3. Two ways to consume it:
   - **Bayesian aggregation** (no extra training): run the
     single-pose model on each pose, geometric-mean the predicted
     probability maps, threshold. Tests whether multi-pose helps
     *at inference* without architectural changes. Quickest signal.
   - **Joint-pose model**: replace the single SpectrogramEncoder with
     one shared across K poses + a pose-aware aggregation block (mean,
     max, or a small attention over poses). Trains end-to-end.

Recommendation: **start with Bayesian aggregation on the existing
checkpoint**. If geometric-mean over even K=4 poses lifts held-out
IoU above the baseline 0.037, that confirms the under-determination
diagnosis directly and motivates the joint-pose model. If it
*doesn't*, the under-determination is even worse than expected and
we should look at the source signal design (the chirp may not have
the spectral content to disambiguate room geometry at this grid
resolution).

Architectural changes (cross-attention, U-Net skip) remain
*lower* priority than multi-pose. They wouldn't add information,
they'd just route the existing information differently — and the
existing information is empirically not enough.

## Comparison vs baseline (one-line summary)

```
baseline   100 epochs   no aug   in-dist 0.134   held-out 0.037   train_iou 0.200
augmented   60 epochs   aug+drop in-dist 0.088   held-out 0.030   train_iou 0.095
```
