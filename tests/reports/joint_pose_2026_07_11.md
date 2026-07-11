# Joint-pose model — 2026-07-11

Task 2.1.4c: train `JointPoseCNN` (latent-space fusion of K poses,
end-to-end) and compare against the two floors set by
[multipose_2026_07_10.md](multipose_2026_07_10.md): the single-pose
baseline (held-out IoU 0.037) and Bayes fusion of single-pose
predictions (0.092 at K=8).

**Verdict: joint training beats single-pose training at every K
(+27 % at its training configuration), but the learned mean-pool
fusion does not beat explicit Bayes fusion. The best recipe found in
Phase 2 is a hybrid — the joint model's per-pose predictions combined
by the Bayes product rule: held-out IoU 0.0924 at K=8, the best
number of the campaign.**

## What

`JointPoseCNN` (232,337 params — identical to `DualInputCNN`; shared
sensor encoder over K poses, mean-pooled latents, same source branch
and decoder) trained on 10 000 rooms × 4 poses (seed 31415), 100
epochs, protocol otherwise identical to the single-pose baseline
(AdamW 1e-3 / wd 1e-4, cosine, batch 32, seed 42). Evaluated on the
500-room × 8-pose held-out archive (seed 424242) two ways:

1. **Native joint inference**: all K poses in one forward pass
   (mean-pooled latents), K ∈ {1, 2, 4, 8} by prefix subset.
2. **Hybrid**: the model run per-pose (K=1 input), the K probability
   maps fused by the rules of `scripts/eval_multipose.py`.

## Why

The 2.1.4b gate showed pose evidence composes under a prior-corrected
product rule. The joint model tests whether a *learned*,
permutation-invariant fusion (mean over pose latents) extracts more
than that fixed rule — and, since mean pooling is K-agnostic, whether
training at K=4 transfers to other K at inference.

## Results

Held-out mean IoU @ 0.5, 500 rooms (single-pose baseline for
reference: 0.038 on these rooms):

| K | joint, native (mean-pool) | joint per-pose + Bayes | single-pose + Bayes (2.1.4b) |
| --- | --- | --- | --- |
| 1 | **0.0557** | 0.0557 | 0.0382 |
| 2 | 0.0548 | 0.0760 | 0.0628 |
| 4 | 0.0472 | 0.0873 | 0.0803 |
| 8 | 0.0387 | **0.0924** | 0.0918 |

In-distribution (training archive, native K=4): 0.1130 — vs the
single-pose baseline's 0.134, i.e. a *smaller* memorisation gap
(0.113/0.047 ≈ 2.4× vs 0.134/0.037 ≈ 3.6×).

Training curves (`joint_pose_2026_07_11_artifacts/loss.png`; epochs on
x, loss / IoU / LR on y): best val IoU 0.0464 at epoch 36 (vs the
single-pose baseline's 0.042 at epoch 50), followed by the familiar
overfit slide (val IoU 0.033 by epoch 100). `best_iou.pt` (epoch 36)
is the evaluated checkpoint. A run-integrity note: an earlier launch
of this exact seeded configuration was killed at epoch 36; the restart
reproduced val IoU 0.0464 at epoch 36 — seeded CPU training replays
near-identically, which doubles as an incidental reproducibility check.

## Interpretation

Reading the table column by column:

1. **Joint training extracts real multi-pose signal.** Native K=4
   (its training configuration) scores 0.047 held-out, +27 % over the
   single-pose model's 0.037, at identical capacity and protocol. The
   K=1 column is the sharpest evidence: as a *single-pose* predictor
   the joint-trained encoder scores 0.056 vs 0.038 — training against
   a 4-pose-conditioned target regularises the encoder into features
   that generalise better even pose-by-pose.
2. **Mean-pool fusion degrades with K instead of improving** (0.056 →
   0.039 from K=1 to K=8). Mechanism: averaging K i.i.d. latents
   shrinks their variance by 1/K; the decoder, calibrated on K=4
   statistics, receives ever-smoother latents as K grows and its
   outputs collapse toward the prior mean — the same under-confidence
   failure the prior-neutral rules showed in 2.1.4b, now in latent
   space. K-agnostic *shapes* do not give K-agnostic *statistics*.
3. **Explicit Bayes fusion remains the best aggregator**, and it
   stacks with the better encoder: fusing the joint model's per-pose
   predictions reaches 0.0924, edging the single-pose hybrid (0.0918)
   and 2.4× the single-pose baseline. The learned mean is simply a
   worse combiner than the product rule that respects how independent
   evidence should accumulate.

**Recipe recommendation for Phase 4 integration:** run the
joint-trained encoder per pose and fuse with the Bayes product rule
($\sum_k \ell_k - (K-1)\,\mathrm{logit}\,\hat\pi$). Architectural
follow-ups if more accuracy is needed later: a sum/log-sum-exp pooling
with per-K normalisation (fixes the variance-shrinkage mismatch), or
training across variable K.

## Test-data rationale

Training rooms (seed 31415) and held-out rooms (seed 424242) are
disjoint RNG streams from each other and from all prior archives;
K=4 training balances information per room against room diversity at
fixed simulation budget (40k sims); the K ∈ {1, 2, 4, 8} eval ladder
uses prefix subsets so all K are scored on identical rooms and the
K=1 column is directly comparable to the 2.1.4b table.

## Runtime

| stage | time |
| --- | --- |
| dataset generation (10k rooms × 4 poses) | 353 s |
| training (100 epochs, CPU; excludes a killed first attempt) | 6618 s (≈ 1 h 50 min, ~66 s/epoch) |
| native K-sweep eval (500 rooms × 4 K) | 8.5 s |
| hybrid fusion eval (500 rooms × 8 poses × 3 rules) | ~10 s |
| in-dist eval (10k rooms) | ~4 min |
| **total** | **≈ 2 h 5 min** |
