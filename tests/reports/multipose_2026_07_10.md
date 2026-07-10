# Bayesian multi-pose aggregation — 2026-07-10

Task 2.1.4b: the decision-gate experiment recommended at the end of
[training_2026_05_14_aug.md](training_2026_05_14_aug.md). Question:
does fusing the single-pose model's predictions across K poses of the
same room lift held-out IoU — i.e. is the single-pose failure really
data under-determination rather than a modelling defect?

**Verdict: yes. The gate passes.** Prior-corrected Bayes fusion lifts
held-out IoU 2.4× (0.038 → 0.092) monotonically in K, with no
retraining. The joint-pose model (2.1.4c) is warranted.

## What

The retrained baseline checkpoint (`checkpoints/long_baseline/
best_iou.pt` — reproduction verified first: in-dist IoU 0.1340 vs the
lost original's 0.134, held-out 0.0374 vs 0.037) was run independently
on each of K ∈ {1, 2, 4, 8} poses of 500 held-out rooms
(`active_sensing_heldout_multipose_500x8.hdf5`, seed 424242: shared
obstacle mask + source per room, independent random (driver, mic-pair)
placements per pose). Per-pose probability maps were fused by three
rules (`scripts/eval_multipose.py`), thresholded at 0.5, and scored by
mean IoU.

With $\ell_k(x)$ the pose-$k$ logit map and $\hat\pi = 0.0582$ the mean
obstacle fraction (prior logit $-2.784$):

| rule | formula |
| --- | --- |
| geo | $\sigma\bigl(\tfrac1K \sum_k \ell_k\bigr)$ — logit average, prior-neutral |
| bayes | $\sigma\bigl(\sum_k \ell_k - (K{-}1)\,\mathrm{logit}\,\hat\pi\bigr)$ — product rule, prior-corrected |
| mix | $\tfrac1K \sum_k \sigma(\ell_k)$ — probability average (mixture) |

## Why

Single-pose training runs concluded the 2-mic recording cannot
constrain a 64×64 mask and that added information must come from more
poses, not more capacity. Fusion-at-inference is the cheapest possible
test of that claim: it adds pose information while changing *nothing*
about the model. If IoU rises with K, under-determination is confirmed
and a model that ingests K poses jointly should do at least as well.

## Results

Mean IoU @ 0.5 over 500 rooms:

| K | geo | bayes | mix |
| --- | --- | --- | --- |
| 1 | 0.0382 | 0.0382 | 0.0382 |
| 2 | 0.0189 | **0.0628** | 0.0189 |
| 4 | 0.0049 | **0.0803** | 0.0062 |
| 8 | 0.0009 | **0.0918** | 0.0015 |

(K=1 is the single-pose baseline on the same rooms; 0.0382 matches the
0.0374 measured on the separate seed-999 archive, as expected.)

Artifacts: `multipose_2026_07_10_artifacts/iou_vs_k.png` (IoU-vs-K per
rule; x is poses fused on a log-2 axis, y is mean IoU — the takeaway
is the bayes curve rising monotonically while geo/mix decay toward 0)
and `preds_multipose.png` (truth vs single-pose vs geo-fused
probability maps for 6 rooms).

## Interpretation

The three rules differ only in how they treat the model's baked-in
prior, and that explains the entire pattern:

- The single-pose model is **systematically under-confident about
  obstacles**: trained on rooms that are ~94 % empty, its per-pixel
  probabilities sit mostly below 0.5, and thresholded predictions are
  small blobs. Averaging (geo, mix) can only *shrink* what little mass
  clears the threshold — pose predictions disagree about location, so
  their average is everywhere-low and IoU collapses toward 0 as K
  grows. This is the expected failure of prior-neutral fusion under a
  strong shared prior, not evidence against multi-pose.
- The Bayes rule subtracts the prior's logit once per extra pose,
  exactly compensating the K-fold re-counting of the "rooms are empty"
  bias in the logit sum. What remains accumulates per-pose *evidence*,
  and IoU rises monotonically: 0.038 → 0.063 → 0.080 → 0.092. Still
  rising at K=8 with no sign of saturation.

Two conclusions:

1. **Under-determination confirmed, quantitatively.** Pose information
   composes: 8 mediocre views beat 1 by 2.4× with zero retraining.
   The information ceiling of the task is far above what single-pose
   training reaches.
2. **The joint-pose model should learn the fusion.** The scalar-prior
   product rule is the crudest possible aggregator (it assumes
   conditional independence and a spatially uniform prior); a learned
   permutation-invariant pooling over pose latents can exploit
   correlations between poses and a spatial prior. The Bayes curve is
   the floor the joint model has to beat.

## Test-data rationale

500 rooms × 8 poses, seed 424242 (disjoint from train 1234 and
held-out 999): 500 rooms bounds the IoU standard error near ±0.002 at
these values; K=8 allows all of {1, 2, 4, 8} as prefix subsets so every
K is scored on identical rooms; prefix subsetting also means K=1
doubles as an internal consistency check against the known baseline.

## Runtime

| stage | time |
| --- | --- |
| baseline verification eval (10k + 500) | ~3 min |
| multi-pose eval (500 rooms × 8 poses, 3 rules × 4 K) | 9.7 s |
| **total** | **≈ 3.5 min** |
