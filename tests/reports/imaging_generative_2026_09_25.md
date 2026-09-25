# A generative model of whole room maps (Phase 6: 6.3.4) — 2026-09-25

**Headline.** A masked (absorbing-state) discrete diffusion over the 64×64
filled mask, conditioned on the four aligned physics images and the prior
(K = 4), was fine-tuned from the 6.3.1 U-Net in 15 minutes on two shared
cores. It draws coherent room hypotheses: solid blobs and bars rather than
salt-and-pepper noise (`examples.png`). Across 500 held-out rooms, with
nothing tuned on them, the result is mixed and mostly negative:

- **Coherence helps when the marginals are held fixed.** Take the same
  samples and shuffle each pixel independently across the 32 samples.
  This keeps every pixel's marginal exactly and destroys the dependence.
  The coherent set is better on every score that can see dependence:
  energy score −0.072 ± 0.004 (z = −20), variogram score −670 (z = −39),
  and oracle best-of-32 IoU +0.167 (z = 47).
- **Against the real baseline, the coherent samples lose on the proper
  scores.** The baseline draws independent Bernoullis from the U-Net's
  calibrated probabilities. The generative samples score worse on the
  energy score, +0.41 ± 0.03 (z = +12.5, lower is better). They are also
  worse on the Jaccard kernel score (z = +11) and the pixel CRPS (z = +12).
  They win only on the variogram score (−424, z = −32), which ignores the
  marginals, and on oracle best-of-N IoU (0.404 against 0.272 at N = 32).
  The cause is marginal quality, not coherence. The model's sample-mean
  map reaches IoU **0.311**, against the U-Net's **0.362** (−0.051 ± 0.004,
  z = −12). It gives 368 bits/room against 480, and its ECE is 0.022
  against 0.002. Its marginals are already worse before any sampling:
  with nothing visible, IoU is 0.302 and the information gain is 397 bits.
  Parallel decoding then drifts to 29 % too much occupancy.
- **Diversity tracks error, but no better than the U-Net's entropy.**
  Across rooms, sample diversity (1 − mean pairwise IoU) has a Spearman
  correlation of 0.58 with the error of the mean map. It correlates 0.51
  with the U-Net's error. The U-Net's own summed entropy correlates 0.54,
  and the diversity of its independent samples 0.61.

**Verdict: coherent samples do not beat independent Bernoulli sampling from
the U-Net here.** They beat independent samples *with the same marginals*,
by a wide margin. They lose overall because fine-tuning for coherence
cost more in marginal accuracy than coherence gains back. The 6.3.4 sampler
exists and is tested, but it is not yet a better uncertainty model than the
calibrated U-Net.

Artifacts: `imaging_generative_2026_09_25_artifacts/` (`results.json`,
`examples.png`, `examples_bias0.png`, `reliability.png`,
`diversity_vs_error.png`, `steps_ablation.png`). Maths: `docs/imaging.md` §9.

## What was built

New files only. The one change to an existing file is an append to
`docs/imaging.md`.

| file | content |
|---|---|
| `src/acoustic_system/imaging/generative.py` | `MaskedDiffusionUNet`, `warm_start_from_aligned`, the forward corruption and hidden-pixel loss, `fit_masked`, the cosine reveal schedule, `sample` (T-step parallel decoding, Rao-Blackwellised mean, optional logit bias), `independent_bernoulli`, `decorrelate`, and the scores (`energy_score`, `jaccard_kernel_score`, `variogram_score`, `pixel_crps`, `mean_pairwise_iou`, `best_of_n_iou`) |
| `scripts/eval_generative.py` | training, validation ablations, sampling, scoring, figures |
| `tests/imaging/test_generative.py` | 18 tests, about 3 s |

### Choice of model and why

The requirements were a joint distribution over 4096 binary pixels, a
budget of 1-2 CPU threads for about an hour, and a strong, calibrated
marginal predictor already available. The model chosen is a **few-step
absorbing-state (masked) diffusion**. It is an order-agnostic
autoregressive model that is decoded in T parallel blocks.

- It is exact in the limit of one pixel per step (the chain rule), and T
  trades speed against accuracy.
- The network is the 6.3.1 U-Net with two extra input channels (the visible
  value and the visibility mask). With nothing visible its job is the U-Net's
  job. The model is **warm-started** from the trained U-Net, with zero
  weights on the new channels, so training starts at the calibrated marginal
  and only has to learn the dependence. A unit test checks that the warm
  start reproduces the U-Net's logit exactly.
- The sample mean can be **Rao-Blackwellised**: average the probability
  each pixel was drawn with, not the bit. The two have the same expectation,
  the average has lower variance, and it never gives an exact 0 or 1.
- A conditional VAE with this strong conditioning tends to posterior
  collapse, and its factorised decoder still needs pixel-independent noise
  to sample. A fixed-order patch autoregression is the special case of this
  model with one order and no parallel decoding. It needs 64+ sequential
  passes and has an arbitrary raster order.

The model has 119.5 k parameters. It trained for 20 epochs (AdamW, lr 1e-3,
cosine, batch 16, D4 augmentation, visible fraction a ~ U(0, 1) per room),
in 907 s on 2 torch threads, on a machine shared with a 2-core training
job. The training rooms are 500-3999 (3500 rooms, K = 4), the same rooms
the U-Net was trained on.

## Protocol (no held-out tuning)

- **Validation rooms 0-499** (training archive): every IoU threshold τ is
  chosen there, with 32 samples per room for the sample-mean maps. On rooms
  0-99, with 16 samples each, two choices were fixed in advance by the
  lowest validation energy score: the number of decoding steps T ∈ {4, 8,
  16, 32}, and then a sampling logit bias b ∈ {0, −0.5, −1, −1.5}.
- **Held-out:** all 500 rooms of `active_sensing_v2_heldout_500x8.hdf5`,
  first 4 poses, 32 samples per room, scored once. Paired per-room
  differences, mean ± SE, z = mean/SE.
- **Baselines.** The U-Net probabilities are the cached `unet_s0` logits,
  which reproduce 0.362 exactly. From them come independent Bernoulli
  samples (N = 32) and the thresholded U-Net as a point forecast. The
  *decorrelated* control permutes each pixel's 32 values independently
  across samples. It keeps every pixel's empirical marginal exactly, so the
  pixel CRPS is identical by construction. Any score difference against it
  comes from the dependence alone.

## Validation ablations (rooms 0-99, 16 samples)

| T | energy score | variogram | mean pairwise IoU | mean sample area (true 291) | time |
|---|---|---|---|---|---|
| 4 | 10.17 | 1016 | 0.273 | 455 | 18 s |
| 8 | 10.03 | 902 | 0.255 | 443 | 35 s |
| 16 | 9.90 | 862 | 0.248 | 411 | 74 s |
| **32** | **9.81** | **850** | 0.232 | 392 | 144 s |

| logit bias (T = 32) | energy score | variogram | mean sample area (true 291) |
|---|---|---|---|
| 0 | 9.81 ± 0.33 | 850 | 392 |
| **−0.5** | **9.68 ± 0.38** | 822 | 225 |
| −1 | 10.41 | 826 | 119 |
| −1.5 | 11.71 | 844 | 57 |

More steps help steadily and had not saturated at T = 32, the edge of the
pre-declared grid. The samples over-fill (392 cells against 291), less so
with more steps, so part of the over-fill is parallel-decoding error. The
area is very sensitive to the bias: −0.5 already under-fills (225). The
chosen b = −0.5 is scored as a second variant. The main variant is b = 0.

## Held-out results (500 rooms, K = 4)

### Mean maps (sample quality and calibration)

| map | IoU | AP | BF | info gain (bits/room) | ECE | ΔIoU vs U-Net (z) | Δ bits vs U-Net (z) |
|---|---|---|---|---|---|---|---|
| no-audio prior | 0.101 | 0.128 | 0.119 | 0 | — | — | — |
| **U-Net (6.3.1)** | **0.362 ± 0.008** | **0.578** | **0.457** | **480 ± 10** | **0.0020** | — | — |
| generative, nothing visible (step-1 marginal) | 0.302 ± 0.007 | 0.511 | 0.383 | 397 ± 9 | 0.0068 | −0.060 ± 0.005 (−12.8) | −83 ± 5 (−16.2) |
| sample mean, Rao-Blackwellised, b = 0 | 0.311 ± 0.008 | 0.514 | 0.416 | 368 ± 12 | 0.0216 | −0.051 ± 0.004 (−11.6) | −112 ± 6 (−18.7) |
| sample mean, empirical (32 bits), b = 0 | 0.308 ± 0.008 | 0.508 | 0.419 | 289 ± 15 | 0.0231 | −0.054 ± 0.005 (−12.0) | −191 ± 10 (−19.5) |
| sample mean, Rao-Blackwellised, b = −0.5 | 0.311 ± 0.008 | 0.512 | 0.418 | 328 ± 12 | 0.0182 | −0.052 ± 0.004 (−11.9) | −152 ± 10 (−15.9) |
| sample mean, empirical, b = −0.5 | 0.307 ± 0.008 | 0.502 | 0.420 | 85 ± 24 | 0.0182 | −0.055 ± 0.004 (−12.7) | −395 ± 22 (−17.7) |

The sample-mean map is well above the prior (0.311 against 0.101) but
0.05 IoU below the U-Net. The reliability curves (`reliability.png`) show
two faults. At b = 0 the samples over-predict occupancy in the confident
bins: 29 % too much area. Cells the sample mean gives about 0.1 % are
occupied about 1 % of the time: the samples commit to a few hypotheses
and miss objects elsewhere. The bias moves the first fault but makes the
second worse. The Rao-Blackwellised mean is clearly better than the raw
32-sample average for information gain (368 against 289 bits), as
expected, since it has no hard zeros.

### Sample sets (N = 32 per room)

Lower is better for the four scores. Best-of-N is the oracle-selected
sample: N = 1 is the mean single-sample IoU, N = 8 averages the best of
four disjoint groups.

| sample set | energy score | Jaccard kernel score | variogram score | pixel CRPS | mean pairwise IoU | best-of-1 IoU | best-of-8 | best-of-32 |
|---|---|---|---|---|---|---|---|---|
| **generative, b = 0** | 9.53 ± 0.15 | 0.399 | **785** | 202.5 | 0.223 | 0.212 | **0.339** | **0.404** |
| generative, decorrelated (same marginals) | 9.60 ± 0.15 | 0.405 | 1456 | 202.5 | 0.219 | 0.205 | 0.226 | 0.237 |
| generative, b = −0.5 | 9.50 ± 0.17 | 0.397 | **772** | 202.0 | 0.169 | 0.188 | 0.326 | 0.394 |
| generative, b = −0.5, decorrelated | 9.53 ± 0.16 | 0.400 | 1068 | 202.0 | 0.166 | 0.183 | 0.209 | 0.222 |
| **U-Net, independent Bernoulli** | **9.12 ± 0.16** | **0.381** | 1210 | **188.1** | 0.229 | **0.234** | 0.260 | 0.272 |
| U-Net thresholded (point forecast) | 17.07 ± 0.30 | 0.638 | 1346 | 336.6 | — | 0.362 | — | — |

Paired differences, generative (b = 0) minus the other set (negative
favours the generative model on the four scores, positive favours it on
IoU):

| comparison | energy (z) | Jaccard (z) | variogram (z) | CRPS (z) | best-of-8 (z) | best-of-32 (z) |
|---|---|---|---|---|---|---|
| − decorrelated (dependence only) | −0.072 ± 0.004 (−20.1) | −0.0056 (−13.5) | −670 ± 17 (−39.3) | 0 (identical) | +0.113 (47.4) | +0.167 (47.0) |
| − U-Net independent Bernoulli | **+0.410 ± 0.033 (+12.5)** | +0.019 (+11.4) | **−424 ± 13 (−32.3)** | +14.4 (+11.6) | +0.079 (31.3) | +0.132 (40.7) |
| b = −0.5: − U-Net independent | +0.385 ± 0.035 (+11.2) | +0.016 (+10.1) | −437 (−32.3) | +13.9 (+11.4) | +0.066 (26.3) | +0.123 (37.1) |
| − U-Net point forecast | −7.54 ± 0.17 (−44.3) | −0.239 (−43.5) | −561 (−36.1) | −134 (−24.5) | | |

How to read this:

- The **pixel CRPS** depends on the marginals only (its expectation is the
  Brier score), so it measures the marginal gap: +14.4 against the U-Net.
- The **energy score** (Euclidean norm, strictly proper) sees both the
  marginals and the dependence. Its sensitivity to dependence is known to
  be weak. The dependence gain (−0.07) is about a sixth of the marginal
  loss (+0.41 net).
- The **variogram score** compares how often neighbouring pixels disagree.
  It sees the dependence and ignores the marginals, and there the
  generative model wins decisively. Independent samples have far too many
  disagreeing neighbours.
- **Best-of-N** favours coherent sets by construction: each sample is a
  complete hypothesis, while a salt-and-pepper sample never matches a solid
  object. Oracle best-of-32 (0.404) beats the thresholded U-Net (0.362).
  So among 32 samples there is usually one better than the U-Net's
  threshold map. But a typical single sample is poor (0.212). Without an
  oracle that recognises the good one, this is only a statement about
  coverage.

### Diversity

| quantity | b = 0 | b = −0.5 |
|---|---|---|
| mean pairwise IoU between samples (quartiles) | 0.223 (0.17 / 0.21 / 0.26) | 0.169 (0.11 / 0.15 / 0.21) |
| Spearman ρ: diversity vs mean-map error | 0.58 | 0.60 |
| Spearman ρ: diversity vs mean single-sample error | 0.72 | 0.80 |
| Spearman ρ: diversity vs U-Net error | 0.51 | 0.53 |
| *reference:* U-Net total entropy vs U-Net error | 0.54 | |
| *reference:* U-Net entropy per predicted cell vs U-Net error | 0.56 | |
| *reference:* diversity of independent U-Net samples vs U-Net error | 0.61 | |
| true area inside [min, max] of the 32 sample areas | 83 % | 77 % |

All p < 10⁻³⁰. The diversity terciles (b = 0) separate rooms well. The
least diverse third has mean-map IoU 0.436 (U-Net 0.487), the middle third
0.290 (0.328), and the most diverse third 0.207 (0.272)
(`diversity_vs_error.png`). So diversity is an informative per-room
confidence signal. But it adds nothing over the U-Net's entropy, which
predicts the U-Net's error as well or better. The samples are very
diverse: two samples of the same room overlap at IoU 0.22. A sample's IoU
with the truth is 0.21. In a calibrated set the truth looks like one more
sample, so the two numbers should match. They nearly do, with the truth
slightly further away: a mild over-confidence.

## Why the marginals degraded, and what would fix it

1. **The marginal with nothing visible got worse during fine-tuning**
   (0.362 → 0.302 IoU, 480 → 397 bits). The same 119 k-parameter network
   now also has to complete partial masks, and the hidden-pixel loss is
   spread over all visible fractions. Only a thin slice of training has
   a ≈ 0. Candidate fixes (untested): a larger network, a loss weight that
   favours small a, or a hybrid that draws the first block from the U-Net.
   The last is inconsistent as a joint model, but it keeps the calibrated
   marginal as the anchor.
2. **Sampling drift.** The model was trained on true masks with random
   pixels hidden (teacher forcing), and at sampling time it conditions on
   its own draws. Pixels revealed together are drawn independently. When
   two nearby positions are each 50 % likely, drawing both is accepted, and
   drawing neither is later filled in. That biases the area upwards, less
   with more steps (455 → 392 cells from T = 4 to 32). A development check
   on 20 validation rooms (not reproduced by the script) tried an adaptive
   order: reveal all near-certain pixels first, then random uncertain ones
   in blocks of 4-64. It was far *worse*, with an area of about 700 against
   318, because the visible pattern ("all of the empty space") never occurs
   in training. The conditionals are only reliable under the random-mask
   visibility they were trained on.
3. A global logit bias chosen on validation rooms fixes the mean area only
   crudely. It moves the whole distribution and trades over-filling for
   missed objects (bias −0.5: area 208 against 283; information gain
   328 bits against 368).

Cheaper routes to "coherent and calibrated" that would reuse this code:
more decoding steps (still improving at T = 32), a model trained on its own
partial samples to reduce exposure bias, or sampling-importance resampling
of the generative samples against the U-Net's marginals.

## Limitations

1. One training seed, one architecture, and 20 epochs set by the budget,
   not tuned.
2. The T grid ended at its best value (32), so the sampler is not
   converged. The bias grid was coarse, and the optimum is probably
   between −0.5 and 0.
3. The energy score is only weakly sensitive to dependence, the variogram
   score is proper but not strictly, and best-of-N is an oracle. The
   verdict rests on the strictly proper scores (energy, and pixel CRPS for
   the marginals), and all of them favour the U-Net baseline.
4. The same caveats as the earlier reports apply: simulation, a matched
   room generator, exact poses, and no noise.
5. About 25 paired tests are reported. Every sign-level conclusion above
   has |z| ≥ 7.

## Runtime (2 torch threads, NUMBA_NUM_THREADS = 1, machine shared with a 2-core training job)

| stage | wall time |
|---|---|
| fine-tuning, 20 epochs | 15 min |
| validation ablations (7 × 100 rooms × 16 samples) | 11 min |
| sampling, 500 rooms × 32 samples × T = 32 (per split and variant; 4 in total) | 24-26 min each |
| scoring and figures | ~6 min per variant |

The total was about 2 h of wall time, most of it sampling. One chain
costs T network passes, about 3 ms each in batches of 128.

## Reproduce

```
NUMBA_NUM_THREADS=1 uv run python scripts/eval_generative.py \
    --out-dir tests/reports/imaging_generative_2026_09_25_artifacts
NUMBA_NUM_THREADS=1 uv run pytest tests/imaging/test_generative.py   # 18 tests, ~3 s
```

The script needs the per-pose image caches and the `unet_s0` model and
logits from `scripts/eval_imaging_models.py` in `--cache-dir`. It caches
the model and all samples under `<cache-dir>/generative/`, so a re-run
only re-scores.
