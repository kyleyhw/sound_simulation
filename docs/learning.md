# `learning/` — Active-Sensing Obstacle-Mask CNN (Phase 2)

Documentation for the `src/acoustic_system/learning/` package and its companion
scripts `scripts/generate_active_sensing.py` and `scripts/eval_and_report.py`.
This is the Phase 2 / Task 2.1 pipeline: learn the inverse map from a stereo
acoustic recording to the room's obstacle occupancy mask.

## 1. Problem statement

Each training sample is a triplet $(s(t),\, u(t),\, M)$:

- $u(t)$ — the known source signal driven into the room at a random free cell
  (a linear chirp by default; a WAV file if an audio corpus is supplied).
- $s(t) \in \mathbb{R}^{T_\mathrm{rec} \times n_\mathrm{mics}}$ — the pressure
  recorded at a randomly placed, randomly oriented 2-microphone pair
  (the stock-laptop hardware constraint: 2–3 channels, no arrays).
- $M \in \{0,1\}^{H \times W}$ — the ground-truth obstacle mask of the room.

The model estimates $\hat{M} = f_\theta(s, u)$; this is a learned approximation
to an inverse scattering problem. The forward map $M \mapsto s$ is computed
exactly (to discretisation error) by the FDTD engine, which is what makes
cheap supervised data generation possible.

## 2. Dataset generation

`scripts/generate_active_sensing.py` builds one random rectangular-obstacle
room per sample, places a driver and a mic pair in free cells, runs the FDTD
simulation, and writes the triplet to HDF5 (`save_type = "active_sensing"`,
channel-last `sensor` dataset of shape `(T_rec, n_mics)`, per-sample attrs
sufficient to reproduce the scene).

Canonical commands (these regenerate the exact archives used by the
2026-05-14 training reports — generation is deterministic given the seed):

```bash
# 10k-sample training set (~100 s)
uv run python scripts/generate_active_sensing.py \
    --output data/training_data/active_sensing_train_10k.hdf5 \
    --num-samples 10000 --grid 64 --duration 200 \
    --n-obstacles 3 --obstacle-min 4 --obstacle-max 14 \
    --n-mics 2 --mic-spacing 12 --seed 1234

# 500-sample held-out set, different seed (~5 s)
uv run python scripts/generate_active_sensing.py \
    --output data/training_data/active_sensing_heldout_500.hdf5 \
    --num-samples 500 --grid 64 --duration 200 \
    --n-obstacles 3 --obstacle-min 4 --obstacle-max 14 \
    --n-mics 2 --mic-spacing 12 --seed 999
```

Parameter rationale:

| parameter | value | why |
| --- | --- | --- |
| `--grid 64` | 64×64 cells | matches the CNN's 64×64 output mask 1:1 and keeps 10k-sample generation ≈100 s on CPU |
| `--duration 200` | 200 FDTD steps | with $\Delta t = \kappa \Delta x / c = 0.5$ (courant 0.5, $c=1$, $\Delta x=1$), this is 100 sim-time units — enough for several wall reflections to reach the mics on a 64-cell room |
| chirp 0.02→0.4 | linear sweep | broadband excitation; the upper edge 0.4 stays well below the simulation Nyquist $1/(2\Delta t) = 1$ |
| `--mic-spacing 12` | 12 cells | scaled analogue of a laptop's built-in stereo-mic baseline; random orientation per sample so the model cannot assume a fixed array geometry |
| `--n-obstacles 3`, sizes 4–14 | | rooms 5–15 % occupied — sparse enough that the empty-room prior is nontrivial to beat, dense enough that reflections carry geometry information |
| seeds 1234 / 999 | train / held-out | disjoint RNG streams make the held-out set a true distribution-level generalisation test, not a random split of the same draw |

### Multi-pose archives (Task 2.1.4)

`--poses-per-room K` records K independent (driver, mic-pair) poses in
each room — the physical picture is one laptop carried to K spots,
chirping and recording at each. The room's obstacle mask and source
audio are shared; `sensor` gains a leading pose axis `(K, T_rec,
n_mics)` and the position attrs become plural (`driver_positions`
`(K, dims)`, `sensor_positions` `(K, n_mics, dims)`). `K = 1` (the
default) writes the original single-pose layout byte-for-byte —
verified against pre-extension archives — so all single-pose commands
above are unaffected. `ActiveSensingDataset` flattens multi-pose
archives pose-major (`index = room · K + pose`), so the single-pose
model trains and evaluates on them unchanged.

The multi-pose held-out set used by the aggregation experiment:

```bash
uv run python scripts/generate_active_sensing.py \
    --output data/training_data/active_sensing_heldout_multipose_500x8.hdf5 \
    --num-samples 500 --poses-per-room 8 --grid 64 --duration 200 \
    --n-obstacles 3 --obstacle-min 4 --obstacle-max 14 \
    --n-mics 2 --mic-spacing 12 --seed 424242
```

K = 8 lets one archive score every $K \in \{1, 2, 4, 8\}$ by prefix
subsets of the pose axis (K = 1 doubles as the single-pose baseline on
the same rooms); seed 424242 is disjoint from both the training stream
(1234) and the single-pose held-out stream (999).

`scripts/eval_multipose.py` fuses per-pose predictions from a
single-pose checkpoint without retraining, comparing three rules:
geometric-mean pooling $\sigma(\frac{1}{K}\sum_k \ell_k)$ (logit
averaging, the report's recommendation), the Bayes product rule
$\sigma(\sum_k \ell_k - (K-1)\,\mathrm{logit}\,\hat\pi)$ with
$\hat\pi$ the mean obstacle fraction, and arithmetic-mean (mixture)
pooling. See the script docstring for the derivations and caveats.

**Decision-gate outcome
([multipose_2026_07_10.md](../tests/reports/multipose_2026_07_10.md)):
the gate passed.** Bayes fusion lifted held-out IoU 0.038 → 0.092
monotonically in K (2.4× at K=8, unsaturated); the prior-neutral rules
collapsed because the single-pose model is systematically
under-confident about obstacles, which only the prior-corrected rule
compensates. This quantitatively confirmed under-determination and
motivated the joint-pose model.

### Joint-pose model (Task 2.1.4c)

`JointPoseCNN` moves the fusion into latent space and trains it end to
end: the sensor encoder is shared across the K poses (placements are
i.i.d., so pose index carries no meaning), the per-pose latents are
**mean-pooled** — permutation-invariant and K-agnostic, so a model
trained at K=4 evaluates at any K — and the pooled latent is
concatenated with the source latent and decoded as before. Parameter
count is identical to `DualInputCNN` (232,337), making single-pose vs
joint-pose a controlled comparison in which only the pose information
differs. Training consumes room-level samples
(`ActiveSensingDataset(..., flatten_poses=False)`, sensor
`(K, n_mics, T_rec)`):

```bash
uv run python scripts/generate_active_sensing.py \
    --output data/training_data/active_sensing_train_multipose_10kx4.hdf5 \
    --num-samples 10000 --poses-per-room 4 --grid 64 --duration 200 \
    --n-obstacles 3 --obstacle-min 4 --obstacle-max 14 \
    --n-mics 2 --mic-spacing 12 --seed 31415

uv run python -m acoustic_system.learning.train \
    --dataset data/training_data/active_sensing_train_multipose_10kx4.hdf5 \
    --model joint --epochs 100 --batch-size 32 --lr 1e-3 \
    --weight-decay 1e-4 --scheduler cosine --target-size 64 \
    --ckpt-dir checkpoints/joint_baseline --log-every 5 --seed 42
```

**Outcome
([joint_pose_2026_07_11.md](../tests/reports/joint_pose_2026_07_11.md)):**
joint training beats single-pose training at every K (held-out 0.047
vs 0.037 at K=4; as a single-pose predictor, 0.056 vs 0.038 — the
multi-pose target regularises the encoder), but mean-pool fusion
*degrades* with K (latent variance shrinks as 1/K, pushing the decoder
toward the prior — K-agnostic shapes are not K-agnostic statistics).
Explicit Bayes fusion remains the best aggregator and stacks with the
better encoder. **Best Phase 2 recipe: run the joint-trained model
per pose and fuse with the Bayes product rule — held-out IoU 0.0924
at K=8**, 2.4× the single-pose baseline.

**Artifact persistence convention.** Datasets and checkpoints live under
`data/training_data/` and `checkpoints/` (both gitignored — `*.hdf5`, `*.pt`).
Do **not** write them to `/tmp`: the 2026-05-14 run's artifacts were written to
the Windows temp directory and were lost to a temp cleanup; the datasets were
regenerated from their seeds on 2026-07-09, but the trained checkpoints require
a ~2 h retrain to recover.

## 3. The model — `model.py`

`DualInputCNN` (232,337 parameters, sized for CPU training):

```
sensor (B, n_mics, T_rec) ── STFT(64/16)  ── log1p ── Encoder_s ──┐
                                                                  ├─ concat ─ 1×1 mix ─ Decoder ─ logits (B, 1, 64, 64)
source (B, 1,      T_aud) ── STFT(512/256) ─ log1p ── Encoder_u ──┘
```

- **STFT front-end**: room responses are most naturally read in the
  time-frequency plane (modal structure, direct-vs-reverberant separation);
  a 2D conv on a log-magnitude spectrogram sees this directly instead of
  spending capacity rediscovering the FFT. The two branches use different
  window/hop (sensor 64/16, source 512/256) because $T_\mathrm{aud} \gg
  T_\mathrm{rec}$; each branch's `AdaptiveAvgPool2d(8)` normalises both to a
  `(B, 64, 8, 8)` latent regardless of spectrogram shape.
- **Concat fusion**: the sensor/source comparison is dominated by
  time-frequency alignment, which channel-concat + a 1×1 mix can express;
  cross-attention is the designated upgrade path if concat plateaus, held
  back deliberately to keep the parameter count CPU-friendly.
- **Decoder**: three transposed-conv doubling stages, 8 → 64, emitting raw
  logits.
- Optional `Dropout2d` in each encoder (used at $p = 0.1$ in the augmentation
  retry; channel dropout, because adjacent pixels of a conv feature map are
  too correlated for per-element dropout to regularise).

## 4. Loss and metric — `losses.py`

Training minimises BCE + soft Dice [[1]](#ref-milletari-2016). With
$\hat{y} = \sigma(\ell)$ the per-pixel probability and $y \in \{0,1\}$ the
target:

$$ \mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_i \bigl[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \bigr] $$

$$ \mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i \hat{y}_i y_i + \epsilon}{\sum_i \hat{y}_i + \sum_i y_i + \epsilon}, \qquad \mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda \mathcal{L}_{\text{Dice}} \;\; (\lambda = 1) $$

BCE keeps per-pixel gradients alive when overlap is near zero; Dice counters
the class imbalance of mostly-empty rooms (5–15 % obstacle fraction), where a
constant "no obstacle" prediction would otherwise score well. The Dice term is
computed per sample and averaged so every scene contributes equally, and
$\epsilon = 10^{-6}$ handles the all-empty 0/0 case.

The evaluation metric is mean IoU at threshold 0.5:
$\mathrm{IoU} = |P \cap T| / |P \cup T|$ with $P = [\sigma(\ell) > 0.5]$.

## 5. Training and evaluation — `train.py`, `eval.py`

Optimiser: AdamW (decoupled weight decay [[2]](#ref-loshchilov-2019)),
$\mathrm{lr} = 10^{-3}$, weight decay $10^{-4}$. Schedule: cosine annealing
[[3]](#ref-loshchilov-2017),

$$ \eta_t = \eta_{\min} + \tfrac{1}{2} (\eta_{\max} - \eta_{\min}) \bigl(1 + \cos(\pi t / T)\bigr), $$

decaying from $10^{-3}$ to ≈0 over the run. Three checkpoints are written:
`best.pt` (lowest val loss), `best_iou.pt` (highest val IoU — the one used for
reporting), `final.pt`.

Canonical training command (the 2026-05-14 baseline configuration, ~2 h for
100 epochs on CPU):

```bash
uv run python -m acoustic_system.learning.train \
    --dataset data/training_data/active_sensing_train_10k.hdf5 \
    --epochs 100 --batch-size 32 --lr 1e-3 --weight-decay 1e-4 \
    --scheduler cosine --target-size 64 \
    --ckpt-dir checkpoints/long_baseline --log-every 5 --seed 42
```

`scripts/eval_and_report.py` evaluates a checkpoint on the in-distribution and
held-out archives and writes the loss/IoU curves and prediction grids used in
the test reports:

```bash
uv run python scripts/eval_and_report.py \
    --checkpoint checkpoints/long_baseline/best_iou.pt \
    --indist data/training_data/active_sensing_train_10k.hdf5 \
    --heldout data/training_data/active_sensing_heldout_500.hdf5 \
    --output-dir tests/reports/<report>_artifacts
```

In the loss plots these produce, the axes are epoch (x) against loss / IoU /
learning rate (y, one panel each); the pattern to watch is the train–val gap:
a val loss that rises from epoch 1 while train loss falls (the signature in
both 2026-05-14 runs) means the model is memorising the training rooms from
the outset rather than learning the inverse map.

## 6. Results to date and status

| run | report | in-dist IoU | held-out IoU |
| --- | --- | --- | --- |
| baseline, 100 epochs | [training_2026_05_14.md](../tests/reports/training_2026_05_14.md) | 0.134 | 0.037 |
| + dropout 0.1 + sensor augmentation, 60 epochs | [training_2026_05_14_aug.md](../tests/reports/training_2026_05_14_aug.md) | 0.088 | 0.030 |

The regularisation retry made every metric worse: it removed the
memorisation strategy without anything to replace it. Conclusion (Task 2.1.3,
closed): **single-pose 2-mic data does not contain enough information to
constrain a 64×64 obstacle mask** — the failure is under-determination, not
model capacity. The next lever is multi-pose aggregation (Task 2.1.4 in
`PROJECT_PLAN.md`): $K$ chirp/record poses per room, fused either by
geometric-mean Bayesian aggregation of single-pose predictions or by a shared
encoder with a pose-aggregation block.

## 7. Passive sensing (Task 2.2)

`PassiveCNN` is the blind-deconvolution counterpart: infer the mask
from the stereo recording alone, with the source $u(t)$ unknown.
Architecturally it is `DualInputCNN` minus the source branch — the same
`SpectrogramEncoder` on the sensor spectrogram feeding the same
`MaskDecoder` (198,529 vs 232,337 parameters). Every other
hyperparameter is identical, so active-vs-passive is a controlled
comparison isolating the value of source knowledge.

Passive datasets must randomise the source per sample
(`--randomize-source`: chirp endpoints $f_0 \sim U[0.01, 0.10]$,
$f_1 \sim U[0.20, 0.45]$) — with the fixed default chirp a "passive"
model could memorise the constant source and the blind setting would
be a sham:

```bash
uv run python scripts/generate_active_sensing.py \
    --output data/training_data/passive_train_10k.hdf5 \
    --num-samples 10000 --randomize-source --grid 64 --duration 200 \
    --n-obstacles 3 --obstacle-min 4 --obstacle-max 14 \
    --n-mics 2 --mic-spacing 12 --seed 5678
# held-out: --output ...passive_heldout_500.hdf5 --num-samples 500 --seed 8765

uv run python -m acoustic_system.learning.train \
    --dataset data/training_data/passive_train_10k.hdf5 \
    --model passive --epochs 100 --batch-size 32 --lr 1e-3 \
    --weight-decay 1e-4 --scheduler cosine --target-size 64 \
    --ckpt-dir checkpoints/passive_baseline --log-every 5 --seed 42
```

Checkpoints carry a `model_type` tag (`dual` / `passive`; absent means
`dual`) which `eval.py` and `eval_multipose.py` use to rebuild the
right class.

Known limitation (shared with the active model): the power
spectrogram front-end discards phase, hence inter-channel time
differences (TDOA) — the strongest passive geometric cue. A
GCC-PHAT-style cross-correlation input channel is the designated
upgrade if the magnitude-only passive model proves too weak.

## 8. Sensing-quality improvements (Task 2.3, a–e)

Five upgrades motivated by the demo 8 diagnosis (predictions weak
because information is discarded, distributions are narrow, and the
fusion rule amplifies miscalibration):

- **(a) Inter-channel phase**: `StereoPhaseFrontEnd` feeds the encoder
  $[\log(1{+}|X_1|^2), \log(1{+}|X_2|^2), \cos\varphi, \sin\varphi]$
  with $\varphi = \arg(X_1 X_2^*)$ — the GCC-PHAT cross-spectrum
  phase, restoring the TDOA cue that power spectrograms discard. A
  test gate proves sensitivity: swapping the mics conjugates
  $\varphi$ and must change the output.
- **(b) Chirp band, corrected analysis**: the binding ceiling is the
  *spatial* Nyquist $f_{\max} = c/2\Delta x = 0.5$, not the temporal
  one (1.0) — wavelengths under $2\Delta x$ cannot propagate on the
  grid. v2 protocol: $f_{\mathrm{end}} = 0.45$ (λ ≈ 2.2 cells) plus a
  doubled 400-step recording window (~3 domain crossings), which adds
  more information than further bandwidth could.
- **(c) Shape-diverse rooms**: `generate_diverse_obstacles` — mixture
  of rectangles, discs, thin walls (incl. diagonal), and L-shapes,
  count $U[1,6]$ (`--room-style mixed`). v2 archives:
  `active_sensing_v2_train_10kx4.hdf5` (seed 271828, 7.0 % mean
  occupancy) and `..._v2_heldout_500x8.hdf5` (seed 161803). The full
  acquisition protocol is written as file attrs and copied into every
  checkpoint, so live inference (`sensing.py`) reproduces it
  automatically per checkpoint.
- **(d) Multi-scale skips**: `SkipSensingCNN` (798,641 params) pools
  the encoder's three stages to 8/16/32 and concatenates them into the
  corresponding decoder stages — multi-resolution conditioning that
  widens the old single-8×8-latent bottleneck (there is no geometric
  correspondence between time-frequency and room pixels to exploit,
  so these are information skips, not U-Net alignment skips).
- **(e) Calibration + operating point**: Platt scaling
  $\ell' = \ell/T + b$ fitted on the *validation* split
  (`scripts/fit_calibration.py`), stored as `calibration.json` with a
  val-selected IoU-optimal decision threshold. Mathematically, scalar
  calibration is an affine monotone map of the Bayes-fused logit at
  fixed K, so it cannot change ranking metrics — its value is honest
  probabilities plus the principled threshold.

### Results (v2 held-out, 500 shape-diverse rooms)

| recipe | K=1 | K=4 | K=8 |
| --- | --- | --- | --- |
| v1 joint model, Bayes @ 0.5 (transfer) | 0.052 | 0.085 | 0.094 |
| skip_v2, Bayes @ 0.5 (raw) | 0.054 | 0.087 | 0.090 |
| **skip_v2, calibrated Bayes @ τ=0.12 (val-selected)** | 0.066 | **0.100** | 0.100 |
| skip_v2, native joint forward @ 0.5 | 0.054 | 0.073 | 0.073 |
| oracle-threshold ceiling: skip_v2 / v1 | 0.093 / 0.093 | 0.100 / 0.094 | 0.100 / 0.096 |

Two findings beyond the headline (+7 % over the v1 recipe at the
project's new best of 0.100, saturating at K≈4):

1. **The fixed 0.5 threshold was doing hidden work in all earlier
   experiments.** At per-K oracle thresholds even K=1 reaches ~0.093,
   so most of the apparent "2.4× fusion gain" in the 2.1.4b/c reports
   was the fused map's distribution shifting relative to a fixed
   threshold; true evidence accumulation is ≈ +8 %. Fusion still
   helps — and the calibrated operating point now captures nearly the
   whole oracle ceiling without test-set leakage.
2. **v1 transfers robustly**: the old model scores essentially the
   same on the new shape-diverse/wider-band archive (0.094 @ K=8) as
   on its own distribution (0.092) — the v1 encoder was not brittle to
   these shifts; it was information-starved.
3. **Attribution (ablation: v1 architecture retrained on v2 data)**:
   at the calibrated operating point the v1 architecture reaches the
   *same* fused ceiling as the skip model (0.101 vs 0.100 @ K=8).
   Decomposition: (e) calibration + operating point ≈ +6 % and
   equalises architectures; (b+c) data/protocol ≈ +5 %; (a+d)
   architecture ≈ 0 at the ceiling (it improves uncalibrated maps and
   single-pose quality only). Two structurally different networks
   converging on ≈ 0.10 indicates the **information ceiling of the
   acquisition** (2 mics, 64² grid, λ ≥ 2.2 cells, K ≤ 8) — further
   gains require changing the acquisition physics, not the model.
   Full table in `tests/reports/sensing_v2_2026_07_15.md`.

Training the skip model: same command shape as before with
`--model skip` on the v2 archive (60 epochs — historical runs peaked
by epoch ~36); then `scripts/fit_calibration.py` for the sidecar. The
web UI and demo default to `checkpoints/skip_v2/best_iou.pt`.

## References

<span id="ref-milletari-2016">[1]</span> Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV 2016. [Link](https://arxiv.org/abs/1606.04797)

<span id="ref-loshchilov-2019">[2]</span> Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019. [Link](https://arxiv.org/abs/1711.05101)

<span id="ref-loshchilov-2017">[3]</span> Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017. [Link](https://arxiv.org/abs/1608.03983)
