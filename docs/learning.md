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

## References

<span id="ref-milletari-2016">[1]</span> Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV 2016. [Link](https://arxiv.org/abs/1606.04797)

<span id="ref-loshchilov-2019">[2]</span> Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019. [Link](https://arxiv.org/abs/1711.05101)

<span id="ref-loshchilov-2017">[3]</span> Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017. [Link](https://arxiv.org/abs/1608.03983)
