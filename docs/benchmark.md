# Room-sensing benchmark

*Plan 10.10: a public dataset and benchmark, with baselines.*

The task: from simulated laptop recordings, predict where the obstacles are
in a 64 × 64 room. The benchmark exists so that every method is measured
against the same thing. That is the **no-audio baseline**: a map that
never listens and knows only where obstacles usually are. The project's
first CNNs scored exactly at that baseline, and the benchmark is designed
to make that visible immediately.

## Data

| split | archive | rooms × poses | seed |
|---|---|---|---|
| train | `active_sensing_v2_train_10kx4.hdf5` | 10,000 × 4 | 271828 |
| held-out | `active_sensing_v2_heldout_500x8.hdf5` | 500 × 8 | 161803 |

- **Grid:** 64², c = Δx = 1, Courant 0.5.
- **Walls:** p = 0 (pressure-release) outer walls and obstacles, drawn
  from the "mixed" family (rectangles, discs, thin walls, L-shapes).
- **Acquisition:** each pose is a driver cell and a mic pair 12 cells
  apart. The source is a linear chirp from 0.02 to 0.45 cycles per time
  unit, and the recording lasts 400 steps.
- **Layout:** each HDF5 group holds `sensor` (K, 400, 2), `source`,
  `obstacles` (64, 64), and the pose positions as attributes.

**Getting the data.** The archives are not hosted, because a storage
target has not been authorised yet. They regenerate deterministically
instead: `data/MANIFEST.json` lists the exact command for each one and a
content digest that ignores HDF5 timestamps. Regenerating the held-out
archive reproduces its digest exactly.

```bash
uv run python scripts/generate_active_sensing.py --output data/training_data/active_sensing_v2_heldout_500x8.hdf5 \
  --num-samples 500 --grid 64 --duration 400 --record-step 1 --poses-per-room 8 --n-mics 2 --mic-spacing 12 \
  --room-style mixed --n-obstacles 3 --obstacle-min 4 --obstacle-max 14 --synth-f-start 0.02 --synth-f-end 0.45 \
  --protocol v2 --seed 161803 --workers 4
uv run python scripts/manifest.py verify
```

## Rules

1. Tune everything on training rooms (thresholds, fusion weights,
   hyperparameters). Never look at held-out masks.
2. Report how many poses per room you use. K = 4 matches training; the
   held-out archive has 8.
3. Submit `prob` (500, 64, 64) in the held-out archive's room order, a
   `threshold` chosen on training rooms, and `poses`.

## Scoring

```bash
uv run python scripts/benchmark.py baseline --out prior.npz   # the no-audio baseline as a submission
uv run python scripts/benchmark.py score my_method.npz --json my_method.json
```

Each metric is reported as mean ± SE over rooms, together with the
**per-room paired difference to the no-audio baseline** (mean ± SE,
z = mean / SE):

- IoU at your threshold;
- boundary F-score (1-cell tolerance);
- average precision, which is threshold-free;
- information gain in bits per room over the baseline's probabilities.
  This one is proper: it rewards calibration and punishes
  over-confidence.

The baseline is the per-pixel mean of the training masks, thresholded at
the value that maximises the mean training IoU (τ = 0.09).

## Leaderboard (held-out, filled obstacle mask)

| method | K | IoU | ΔIoU vs baseline (z) | AP | source |
|---|---|---|---|---|---|
| no-audio baseline (training prior map) | 0 | 0.101 | — | 0.13 | this page |
| Phase 2 CNN (skip_v2, calibrated fusion) | 4 | 0.100 | ≈ 0 | — | `tests/reports/sensing_v2_2026_07_15.md` |
| echo ellipses + prior | 4 | 0.128 | +0.027 (9.0) | 0.21 | `tests/reports/imaging_2026_09_24.md` |
| back-projection + prior | 4 | 0.168 | +0.066 (16.8) | 0.28 | same |
| four physics images fused | 4 | 0.189 | +0.088 (19.3) | 0.34 | same |
| four physics images fused | 8 | 0.244 | +0.143 (34.1) | 0.46 | same |
| full-waveform inversion (40 rooms only) | 4 | 0.337 vs 0.087 | +0.251 (4.5) | 0.45 | same |
| aligned U-Net on physics images + prior | 4 | 0.362 | +0.261 (28.2) | 0.58 | `tests/reports/imaging_models_2026_09_24.md` |
| IR migration net (learned filters, delay-and-sum) | 4 | 0.370 | +0.268 (32.1) | 0.60 | same |
| aligned U-Net | 8 | 0.450 | +0.349 (35.9) | 0.68 | same |
| global IR encoder, not spatially aligned (control) | 4 | 0.099 | −0.002 (−1.7) | 0.16 | same |
| passive (unknown source), U-Net | 4 | 0.110 | +0.009 (4.1) | 0.15 | same |

Rows from the physics and imaging-models reports were scored with the same metrics and the
same paired protocol by `scripts/eval_imaging.py`. `scripts/benchmark.py`
reproduces that protocol for new submissions.

## Other targets

The physics report also scores the **illuminated boundary**: only the
obstacle surfaces the sound can reach. Audio helps that target more,
relative to the baseline, because hidden interiors are unobservable by
construction. Room-outline and signed-distance targets are available in
`acoustic_system.imaging.targets`.
