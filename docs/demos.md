# Demonstrations of Phases 1 + 2

Two demonstrations exercise the full stack — the FDTD engine (Phase 1)
generating the physics, the sensing recipe (Phase 2) inverting it.
Both consume `src/acoustic_system/learning/sensing.py`: run the
trained encoder on each of K (driver, mic-pair) poses independently
and fuse the per-pose logit maps $\ell_k$ with the calibrated,
prior-corrected Bayes product rule

$$ p_K(x) = \sigma\Bigl(\sum_{k=1}^{K} \bigl(\ell_k(x)/T + b\bigr) - (K-1)\,\operatorname{logit}(\hat\pi)\Bigr), $$

where $\hat\pi$ is the training archive's marginal obstacle fraction
and $(T, b)$ the checkpoint's calibration (identity when no
`calibration.json` sidecar exists). The acquisition protocol, prior,
calibration, and decision threshold all travel with the checkpoint —
`sensing.py` reads them, so the demos work unchanged for any model
generation. Both demos require a trained checkpoint (default:
`checkpoints/skip_v2/best_iou.pt`, the Task 2.3 model — see
[`learning.md`](learning.md) §8) and the `ml` extra. IoU is scored at
the checkpoint's validation-selected operating threshold (shown as τ
in outputs), not a fixed 0.5 — see
[`sensing_v2_2026_07_15.md`](../tests/reports/sensing_v2_2026_07_15.md)
for why this distinction matters.

## 1. Standalone: `scripts/demo_room_mapping.py`

```bash
uv run python scripts/demo_room_mapping.py            # defaults below
uv run python scripts/demo_room_mapping.py --room-seed 12 --seed 400  # a fresh roll
```

Generates a fresh room (never seen in training), acquires K = 8 chirp
poses, and writes a panel figure (`data/plots/demo_room_mapping.png`):

- **Left panel**: ground truth (white = obstacle) with acquisition
  geometry overlaid — red crosses are the chirp positions, cyan dots
  the stereo mic pairs.
- **Remaining panels**: the fused obstacle-probability map at
  K = 1, 2, 4, 8 (viridis; dark = free, bright = obstacle), IoU at
  threshold 0.5 in each title.

How to interpret it: the demonstration *is* the left-to-right
progression — diffuse probability mass at K = 1 contracting onto the
true rectangles as poses accumulate. Expect coarse blobs, not sharp
walls: the campaign's held-out mean is IoU ≈ 0.10 at K = 8.

Default seeds (`--room-seed 7`, `--seed 107`) were chosen from a
24-room sweep (see the test report) as a representative case with a
clean monotone progression (0.075 → 0.195); the sweep's *mean*
progression was 0.049 → 0.099, consistent with the held-out benchmark,
and individual rooms vary widely — other seeds are an honest roll of
the dice.

## 2. In the web UI: the "Acoustic sensing" panel

Start the UI as usual (`docs/web_ui.md`), draw obstacles in 2D mode,
and press **Sense room** in the *Acoustic sensing* panel. The backend
snapshots the drawn obstacle mask, resamples it to the 64×64
acquisition grid, runs the same K-pose pipeline in a worker thread
(the live simulation keeps streaming), and returns the fused map,
which the panel renders in viridis with the true obstacle cells
whitened for comparison. The caption reports the IoU progression
K = 1…K and the runtime (~0.3 s at K = 8).

Wire protocol addition (full table in `web_ui.md`):

| event | direction | payload |
| --- | --- | --- |
| `sense_room` | client → server | `{poses?: 1-16, seed?: int}` |
| `sense_result` | server → requester | `{ok, prob (64×64), truth, ious, poses, elapsed}` or `{ok: false, error}` |

`sense_result` goes only to the requesting client (it answers a button
press), unlike the broadcast geometry/status events. Failure modes
(3D grid, missing `ml` extra, missing checkpoint) come back as error
strings and are shown in the panel.

Model-fidelity caveat: the checkpoint was trained on rooms with three
rectangles of side 4–14 cells on a 64×64 grid at ~6 % occupancy.
Rooms drawn far outside that distribution (dense mazes, thin walls,
huge blobs) will produce correspondingly worse maps — the model
interpolates its training distribution; it does not solve general
inverse scattering.

## Testing

- `tests/learning/test_sensing.py` — pipeline gates: shapes/ranges,
  seeded determinism, the K = 1 fusion identity
  ($p_1 = \sigma(\ell_1)$, i.e. the prior term vanishes), and pose
  geometry legality.
- `tests/app/test_sense_event.py` — bridge gates: payload contract on
  the happy path (including truth preservation through the 200→64
  resample), and the 3D / missing-checkpoint error paths.
- End-to-end: servers up, room drawn over the socket, **Sense room**
  clicked in a real browser via Playwright, fused map + IoU line
  verified rendered. Evidence and runtimes:
  [`demos_2026_07_12.md`](../tests/reports/demos_2026_07_12.md).
