"""End-to-end room sensing: simulate K chirp poses, infer, Bayes-fuse.

This module packages the **recommended Phase 2 sensing recipe**
(`tests/reports/joint_pose_2026_07_11.md`) as a reusable function:

1. Given a room's obstacle mask, place K independent (driver, mic-pair)
   poses — the physical picture is a laptop carried to K spots, playing
   a chirp and recording at each.
2. Run the FDTD forward simulation per pose and record the stereo
   sensor timeseries (same parameters as the training distribution).
3. Run the joint-trained encoder on each pose *separately* (K=1 input).
4. Fuse the per-pose logit maps with the prior-corrected Bayes product
   rule:

   $$ p_K(x) = \\sigma\\Bigl(\\sum_{k=1}^{K} \\ell_k(x)
      - (K-1)\\,\\operatorname{logit}(\\hat\\pi)\\Bigr), $$

   where $\\hat\\pi$ is the marginal obstacle prior. Held-out IoU of
   this recipe: 0.0924 at K=8 vs 0.037 single-pose.

Consumers: ``scripts/demo_room_mapping.py`` (standalone demo) and the
web UI's ``sense_room`` socket event (``app/main.py``).

Simulation parameters deliberately mirror the training-dataset
defaults recorded in ``docs/learning.md`` (64-cell grid, 200 steps,
courant 0.5, stereo pair at spacing 12, synthetic chirp 0.02 -> 0.4 at
fs = 200): the model has only ever seen this acquisition protocol, so
the demo must reproduce it. Rooms whose masks come from another grid
size are resampled to 64x64 nearest-neighbour, exactly as the training
pipeline resizes masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from acoustic_system.learning.model import build_model
from acoustic_system.simulation.dataset import (
    pick_mic_positions,
    random_free_position,
    run_with_sensors,
    synthetic_chirp,
)
from acoustic_system.simulation.setup import Driver, Sensor
from acoustic_system.simulation.simulate import Simulate
from acoustic_system.simulation.waveforms import AudioFileWaveform

# Marginal obstacle prior of the training distribution: mean obstacle
# fraction of the 64x64 masks produced by the canonical generator
# settings (3 rectangles, sides 4-14). Measured as 0.0582 over the
# 500-room held-out archive (tests/reports/multipose_2026_07_10.md).
# Used by the Bayes product rule; correct value matters more as K grows.
TRAINING_OBSTACLE_PRIOR: float = 0.0582

# Canonical acquisition protocol (must match the training dataset).
GRID: int = 64
DURATION_STEPS: int = 200
COURANT: float = 0.5
MIC_SPACING: float = 12.0
CHIRP_F_START: float = 0.02
CHIRP_F_END: float = 0.4
CHIRP_SAMPLE_RATE: float = 200.0
AUDIO_AMPLITUDE: float = 5.0

_model_cache: dict[str, tuple[Any, int]] = {}


def load_sensing_model(checkpoint_path: str | Path) -> tuple[Any, int]:
    """Load (and cache) a sensing checkpoint; returns ``(model, target_size)``.

    The cache avoids re-deserialising the ~1 MB state dict on every
    socket event. Keyed by resolved path.
    """
    key = str(Path(checkpoint_path).resolve())
    if key not in _model_cache:
        ckpt = torch.load(key, map_location="cpu", weights_only=False)
        model = build_model(str(ckpt.get("model_type", "dual")), n_mics=int(ckpt.get("n_mics", 2)))
        model.load_state_dict(ckpt["model"])
        model.eval()
        _model_cache[key] = (model, int(ckpt["args"].get("target_size", 64)))
    return _model_cache[key]


def resize_mask(
    mask: NDArray[np.floating[Any]] | NDArray[np.bool_], size: int
) -> NDArray[np.float32]:
    """Nearest-neighbour resize to ``(size, size)`` (training convention)."""
    m = np.asarray(mask, dtype=np.float32)
    if m.shape == (size, size):
        return m
    t = torch.from_numpy(m)[None, None]
    return torch.nn.functional.interpolate(t, size=(size, size), mode="nearest")[0, 0].numpy()


@dataclass
class SenseResult:
    """Output of :func:`sense_room`.

    Attributes
    ----------
    logits
        Per-pose logit maps, shape ``(K, H, W)`` float32.
    fused_probs
        Bayes-fused probability map for each prefix ``k = 1..K``,
        shape ``(K, H, W)``: ``fused_probs[k-1]`` uses the first k poses.
    ious
        Mean IoU (threshold 0.5) of each fused map against the room's
        (resized) truth mask, length ``K``.
    truth
        The 64x64 truth mask the IoUs are scored against, float32 {0,1}.
    driver_positions, mic_positions
        Pose geometry: ``(K, 2)`` and ``(K, n_mics, 2)`` int arrays.
    """

    logits: NDArray[np.float32]
    fused_probs: NDArray[np.float32]
    ious: list[float]
    truth: NDArray[np.float32]
    driver_positions: NDArray[np.int32]
    mic_positions: NDArray[np.int32]


def _iou(pred_bin: NDArray[np.float32], truth: NDArray[np.float32], eps: float = 1e-6) -> float:
    inter = float((pred_bin * truth).sum())
    union = float(pred_bin.sum() + truth.sum() - inter)
    return (inter + eps) / (union + eps)


def sense_room(
    obstacle_mask: NDArray[np.bool_] | NDArray[np.floating[Any]],
    checkpoint_path: str | Path,
    n_poses: int = 8,
    seed: int = 0,
    prior: float = TRAINING_OBSTACLE_PRIOR,
    rng: Optional[np.random.Generator] = None,
) -> SenseResult:
    """Run the full sense -> infer -> fuse pipeline on one room.

    Parameters
    ----------
    obstacle_mask
        2D boolean/0-1 mask of the room at any resolution; resampled to
        the 64x64 acquisition grid.
    checkpoint_path
        A sensing checkpoint (typically ``checkpoints/joint_baseline/
        best_iou.pt``). Any ``model_type`` works — the model is run
        per pose, so joint models take their K=1 path.
    n_poses
        Number of (driver, mic-pair) poses to acquire.
    seed / rng
        Reproducibility. ``rng`` wins if provided.

    Notes
    -----
    The FDTD forward runs are exact physics (to discretisation error);
    only the inverse map is learned. Poses are drawn i.i.d. exactly as
    in dataset generation, so results are statistically comparable to
    the held-out benchmarks.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    model, target_size = load_sensing_model(checkpoint_path)

    truth = resize_mask(np.asarray(obstacle_mask), GRID)
    mask_bool = truth > 0.5

    # Shared chirp source for all poses (one device, one excitation).
    probe = Simulate(grid_shape=(GRID, GRID), courant=COURANT)
    dt = probe.timestep
    n_audio = int(CHIRP_SAMPLE_RATE * DURATION_STEPS * dt)
    chirp = synthetic_chirp(
        duration_samples=n_audio,
        sample_rate=CHIRP_SAMPLE_RATE,
        f_start=CHIRP_F_START,
        f_end=CHIRP_F_END,
        amplitude=1.0,
    )
    source_t = torch.from_numpy(chirp.copy())[None, None]  # (1, 1, T_audio)

    # One engine per room; reset() between poses (keeps geometry).
    sim = Simulate(grid_shape=(GRID, GRID), drivers=[], courant=COURANT)
    cells = np.argwhere(mask_bool)
    if cells.size:
        sim.set_obstacle([tuple(int(c) for c in row) for row in cells])

    logits_list: list[NDArray[np.float32]] = []
    drivers: list[tuple[int, ...]] = []
    mics: list[list[tuple[int, ...]]] = []
    with torch.no_grad():
        for _ in range(int(n_poses)):
            driver_pos = random_free_position((GRID, GRID), mask_bool, rng=rng)
            mic_pos = pick_mic_positions(
                (GRID, GRID), mask_bool, n_mics=2, spacing=MIC_SPACING, rng=rng
            )
            wf = AudioFileWaveform.from_samples(
                samples=chirp,
                sample_rate=CHIRP_SAMPLE_RATE,
                amplitude=AUDIO_AMPLITUDE,
                delay=0.0,
                sim_time_per_second=1.0,
            )
            sim.reset()
            sim.set_drivers([Driver(position=driver_pos, waveform=wf)])
            rec = run_with_sensors(
                sim=sim,
                duration=DURATION_STEPS,
                sensors=[Sensor(position=p) for p in mic_pos],
                record_step=1,
            )  # (T_rec, 2)
            sensor_t = torch.from_numpy(rec.T.copy())[None]  # (1, 2, T_rec)
            logits = model(sensor_t, source_t)[0, 0].numpy().astype(np.float32)
            logits_list.append(logits)
            drivers.append(driver_pos)
            mics.append(mic_pos)

    logits_arr = np.stack(logits_list, axis=0)  # (K, H, W)
    prior_logit = float(np.log(prior / (1.0 - prior)))

    fused = np.empty_like(logits_arr)
    ious: list[float] = []
    running_sum = np.zeros_like(logits_arr[0], dtype=np.float64)
    truth_scored = resize_mask(truth, target_size)
    for k in range(len(logits_list)):
        running_sum += logits_arr[k]
        fused_logit = running_sum - k * prior_logit  # k = K-1 extra poses
        prob = 1.0 / (1.0 + np.exp(-fused_logit))
        fused[k] = prob.astype(np.float32)
        pred = (resize_mask(fused[k], target_size) > 0.5).astype(np.float32)
        ious.append(_iou(pred, truth_scored))

    return SenseResult(
        logits=logits_arr,
        fused_probs=fused,
        ious=ious,
        truth=truth,
        driver_positions=np.asarray(drivers, dtype=np.int32),
        mic_positions=np.asarray(mics, dtype=np.int32),
    )
