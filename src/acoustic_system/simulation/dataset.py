"""Active-sensing dataset helpers.

Bundles the three things the active-sensing data generator needs but the
core ``Simulate`` engine intentionally does not own:

1. Random 2D room geometry (``generate_random_obstacles``).
2. Free-cell sampling so a driver / sensor does not get placed inside a
   wall (``random_free_position``).
3. A streaming-recording runner that advances the simulation ``duration``
   steps and returns the per-sensor pressure timeseries without keeping
   the full pressure history in memory (``run_with_sensors``).

These helpers are deliberately kept out of ``simulate.py`` to preserve
the engine's tight step-at-a-time contract and to leave the public
attribute surface tested by ``tests/perf/check_simulate.py`` unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .setup import Sensor
from .simulate import Simulate


# =====================================================================
# Synthetic acoustic geometry
# =====================================================================


def generate_random_obstacles(
    grid_shape: Tuple[int, ...],
    n_obstacles: int = 3,
    min_size: int = 5,
    max_size: int = 30,
    margin: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a boolean obstacle mask consisting of axis-aligned rectangles.

    Mathematical setup
    ------------------
    For each of ``n_obstacles`` rectangles, dimensions $(h, w)$ are drawn
    uniformly from $[\\text{min\\_size}, \\text{max\\_size}]^2$ and the
    top-left corner $(i_0, j_0)$ is drawn uniformly from the legal
    sub-grid

    $$ i_0 \\in [\\text{margin},\\, N_i - h - \\text{margin}],
       \\quad j_0 \\in [\\text{margin},\\, N_j - w - \\text{margin}]. $$

    The ``margin`` keeps obstacles strictly inside the outer hard-wall
    boundary so a driver placed at $(1, 1)$ is never coincident with an
    obstacle.

    Rectangles may overlap; overlapping cells stay True. This yields a
    diversity of room topologies — isolated boxes, L-shapes, T-shapes —
    at no extra cost. A larger ``n_obstacles`` tends to produce more
    contiguous clutter, smaller values give sparse scatterers.

    Parameters
    ----------
    grid_shape
        The full simulation grid shape (matches ``Simulate.grid_shape``).
    n_obstacles
        How many rectangles to draw. ``0`` returns an empty mask.
    min_size, max_size
        Inclusive lower / exclusive upper bounds (numpy convention) on
        rectangle side length, in cells.
    margin
        Minimum free border between any obstacle and the outer wall.
    rng
        A ``numpy.random.Generator`` for reproducibility. If ``None``,
        a fresh generator is used and results are non-deterministic.

    Returns
    -------
    A boolean ``np.ndarray`` of shape ``grid_shape``.
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = np.zeros(grid_shape, dtype=bool)
    if n_obstacles <= 0 or len(grid_shape) != 2:
        return mask
    ni, nj = grid_shape
    for _ in range(int(n_obstacles)):
        h = int(rng.integers(min_size, max_size + 1))
        w = int(rng.integers(min_size, max_size + 1))
        h = min(h, ni - 2 * margin - 1)
        w = min(w, nj - 2 * margin - 1)
        if h <= 0 or w <= 0:
            continue
        i0 = int(rng.integers(margin, ni - h - margin))
        j0 = int(rng.integers(margin, nj - w - margin))
        mask[i0 : i0 + h, j0 : j0 + w] = True
    return mask


def random_free_position(
    grid_shape: Tuple[int, ...],
    obstacle_mask: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    margin: int = 2,
    max_attempts: int = 1000,
) -> Tuple[int, ...]:
    """Pick a uniform-random interior cell that is *not* an obstacle.

    Rejection sampling: draw a candidate $(i, j)$ uniformly from the
    interior (one ``margin`` away from each wall) and accept it iff
    ``obstacle_mask[i, j] == False``. Falls back to an exhaustive scan
    after ``max_attempts`` rejections so a heavily obstructed scene
    cannot livelock the generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    ni, nj = grid_shape
    for _ in range(max_attempts):
        i = int(rng.integers(margin, ni - margin))
        j = int(rng.integers(margin, nj - margin))
        if not obstacle_mask[i, j]:
            return (i, j)
    # Exhaustive fallback. Build the list of free interior cells and
    # pick uniformly. ``np.argwhere`` keeps this allocation bounded.
    interior = np.zeros(grid_shape, dtype=bool)
    interior[margin : ni - margin, margin : nj - margin] = True
    free = np.argwhere(interior & ~obstacle_mask)
    if free.size == 0:
        raise RuntimeError(
            "no free interior cells: obstacle_mask fills the entire interior"
        )
    idx = int(rng.integers(len(free)))
    return tuple(int(c) for c in free[idx])


# =====================================================================
# Streaming sensor runner
# =====================================================================


def run_with_sensors(
    sim: Simulate,
    duration: int,
    sensors: Sequence[Sensor],
    record_step: int = 1,
) -> np.ndarray:
    """Advance ``sim`` by ``duration`` steps, recording each sensor in place.

    Returns an array of shape ``(ceil(duration / record_step), n_sensors)``
    holding ``sim.p[sensor.position]`` at every recorded step. Memory is
    $O(\\text{duration} \\cdot n_{\\text{sensors}})$ rather than
    $O(\\text{duration} \\cdot N_x N_y)$, which matters once you start
    generating thousands of training samples per dataset.

    The ``record_step`` knob lets the caller decimate by an integer factor
    when the sim runs at a much higher rate than is useful for the
    downstream model (e.g. dt_sim = 1e-5 s but the model only needs
    16 kHz observations: record_step = round(1 / (16000 * dt_sim))).
    Set to 1 to record every step (the default).

    The sensors' positions must be in-bounds tuples matching ``sim.dims``;
    the runner does not validate this (a bad index will raise from numpy).
    """
    if duration <= 0:
        return np.zeros((0, len(sensors)), dtype=np.float32)
    n_recorded = (duration + record_step - 1) // record_step
    out = np.zeros((n_recorded, len(sensors)), dtype=np.float32)
    positions = [tuple(int(c) for c in s.position) for s in sensors]
    write_idx = 0
    for step_idx in range(duration):
        sim.step()
        if step_idx % record_step == 0:
            row = out[write_idx]
            for s_idx, pos in enumerate(positions):
                row[s_idx] = sim.p[pos]
            write_idx += 1
    # Attach the timeseries back onto the Sensor objects too, so callers
    # that prefer the existing main.py-style "sensor.timeseries" pattern
    # find the data where they expect it.
    for s_idx, sensor in enumerate(sensors):
        sensor.timeseries = out[:, s_idx].copy()
        sensor.sample_rate = (1.0 / sim.timestep) / float(record_step)
    return out


# =====================================================================
# Synthetic source (for when the user has no audio corpus yet)
# =====================================================================


def synthetic_chirp(
    duration_samples: int,
    sample_rate: float,
    f_start: float,
    f_end: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Linear chirp $u(t) = A \\sin(2\\pi (f_0 + 0.5 k t) t)$.

    Useful as a stand-in source for the active-sensing pipeline when no
    real audio is available. The instantaneous frequency sweeps linearly
    from ``f_start`` to ``f_end`` over ``duration_samples / sample_rate``
    seconds, with sweep rate $k = (f_{\\text{end}} - f_{\\text{start}}) / T$.

    Returns a float32 numpy array of length ``duration_samples`` in the
    range $[-A, A]$.
    """
    T = float(duration_samples) / float(sample_rate)
    k = (float(f_end) - float(f_start)) / T if T > 0 else 0.0
    t = np.arange(duration_samples, dtype=np.float64) / float(sample_rate)
    phase = 2.0 * np.pi * (float(f_start) * t + 0.5 * k * t * t)
    return (amplitude * np.sin(phase)).astype(np.float32)


__all__ = [
    "generate_random_obstacles",
    "random_free_position",
    "run_with_sensors",
    "synthetic_chirp",
]
