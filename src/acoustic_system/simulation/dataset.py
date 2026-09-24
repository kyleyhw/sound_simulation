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
    protocol: str = "v3",
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
        Inclusive bounds on rectangle side length, in cells.
    margin
        Minimum free border between any obstacle and the outer wall.
    rng
        A ``numpy.random.Generator`` for reproducibility. If ``None``,
        a fresh generator is used and results are non-deterministic.
    protocol
        ``"v3"`` (default): the corner draw's upper bound is inclusive, so
        the bottom/right margin equals ``margin`` exactly. ``"v2"``: the
        original draw (effective bottom/right margin ``margin + 1``),
        kept so the v1/v2 archives regenerate byte-identically.

    Returns
    -------
    A boolean ``np.ndarray`` of shape ``grid_shape``.
    """
    _check_protocol(protocol)
    edge = 0 if protocol == "v2" else 1
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
        i0 = int(rng.integers(margin, ni - h - margin + edge))
        j0 = int(rng.integers(margin, nj - w - margin + edge))
        mask[i0 : i0 + h, j0 : j0 + w] = True
    return mask


def _check_protocol(protocol: str) -> None:
    if protocol not in ("v2", "v3"):
        raise ValueError(f"protocol must be 'v2' or 'v3', got {protocol!r}")


def generate_diverse_obstacles(
    grid_shape: Tuple[int, ...],
    n_obstacles_range: Tuple[int, int] = (1, 6),
    min_size: int = 4,
    max_size: int = 14,
    margin: int = 2,
    rng: Optional[np.random.Generator] = None,
    protocol: str = "v3",
) -> np.ndarray:
    """Sample a boolean obstacle mask from a mixture of shape families.

    Motivation (Task 2.3c): models trained on the pure-rectangle
    distribution of :func:`generate_random_obstacles` are brittle on
    anything else — hand-drawn rooms in the web UI (strokes, rings,
    blobs) sit far outside that distribution and produce degenerate
    predictions. This generator widens the *training* distribution
    with the shape families such rooms actually contain.

    Per room, the obstacle count is drawn uniformly from
    ``n_obstacles_range`` (inclusive); each obstacle draws its family
    uniformly from four options:

    1. **Axis-aligned rectangle** — sides $\\sim U[s_{\\min}, s_{\\max}]$
       (the original family, kept so v1-style rooms stay in-support).
    2. **Filled disc** — radius $\\sim U[s_{\\min}/2, s_{\\max}/2]$;
       cells within Euclidean distance $r$ of the centre.
    3. **Thin wall** — a line segment of length
       $\\sim U[2 s_{\\min}, 2 s_{\\max}]$ and thickness $\\sim U[1, 3]$,
       oriented horizontally, vertically, or at $\\pm 45°$; rasterised
       by stamping a square brush of the given thickness along the
       segment (the diagonal case matches what a mouse drag draws).
    4. **L-shape** — two overlapping rectangles sharing a corner
       region, giving concave geometry the convex families lack.

    Same ``margin`` contract as the rectangle generator: every obstacle
    cell stays at least ``margin`` cells from the outer wall. Shapes
    may overlap (union), as before. Typical occupancy for the default
    band on a 64x64 grid is ~2-15 % depending on the count draw;
    consumers should measure and record the realised mean rather than
    assume the v1 prior of 0.058.

    Returns a boolean ``np.ndarray`` of shape ``grid_shape``. 2D only
    (matching the sensing pipeline); non-2D shapes, and grids too small to
    hold a ``min_size`` obstacle inside the margin, return an empty mask
    like the rectangle generator does.

    ``protocol="v3"`` (default) stamps thin walls with exactly the drawn
    thickness (``"v2"`` stamped thickness 2 as 3 cells, so realised
    thicknesses were {1, 3, 3}); ``"v2"`` reproduces the v2 archives.
    """
    _check_protocol(protocol)
    if rng is None:
        rng = np.random.default_rng()
    mask = np.zeros(grid_shape, dtype=bool)
    if len(grid_shape) != 2:
        return mask
    ni, nj = grid_shape
    if min(ni, nj) - 2 * margin <= max(min_size, 1):
        return mask
    lo, hi = int(n_obstacles_range[0]), int(n_obstacles_range[1])
    n_obstacles = int(rng.integers(lo, hi + 1))

    def clip_i(v: int) -> int:
        return int(np.clip(v, margin, ni - 1 - margin))

    def clip_j(v: int) -> int:
        return int(np.clip(v, margin, nj - 1 - margin))

    def stamp_rect(i0: int, j0: int, h: int, w: int) -> None:
        h = min(h, ni - 2 * margin - 1)
        w = min(w, nj - 2 * margin - 1)
        if h <= 0 or w <= 0:
            return
        i0 = int(np.clip(i0, margin, ni - h - margin))
        j0 = int(np.clip(j0, margin, nj - w - margin))
        mask[i0 : i0 + h, j0 : j0 + w] = True

    for _ in range(n_obstacles):
        family = int(rng.integers(4))
        if family == 0:  # rectangle
            h = int(rng.integers(min_size, max_size + 1))
            w = int(rng.integers(min_size, max_size + 1))
            stamp_rect(
                int(rng.integers(margin, max(ni - h - margin, margin + 1))),
                int(rng.integers(margin, max(nj - w - margin, margin + 1))),
                h,
                w,
            )
        elif family == 1:  # filled disc
            r = float(rng.uniform(min_size / 2.0, max_size / 2.0))
            ci = int(rng.integers(margin + int(r), max(ni - margin - int(r), margin + int(r) + 1)))
            cj = int(rng.integers(margin + int(r), max(nj - margin - int(r), margin + int(r) + 1)))
            ii, jj = np.ogrid[0:ni, 0:nj]
            disc = (ii - ci) ** 2 + (jj - cj) ** 2 <= r * r
            disc[:margin, :] = disc[ni - margin :, :] = False
            disc[:, :margin] = disc[:, nj - margin :] = False
            mask |= disc
        elif family == 2:  # thin wall (h / v / diagonal), stamped brush
            length = int(rng.integers(2 * min_size, 2 * max_size + 1))
            thickness = int(rng.integers(1, 4))
            di, dj = [(0, 1), (1, 0), (1, 1), (1, -1)][int(rng.integers(4))]
            i, j = (
                int(rng.integers(margin, ni - margin)),
                int(rng.integers(margin, nj - margin)),
            )
            half = thickness // 2
            # v3: a brush spanning exactly `thickness` cells
            # [i - half, i - half + thickness - 1]; v2 spanned 2*half + 1.
            span = thickness - 1 if protocol == "v3" else 2 * half
            for _step in range(length):
                i0, i1 = clip_i(i - half), clip_i(i - half + span)
                j0, j1 = clip_j(j - half), clip_j(j - half + span)
                mask[i0 : i1 + 1, j0 : j1 + 1] = True
                i, j = i + di, j + dj
                if not (margin <= i < ni - margin and margin <= j < nj - margin):
                    break
        else:  # L-shape: two rectangles sharing a corner region
            h1 = int(rng.integers(min_size, max_size + 1))
            w1 = int(rng.integers(min_size, max_size + 1))
            i0 = int(rng.integers(margin, max(ni - h1 - margin, margin + 1)))
            j0 = int(rng.integers(margin, max(nj - w1 - margin, margin + 1)))
            stamp_rect(i0, j0, h1, w1)
            # Second arm grows from the first rectangle's corner.
            h2 = int(rng.integers(min_size, max_size + 1))
            w2 = int(rng.integers(min_size, max_size + 1))
            stamp_rect(i0 + h1 - max(1, min_size // 2), j0, h2, w2)
    return mask


def pick_mic_positions(
    grid_shape: Tuple[int, ...],
    obstacle_mask: np.ndarray,
    n_mics: int = 2,
    spacing: float = 16.0,
    rng: Optional[np.random.Generator] = None,
    margin: int = 2,
    max_attempts: int = 200,
    exclude: Optional[Sequence[Tuple[int, ...]]] = None,
) -> list[tuple[int, ...]]:
    """Pick ``n_mics`` interior cells representing a microphone array.

    Geometry
    --------
    Mics are placed about a randomly drawn centre, oriented along a
    uniformly random unit direction:

    - ``n_mics == 1``: one mic at the centre.
    - ``n_mics == 2``: a stereo pair at $\\vec{c} \\pm \\tfrac{1}{2}\\, s\\, \\hat{u}$,
      where $\\hat{u}$ is uniform on the unit sphere ($d=3$) or unit
      circle ($d=2$) and $s$ is the requested ``spacing`` in cells.
      This mimics a stock-laptop binaural arrangement: two mics
      separated by a fixed baseline, orientation determined by however
      the user is holding the device.

    Rejection sampling: redraw centre + orientation if any mic falls
    outside the interior margin, coincides with an obstacle, or lands on
    a cell in ``exclude`` (pass the source cell so a mic never records
    the injected drive directly). Raises ``RuntimeError`` after
    ``max_attempts``. ``exclude=None`` keeps the original RNG stream.

    Higher mic counts (3-mic webcam setups) are not yet implemented;
    the project's hardware constraint caps at 2-3 channels, so this
    slot is reserved for that extension.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_mics < 1:
        raise ValueError("n_mics must be >= 1")
    banned = {tuple(int(c) for c in e) for e in (exclude or [])}
    if n_mics == 1:
        for _ in range(max_attempts):
            pos = random_free_position(grid_shape, obstacle_mask, rng, margin)
            if pos not in banned:
                return [pos]
        raise RuntimeError("could not place the mic away from the excluded cells")
    if n_mics != 2:
        raise NotImplementedError(f"n_mics={n_mics} not yet supported; only 1 and 2 implemented")

    dims = len(grid_shape)
    half_spacing = float(spacing) / 2.0
    for _ in range(max_attempts):
        centre = np.array(
            random_free_position(grid_shape, obstacle_mask, rng, margin),
            dtype=np.float64,
        )
        # Uniform direction on the unit sphere (3D) or unit circle (2D).
        # standard_normal sampling normalised by L2 gives the rotation-
        # invariant uniform distribution on the sphere.
        direction = rng.standard_normal(dims)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        direction /= norm

        a_pos = centre - half_spacing * direction
        b_pos = centre + half_spacing * direction
        positions: list[tuple[int, ...]] = []
        ok = True
        for raw in (a_pos, b_pos):
            tpos = tuple(int(round(c)) for c in raw)
            if not all(margin <= c < s - margin for c, s in zip(tpos, grid_shape)):
                ok = False
                break
            if obstacle_mask[tpos] or tpos in banned:
                ok = False
                break
            positions.append(tpos)
        if ok and positions[0] != positions[1]:
            return positions
    raise RuntimeError(
        f"could not place {n_mics} mics with spacing {spacing} after {max_attempts} attempts"
    )


def random_free_position(
    grid_shape: Tuple[int, ...],
    obstacle_mask: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    margin: int = 2,
    max_attempts: int = 1000,
) -> Tuple[int, ...]:
    """Pick a uniform-random interior cell that is *not* an obstacle.

    Rejection sampling: draw a candidate uniformly from the interior
    (one ``margin`` away from each wall) and accept it iff
    ``obstacle_mask[idx] == False``. Falls back to an exhaustive scan
    after ``max_attempts`` rejections so a heavily obstructed scene
    cannot livelock the generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    # N-D: one draw per axis in axis order (identical stream to the original
    # 2D (i, j) draw order, so existing archives regenerate unchanged).
    for _ in range(max_attempts):
        pos = tuple(int(rng.integers(margin, n - margin)) for n in grid_shape)
        if not obstacle_mask[pos]:
            return pos
    # Exhaustive fallback. Build the list of free interior cells and
    # pick uniformly. ``np.argwhere`` keeps this allocation bounded.
    interior = np.zeros(grid_shape, dtype=bool)
    interior[tuple(slice(margin, n - margin) for n in grid_shape)] = True
    free = np.argwhere(interior & ~obstacle_mask)
    if free.size == 0:
        raise RuntimeError("no free interior cells: obstacle_mask fills the entire interior")
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

    Sensor positions must be in-bounds tuples matching ``sim.dims``;
    anything else raises ``ValueError`` up front (a negative index would
    otherwise silently wrap to the opposite wall).
    """
    if record_step < 1:
        raise ValueError(f"record_step must be >= 1, got {record_step}")
    positions = [tuple(int(c) for c in s.position) for s in sensors]
    for pos in positions:
        if len(pos) != sim.dims or not all(0 <= c < n for c, n in zip(pos, sim.grid_shape)):
            raise ValueError(f"sensor position {pos} is outside grid {sim.grid_shape}")
    if duration <= 0:
        return np.zeros((0, len(sensors)), dtype=np.float32)
    n_recorded = (duration + record_step - 1) // record_step
    out = np.zeros((n_recorded, len(sensors)), dtype=np.float32)
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
    "generate_diverse_obstacles",
    "random_free_position",
    "pick_mic_positions",
    "run_with_sensors",
    "synthetic_chirp",
]
