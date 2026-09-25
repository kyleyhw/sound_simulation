"""A small 3D sensing demonstration (plan Task 6.6.5).

Rooms are :math:`N^3` boxes with :math:`p = 0` walls (the same boundary
as the 2D archives) holding a few axis-aligned :math:`p = 0` box
obstacles. A pose is a compact array: one source and four mics on a
horizontal square of half-width ``a`` cells around it (a laptop-sized
aperture, :math:`2a \\approx` a few wavelengths). The drive is a Ricker
wavelet, known to the imager.

The imagers are the 3D versions of the 2D ones: the scattered residual
:math:`r = y - y^{(0)}` against the engine's empty box, its Hilbert
envelope back-projected along the exact 3D bistatic travel lags

.. math::
    t_{km}(\\mathbf{x}) = \\frac{\\lVert\\mathbf{x}-\\mathbf{s}_k\\rVert +
        \\lVert\\mathbf{x}-\\mathbf{m}_{km}\\rVert}{c\\,\\Delta t},

and free-space carving from the first scattered arrival. Everything runs
on ``Simulate``'s 3D fused kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..simulation.setup import Driver
from ..simulation.simulate import Simulate
from .image_source import first_arrival
from .ir import SampledWaveform, envelope


def drive_offsets(drive: NDArray, threshold: float = 1e-3) -> tuple[float, float]:
    """(envelope-peak lag, onset lag at ``threshold`` of the maximum) of a drive."""
    d = np.asarray(drive, dtype=np.float64)
    e = envelope(d)
    onset = int(np.argmax(np.abs(d) >= threshold * np.abs(d).max()))
    return float(np.argmax(e)), float(onset)


def ricker(n_steps: int, dt: float, f0: float, t0: float | None = None) -> NDArray[np.float64]:
    """Ricker wavelet :math:`(1 - 2\\pi^2 f_0^2 \\tau^2) e^{-\\pi^2 f_0^2 \\tau^2}`, :math:`\\tau = t - t_0`."""
    t0 = 1.2 / f0 if t0 is None else t0
    tau = np.arange(n_steps) * dt - t0
    a = (np.pi * f0 * tau) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def random_room_3d(
    n: int,
    rng: np.random.Generator,
    n_obstacles: tuple[int, int] = (2, 4),
    size: tuple[int, int] = (3, 10),
) -> NDArray[np.bool_]:
    """Boolean ``(n, n, n)`` mask of a few random axis-aligned boxes, 2 cells from the walls."""
    mask = np.zeros((n, n, n), dtype=bool)
    for _ in range(int(rng.integers(n_obstacles[0], n_obstacles[1] + 1))):
        ext = rng.integers(size[0], size[1] + 1, size=3)
        lo = [int(rng.integers(2, n - 2 - e)) for e in ext]
        mask[lo[0] : lo[0] + ext[0], lo[1] : lo[1] + ext[1], lo[2] : lo[2] + ext[2]] = True
    return mask


def array_offsets(half_width: int) -> NDArray[np.int64]:
    """Four mics on a horizontal square around the source: ``(4, 3)`` offsets."""
    a = int(half_width)
    return np.array([[a, 0, 0], [-a, 0, 0], [0, a, 0], [0, -a, 0]], dtype=np.int64)


def random_pose_3d(
    mask: NDArray, rng: np.random.Generator, half_width: int = 3, clearance: int = 1
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """A source cell and its 4 mics, all in air at least ``clearance`` cells from obstacles."""
    n = mask.shape[0]
    pad = np.pad(mask, clearance)
    grown = np.zeros_like(mask)
    for d in np.ndindex(*(2 * clearance + 1,) * 3):
        grown |= pad[d[0] : d[0] + n, d[1] : d[1] + n, d[2] : d[2] + n]
    off = array_offsets(half_width)
    lo, hi = 2 + half_width, n - 3 - half_width
    for _ in range(1000):
        s = rng.integers(lo, hi + 1, size=3)
        devs = np.concatenate([s[None], s[None] + off])
        if not grown[tuple(devs.T)].any():
            return s.astype(np.int64), (s[None] + off).astype(np.int64)
    raise RuntimeError("no free pose found")


def record_3d(
    mask: NDArray | None,
    source: NDArray,
    mics: NDArray,
    drive: NDArray,
    grid_shape: tuple[int, int, int],
) -> NDArray[np.float64]:
    """Recordings ``(M, T)`` of one pose (``mask=None``: the empty box)."""
    sim = Simulate(grid_shape=grid_shape)
    if mask is not None:
        sim.set_obstacle_mask(np.asarray(mask, dtype=bool))
    sim.set_drivers(
        [
            Driver(
                position=tuple(int(c) for c in source),
                waveform=SampledWaveform(drive, sim.timestep),
            )
        ]
    )
    idx = tuple(np.asarray(mics, dtype=np.int64).T)
    T = len(drive)
    rec = np.zeros((len(mics), T))
    for n in range(T):
        sim.step()
        rec[:, n] = sim.p[idx]
    return rec


def travel_lags_3d(
    grid_shape: tuple[int, int, int], source: NDArray, mic: NDArray, dt: float
) -> NDArray[np.float64]:
    """Bistatic travel lag (steps, :math:`c = 1`) from ``source`` via every voxel to ``mic``."""
    g = np.indices(grid_shape, dtype=np.float64)
    s = np.asarray(source, dtype=np.float64).reshape(3, 1, 1, 1)
    m = np.asarray(mic, dtype=np.float64).reshape(3, 1, 1, 1)
    return (np.sqrt(((g - s) ** 2).sum(0)) + np.sqrt(((g - m) ** 2).sum(0))) / dt


def _sample(trace: NDArray, lag: NDArray) -> NDArray[np.float64]:
    n = trace.shape[-1]
    i0 = np.floor(lag).astype(np.int64)
    frac = lag - i0
    ok = (i0 >= 0) & (i0 + 1 < n)
    i0c = np.clip(i0, 0, n - 2)
    return np.where(ok, (1 - frac) * trace[i0c] + frac * trace[i0c + 1], 0.0)


@dataclass
class Images3D:
    backprojection: NDArray[np.float64]
    carving: NDArray[np.float64]


def image_room_3d(
    grid_shape: tuple[int, int, int],
    sources: NDArray,
    mics: NDArray,
    residual: NDArray,
    dt: float,
    lag_offset: float,
    carve_offset: float,
    carve_threshold: float = 1e-3,
) -> Images3D:
    """Envelope back-projection and first-arrival carving over all poses.

    ``residual`` is ``(K, M, T)``. ``lag_offset`` is the lag of the drive's
    envelope peak (a scatterer's echo peaks that many steps after its
    geometric arrival); ``carve_offset`` is the lag at which the drive first
    reaches ``carve_threshold`` of its maximum, so a voxel is carved when
    :math:`t_{km}(\\mathbf{x}) + \\text{carve\\_offset} < j_1`.
    """
    K, M, _ = residual.shape
    bp = np.zeros(grid_shape)
    carve = np.zeros(grid_shape)
    env = envelope(residual)
    for k in range(K):
        for m in range(M):
            lag = travel_lags_3d(grid_shape, sources[k], mics[k][m], dt)
            e = env[k, m]
            rms = float(np.sqrt(np.mean(e**2)))
            if rms > 0:
                bp += _sample(e / rms, lag + lag_offset)
            j1 = first_arrival(residual[k, m], carve_threshold)
            if j1 is not None:
                carve += (lag + carve_offset < j1).astype(np.float64)
    return Images3D(bp, carve)


__all__ = [
    "drive_offsets",
    "ricker",
    "random_room_3d",
    "array_offsets",
    "random_pose_3d",
    "record_3d",
    "travel_lags_3d",
    "Images3D",
    "image_room_3d",
]
