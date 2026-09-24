"""Per-room physics images from an active-sensing archive group.

One call runs every no-ML imager of this package on a room's recordings:
the empty-room background (``ir.empty_room_response``), the scattered
residual, its regularised deconvolution, and the delay-and-sum,
echo-ellipse, free-space-carving and time-reversal images. The images are
spatially aligned with the room grid, which is what plan 6.3.1 needs as
network inputs, and what ``scripts/eval_imaging.py`` scores.

Default settings were chosen on *training* rooms only (the sweep is
described in ``docs/imaging.md``): a 200-lag gate and a :math:`-2`-lag
envelope alignment for back-projection, no spreading compensation (it
amplifies the late, multiply scattered tail), and a :math:`10^{-3}`
relative onset threshold on the raw residual for carving.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .backprojection import backproject
from .image_source import carve_free_space, image_source_maps
from .ir import TikhonovDeconvolver, empty_room_response, envelope, source_drive
from .time_reversal import backpropagate


@dataclass
class ArchiveRoom:
    """One room of a multi-pose archive, restricted to the first ``K`` poses."""

    mask: NDArray[np.bool_]
    sources: NDArray[np.int64]  # (K, 2)
    mics: NDArray[np.int64]  # (K, M, 2)
    recordings: NDArray[np.float64]  # (K, M, T), channel-first
    drive: NDArray[np.float64]  # (T,)
    dt: float

    @classmethod
    def from_group(cls, grp, n_poses: int | None = None) -> "ArchiveRoom":
        """Read an ``h5py`` group written by ``generate_active_sensing.py``."""
        a = grp.attrs
        sensor = np.asarray(grp["sensor"][()], dtype=np.float64)
        if sensor.ndim == 2:  # single-pose layout
            sensor = sensor[None]
            src = np.asarray(a["driver_position"])[None]
            mics = np.asarray(a["sensor_positions"])[None]
        else:
            src = np.asarray(a["driver_positions"])
            mics = np.asarray(a["sensor_positions"])
        k = sensor.shape[0] if n_poses is None else min(int(n_poses), sensor.shape[0])
        T = sensor.shape[1]
        dt = float(a["timestep"])
        drive = source_drive(
            grp["source"][()],
            T,
            dt,
            float(a["audio_native_fs"]) * float(a["sim_time_per_second"]),
            float(a["audio_amplitude"]),
        )
        return cls(
            mask=np.asarray(grp["obstacles"][()], dtype=bool),
            sources=src[:k].astype(np.int64),
            mics=mics[:k].astype(np.int64),
            recordings=np.transpose(sensor[:k], (0, 2, 1)),
            drive=drive,
            dt=dt,
        )


@dataclass
class RoomImages:
    residual: NDArray[np.float64]  # (K, M, T) scattered residual
    ir: NDArray[np.float64]  # (K, M, T) deconvolved scattered IR
    backprojection: NDArray[np.float64]
    ellipses: NDArray[np.float64]
    carving: NDArray[np.float64]  # count of first-arrival ellipses covering each pixel
    time_reversal: NDArray[np.float64]


def compute_room_images(
    room: ArchiveRoom,
    deconvolver: TikhonovDeconvolver | None = None,
    noise_db: float | None = None,
    rng: np.random.Generator | None = None,
    bp_max_lag: int = 200,
    bp_lag_offset: float = -2.0,
    carve_threshold: float = 1e-3,
    carve_offset: float = 0.0,
) -> RoomImages:
    """Run every imager on one room.

    ``noise_db`` adds white Gaussian noise to each recording at that SNR
    (recording RMS over noise RMS) before processing, to test robustness;
    the noise-free archives are otherwise processed as stored.
    """
    grid = room.mask.shape
    K, M, T = room.recordings.shape
    rec = room.recordings.copy()
    if noise_db is not None:
        rng = np.random.default_rng(0) if rng is None else rng
        rms = np.sqrt(np.mean(rec**2, axis=-1, keepdims=True))
        rec = rec + rng.standard_normal(rec.shape) * rms * 10.0 ** (-noise_db / 20.0)
    if deconvolver is None:
        deconvolver = TikhonovDeconvolver(room.drive, T, lam=1e-2)
    res = np.zeros((K, M, T))
    rtm = np.zeros(grid)
    for k in range(K):
        y0, P = empty_room_response(
            grid, room.sources[k], room.drive, room.mics[k], return_field=True
        )
        assert P is not None
        res[k] = rec[k] - y0
        Q = backpropagate(grid, room.mics[k], res[k]).astype(np.float64)
        P64 = P.astype(np.float64)
        illum = np.einsum("nij,nij->ij", P64, P64)
        img = -np.einsum("nij,nij->ij", P64, Q) / (illum + 1e-3 * illum.max())
        rms = float(np.sqrt(np.mean(img**2)))
        if rms > 0:
            rtm += img / rms
    h = deconvolver(res)
    env = envelope(h)
    bp = backproject(
        grid,
        room.sources,
        room.mics,
        h,
        room.dt,
        spreading=False,
        max_lag=bp_max_lag,
        lag_offset=bp_lag_offset,
    )
    ell = image_source_maps(grid, room.sources, room.mics, env, room.dt).sum(axis=0)
    carve = carve_free_space(
        grid,
        room.sources,
        room.mics,
        res,
        room.dt,
        rel_threshold=carve_threshold,
        lag_offset=carve_offset,
    )
    return RoomImages(res, h, bp, ell, carve, rtm)


def device_mask(room: ArchiveRoom) -> NDArray[np.bool_]:
    """Cells occupied by a source or a mic (known to be air)."""
    m = np.zeros(room.mask.shape, dtype=bool)
    for p in np.concatenate([room.sources, room.mics.reshape(-1, 2)]):
        m[int(p[0]), int(p[1])] = True
    return m


__all__ = ["ArchiveRoom", "RoomImages", "compute_room_images", "device_mask"]
