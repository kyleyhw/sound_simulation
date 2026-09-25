"""Per-pose physics images and pose geometry channels (plan 6.3, 6.6.1).

``pipeline.compute_room_images`` returns images already summed over the
:math:`K` poses. The learned models of ``models.py`` need them *per pose*
(the pose-aware set model of plan 6.3.3 fuses them itself), together with
the pose geometry. Every imager of this package is a sum of per-pose terms,

.. math::
    I(\\mathbf{x}) = \\sum_{k=1}^{K} I_k(\\mathbf{x}),

because back-projection normalises each (pose, mic) trace separately,
echo ellipses and carving counts add per (pose, mic) pair, and the
time-reversal stack scales each pose image by its own RMS. So
:func:`aggregate` of :func:`compute_pose_images` reproduces
``compute_room_images`` exactly (``tests/imaging/test_pose_images.py``).

Pose error (6.6.1) enters here: :class:`~.pipeline.ArchiveRoom` keeps the
recordings at the *true* device cells, while the positions stored in the
room are the ones the imager *assumes*. :func:`perturb_poses` returns a
room whose assumed positions are the true ones plus rounded Gaussian
errors; the empty-room background and every travel time then use the
wrong cells, exactly as a mis-placed laptop would.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .backprojection import backproject, travel_lags
from .image_source import carve_free_space, image_source_maps
from .ir import TikhonovDeconvolver, empty_room_response, envelope
from .pipeline import ArchiveRoom
from .time_reversal import backpropagate

#: Channel order of :attr:`PoseImages.images`.
POSE_CHANNELS = ("backprojection", "ellipses", "carving", "time_reversal")


@dataclass
class PoseImages:
    """Physics images of one room, one per pose.

    Attributes
    ----------
    images
        ``(K, 4, H, W)`` in the order of :data:`POSE_CHANNELS`.
    residual
        ``(K, M, T)`` scattered residual :math:`y - y^{(0)}`.
    ir
        ``(K, M, T)`` deconvolved scattered impulse response.
    """

    images: NDArray[np.float64]
    residual: NDArray[np.float64]
    ir: NDArray[np.float64]


def compute_pose_images(
    room: ArchiveRoom,
    deconvolver: TikhonovDeconvolver | None = None,
    bp_max_lag: int = 200,
    bp_lag_offset: float = -2.0,
    carve_threshold: float = 1e-3,
    incident: list[tuple[NDArray, NDArray]] | None = None,
) -> PoseImages:
    """Every imager of ``pipeline.compute_room_images``, kept per pose.

    Parameters
    ----------
    room
        Recordings plus the *assumed* device cells.
    incident
        Optional precomputed ``(y0, P)`` per pose from
        :func:`~.ir.empty_room_response` at the assumed cells (the pose
        refiner of ``pose_refine.py`` already has them).
    """
    grid = room.mask.shape
    K, M, T = room.recordings.shape
    if deconvolver is None:
        deconvolver = TikhonovDeconvolver(room.drive, T, lam=1e-2)
    res = np.zeros((K, M, T))
    imgs = np.zeros((K, len(POSE_CHANNELS)) + tuple(grid))
    for k in range(K):
        if incident is None:
            y0, P = empty_room_response(
                grid, room.sources[k], room.drive, room.mics[k], return_field=True
            )
        else:
            y0, P = incident[k]
        assert P is not None
        res[k] = room.recordings[k] - y0
        Q = backpropagate(grid, room.mics[k], res[k]).astype(np.float64)
        P64 = P.astype(np.float64)
        illum = np.einsum("nij,nij->ij", P64, P64)
        img = -np.einsum("nij,nij->ij", P64, Q) / (illum + 1e-3 * illum.max())
        rms = float(np.sqrt(np.mean(img**2)))
        if rms > 0:
            imgs[k, 3] = img / rms
    h = deconvolver(res)
    env = envelope(h)
    ell = image_source_maps(grid, room.sources, room.mics, env, room.dt)
    for k in range(K):
        sl = slice(k, k + 1)
        imgs[k, 0] = backproject(
            grid,
            room.sources[sl],
            room.mics[sl],
            h[sl],
            room.dt,
            spreading=False,
            max_lag=bp_max_lag,
            lag_offset=bp_lag_offset,
        )
        imgs[k, 2] = carve_free_space(
            grid, room.sources[sl], room.mics[sl], res[sl], room.dt, rel_threshold=carve_threshold
        )
    imgs[:, 1] = ell
    return PoseImages(imgs, res, h)


def aggregate(images: NDArray, n_poses: int | None = None) -> dict[str, NDArray[np.float64]]:
    """Sum per-pose images ``(..., K, 4, H, W)`` over the first ``n_poses`` poses."""
    images = np.asarray(images)
    k = images.shape[-4] if n_poses is None else int(n_poses)
    s = images[..., :k, :, :, :].sum(axis=-4)
    return {name: s[..., i, :, :] for i, name in enumerate(POSE_CHANNELS)}


def geometry_channels(
    grid_shape: tuple[int, int], source: NDArray, mics: NDArray, dt: float
) -> NDArray[np.float32]:
    """Pose geometry rendered on the grid: ``(1 + M, H, W)``.

    Channel 0 is the source distance :math:`\\lVert\\mathbf{x}-\\mathbf{s}\\rVert / N`;
    channel :math:`1 + m` is the bistatic travel lag
    :math:`t_{m}(\\mathbf{x})/\\Delta t` to mic :math:`m`, divided by
    :math:`4N` (so both lie in :math:`[0, \\approx 1.5]`). They tell a
    convolutional network where the pose's devices are and which pixels
    share an arrival time.
    """
    n = float(max(grid_shape))
    ii, jj = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing="ij")
    src = np.asarray(source, dtype=np.float64)
    out = [np.hypot(ii - src[0], jj - src[1]) / n]
    for m in np.asarray(mics).reshape(-1, 2):
        lag, _, _ = travel_lags(grid_shape, src, m, dt)
        out.append(lag / (4.0 * n))
    return np.stack(out).astype(np.float32)


def perturb_poses(
    room: ArchiveRoom, sigma: float, rng: np.random.Generator
) -> tuple[ArchiveRoom, NDArray[np.int64]]:
    """Room with assumed device cells = truth + ``round(N(0, sigma^2))`` per coordinate.

    Every device (source and each mic) is displaced independently, then
    clipped to the interior ``[1, N-2]`` (the outer ring is the p = 0 wall).
    The recordings are unchanged, since they were made at the true cells.

    Returns
    -------
    room, error
        The room with assumed cells, and the per-device integer error
        ``(K, 1 + M, 2)`` (source first).
    """
    n0, n1 = room.mask.shape
    lo = np.array([1, 1])
    hi = np.array([n0 - 2, n1 - 2])
    devs = np.concatenate([room.sources[:, None, :], room.mics], axis=1)
    noisy = np.clip(np.rint(devs + sigma * rng.standard_normal(devs.shape)), lo, hi)
    noisy = noisy.astype(np.int64)
    out = replace(room, sources=noisy[:, 0].copy(), mics=noisy[:, 1:].copy())
    return out, noisy - devs


__all__ = [
    "POSE_CHANNELS",
    "PoseImages",
    "compute_pose_images",
    "aggregate",
    "geometry_channels",
    "perturb_poses",
]
