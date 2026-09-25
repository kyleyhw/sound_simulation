"""Pose-robust sensing: a rigid pose-error model, rigid refinement and autofocus.

``pose_images.perturb_poses`` displaces every device (the source and each
mic) independently. A real device has a *known internal geometry* (the
speaker and the mics sit at fixed offsets in one chassis), so what a hand
placement gets wrong is one rigid transform per placement: a translation
:math:`\\mathbf{t}` and a rotation :math:`\\theta` of the whole pose about its
centroid :math:`\\mathbf{c}`,

.. math::
    \\hat{\\mathbf{d}}_i = \\operatorname{round}\\bigl(R(\\theta)
        (\\mathbf{d}_i - \\mathbf{c}) + \\mathbf{c} + \\mathbf{t}\\bigr),
    \\qquad \\mathbf{t} \\sim \\mathcal{N}(0, \\sigma_t^2 I),\\
    \\theta \\sim \\mathcal{N}(0, \\sigma_\\theta^2),

(:func:`perturb_poses_rigid`). Rounding to cells keeps the inter-device
distances to within a cell, not exactly.

Every correction below searches the same 3-DOF candidate set per pose
(:func:`rigid_candidates`: integer translations within a Chebyshev radius
and a few rotations about the assumed centroid, rounded and de-duplicated)
and scores each candidate once (:func:`evaluate_candidates`): the empty-box
recording :math:`y^{(0)}` at the candidate cells (one engine run per
distinct source cell), the data misfit against the recording, and small
per-pose *focus images* (back-projection and first-arrival carving of the
candidate's scattered residual). The selectors then differ only in what
they maximise:

* :func:`select_misfit` -- rigid least squares (or a robust loss) against
  the empty-box model, the 3-DOF analogue of ``pose_refine.refine_pose_joint``;
* :func:`autofocus_select` -- *autofocus*: pick one candidate per pose so
  that the per-pose images agree with each other, without any map model.
  With every per-pose image :math:`\\tilde I_k` zero-mean and of unit norm,
  the coherence :math:`\\lVert\\sum_k \\tilde I_k\\rVert^2 = K + 2\\sum_{k<l}
  \\langle\\tilde I_k, \\tilde I_l\\rangle` is the quadratic sharpness metric of
  SAR autofocus (Fienup and Miller 2003); ``metric="entropy"`` is the
  minimum-entropy variant. Coordinate ascent over poses, with an optional
  Gaussian prior penalty on the correction and an optional data-misfit term.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from .backprojection import backproject
from .image_source import carve_free_space
from .ir import TikhonovDeconvolver, empty_room_response
from .pipeline import ArchiveRoom
from .pose_refine import quiet_cost

# ---------------------------------------------------------------------------
# Rigid error model
# ---------------------------------------------------------------------------


def rigid_transform(
    points: ArrayLike, t: ArrayLike, theta_deg: float, center: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Rotate ``points`` ``(n, 2)`` by ``theta_deg`` about ``center`` (default: centroid), then shift by ``t``."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    c = p.mean(0) if center is None else np.asarray(center, dtype=np.float64)
    th = np.deg2rad(float(theta_deg))
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (p - c) @ rot.T + c + np.asarray(t, dtype=np.float64)


def _devices(room: ArchiveRoom) -> NDArray[np.int64]:
    """``(K, 1 + M, 2)`` device cells, source first."""
    return np.concatenate([room.sources[:, None, :], room.mics], axis=1).astype(np.int64)


def _with_devices(room: ArchiveRoom, devs: NDArray) -> ArchiveRoom:
    devs = np.asarray(devs, dtype=np.int64)
    return replace(room, sources=devs[:, 0].copy(), mics=devs[:, 1:].copy())


def perturb_poses_rigid(
    room: ArchiveRoom,
    sigma_t: float,
    sigma_theta_deg: float,
    rng: np.random.Generator,
) -> tuple[ArchiveRoom, NDArray[np.int64], NDArray[np.float64]]:
    """Room whose assumed cells are the true pose moved by one random rigid transform per pose.

    Per pose, :math:`\\mathbf{t} \\sim \\mathcal{N}(0, \\sigma_t^2 I)` (cells) and
    :math:`\\theta \\sim \\mathcal{N}(0, \\sigma_\\theta^2)` (degrees) about the
    pose centroid; the result is rounded to cells and clipped to the
    interior ``[1, N - 2]``. The recordings stay at the true cells.

    Returns
    -------
    room, error, params
        The room with assumed cells, the integer per-device error
        ``(K, 1 + M, 2)`` and the drawn ``(t_0, t_1, theta_deg)`` per pose.
    """
    n0, n1 = room.mask.shape
    lo, hi = np.array([1, 1]), np.array([n0 - 2, n1 - 2])
    devs = _devices(room)
    out = np.empty_like(devs)
    params = np.zeros((len(devs), 3))
    for k in range(len(devs)):
        t = sigma_t * rng.standard_normal(2)
        th = sigma_theta_deg * rng.standard_normal()
        params[k] = (t[0], t[1], th)
        out[k] = np.clip(np.rint(rigid_transform(devs[k], t, th)), lo, hi)
    return _with_devices(room, out), out - devs, params


# ---------------------------------------------------------------------------
# Candidate corrections
# ---------------------------------------------------------------------------


@dataclass
class RigidCandidates:
    """Distinct rigid corrections of one assumed pose.

    Attributes
    ----------
    params
        ``(C, 3)`` correction ``(t_0, t_1, theta_deg)`` applied to the assumed
        pose (row 0 is the identity).
    devices
        ``(C, 1 + M, 2)`` resulting cells, source first.
    """

    params: NDArray[np.float64]
    devices: NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.params)


def rigid_candidates(
    devices: NDArray,
    radius: int,
    thetas_deg: NDArray | list[float],
    grid_shape: tuple[int, int],
) -> RigidCandidates:
    """All corrections with integer translation in ``[-radius, radius]^2`` and rotation in ``thetas_deg``.

    Rotations are about the assumed pose's centroid. Candidates whose rounded
    cells coincide are merged, keeping the smallest correction (so row 0 is
    the identity); candidates leaving the interior are dropped.
    """
    d = np.asarray(devices, dtype=np.float64).reshape(-1, 2)
    n0, n1 = grid_shape
    rng_t = np.arange(-radius, radius + 1)
    raw = [(float(a), float(b), float(th)) for th in thetas_deg for a in rng_t for b in rng_t]
    if (0.0, 0.0, 0.0) not in raw:
        raw.append((0.0, 0.0, 0.0))
    raw.sort(key=lambda p: (p[0] ** 2 + p[1] ** 2, abs(p[2])))
    seen: set[bytes] = set()
    params, devs = [], []
    for a, b, th in raw:
        cells = np.rint(rigid_transform(d, (a, b), th)).astype(np.int64)
        if cells.min() < 1 or (cells[:, 0] > n0 - 2).any() or (cells[:, 1] > n1 - 2).any():
            continue
        key = cells.tobytes()
        if key in seen:
            continue
        seen.add(key)
        params.append((a, b, th))
        devs.append(cells)
    return RigidCandidates(np.asarray(params, dtype=np.float64), np.stack(devs))


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------


def huber_cost(r: NDArray, delta: float) -> NDArray[np.float64]:
    """:math:`\\sum_j \\rho_\\delta(r_j)` along the last axis (quadratic within ``delta``, linear outside)."""
    a = np.abs(np.asarray(r, dtype=np.float64))
    q = np.where(a <= delta, 0.5 * a**2, delta * (a - 0.5 * delta))
    return q.sum(-1)


@dataclass
class CandidateSet:
    """Scores of every rigid candidate of one pose.

    Attributes
    ----------
    candidates
        The corrections.
    misfit
        ``{"l2", "huber", "quiet"}`` -> ``(C,)``: data misfit of the empty-box
        recording at the candidate cells (summed over mics), lower is better.
    focus
        ``(C, 2, H, W)`` per-pose back-projection and carving images of the
        candidate's scattered residual (``None`` if not requested).
    n_samples
        Number of recorded samples :math:`MT` behind the misfits.
    """

    candidates: RigidCandidates
    misfit: dict[str, NDArray[np.float64]]
    focus: NDArray[np.float32] | None
    n_samples: int = 1


def evaluate_candidates(
    grid_shape: tuple[int, int],
    recordings: NDArray,
    drive: NDArray,
    candidates: RigidCandidates,
    dt: float,
    deconvolver: TikhonovDeconvolver | None = None,
    focus: bool = True,
    huber_rel: float = 0.05,
    bp_max_lag: int = 200,
    bp_lag_offset: float = -2.0,
    carve_threshold: float = 1e-3,
) -> CandidateSet:
    """Score every candidate of one pose (one engine run per distinct source cell).

    ``recordings`` is ``(M, T)`` at the true cells. ``huber_rel`` sets the
    Huber scale :math:`\\delta` relative to the recording RMS: samples whose
    misfit exceeds it (scattered arrivals, which the empty-box model cannot
    explain) count linearly instead of quadratically.
    """
    y = np.asarray(recordings, dtype=np.float64)
    M, T = y.shape
    devs = candidates.devices
    C = len(devs)
    y0 = np.zeros((C, M, T))
    src_keys = [tuple(int(v) for v in devs[c, 0]) for c in range(C)]
    for s in dict.fromkeys(src_keys):
        rows = [c for c in range(C) if src_keys[c] == s]
        cells = devs[rows, 1:].reshape(-1, 2)
        rec, _ = empty_room_response(grid_shape, s, drive, cells)
        y0[rows] = rec.reshape(len(rows), M, T)
    res = y[None] - y0
    delta = huber_rel * float(np.sqrt(np.mean(y**2))) + 1e-300
    misfit = {
        "l2": (res**2).sum((1, 2)),
        "huber": huber_cost(res, delta).sum(1),
        "quiet": np.stack([quiet_cost(y[m], y0[:, m].T) for m in range(M)]).sum(0),
    }
    img = None
    if focus:
        if deconvolver is None:
            deconvolver = TikhonovDeconvolver(np.asarray(drive), T, lam=1e-2)
        h = deconvolver(res)
        img = np.zeros((C, 2) + tuple(grid_shape), dtype=np.float32)
        for c in range(C):
            s1, m1 = devs[c, :1], devs[c, 1:][None]
            img[c, 0] = backproject(
                grid_shape,
                s1,
                m1,
                h[c : c + 1],
                dt,
                spreading=False,
                max_lag=bp_max_lag,
                lag_offset=bp_lag_offset,
            )
            img[c, 1] = carve_free_space(
                grid_shape, s1, m1, res[c : c + 1], dt, rel_threshold=carve_threshold
            )
    return CandidateSet(candidates, misfit, img, M * T)


def prior_penalty(params: NDArray, sigma_t: float, sigma_theta_deg: float) -> NDArray[np.float64]:
    """Negative log Gaussian prior of the corrections, :math:`\\tfrac12(|\\mathbf t|^2/\\sigma_t^2 + \\theta^2/\\sigma_\\theta^2)`."""
    p = np.asarray(params, dtype=np.float64)
    out = 0.5 * (p[:, 0] ** 2 + p[:, 1] ** 2) / max(sigma_t, 1e-6) ** 2
    if sigma_theta_deg > 0:
        out = out + 0.5 * p[:, 2] ** 2 / sigma_theta_deg**2
    return out


def relative_cost(cset: CandidateSet, loss: str = "l2") -> NDArray[np.float64]:
    """Misfit relative to the identity candidate: :math:`\\log(m/m_0)` for ``l2``/``huber``.

    ``quiet`` is already a sum of logs, so it is differenced and divided by
    the number of samples instead.
    """
    m = cset.misfit[loss]
    if loss == "quiet":
        return (m - m[0]) / max(cset.n_samples, 1)
    return np.log(m + 1e-300) - np.log(m[0] + 1e-300)


def select_misfit(
    cset: CandidateSet,
    loss: str = "l2",
    sigma_t: float | None = None,
    sigma_theta_deg: float = 0.0,
    prior_weight: float = 0.0,
) -> int:
    """Rigid least squares (``loss="l2"``) or a robust variant against the empty-box model.

    With ``prior_weight > 0`` the cost is the misfit normalised by its value
    at the identity, in log form, plus ``prior_weight`` times the Gaussian
    prior penalty (a MAP estimate under that prior).
    """
    m = cset.misfit[loss]
    if prior_weight <= 0 or sigma_t is None:
        return int(np.argmin(m))
    score = relative_cost(cset, loss) + prior_weight * prior_penalty(
        cset.candidates.params, sigma_t, sigma_theta_deg
    )
    return int(np.argmin(score))


def focus_features(
    focus: NDArray, channels: tuple[int, ...] = (0, 1), smooth: float = 1.0
) -> NDArray[np.float64]:
    """Flattened, zero-mean, unit-norm focus features ``(C, D)``.

    Each chosen channel is smoothed (Gaussian, ``smooth`` cells), centred and
    scaled to unit norm, and the channels are concatenated and rescaled so
    the feature vector has unit norm.
    """
    f = np.asarray(focus, dtype=np.float64)[:, list(channels)]
    if smooth > 0:
        f = ndimage.gaussian_filter(f, (0, 0, smooth, smooth))
    f = f - f.mean(axis=(2, 3), keepdims=True)
    f = f / (np.sqrt((f**2).sum(axis=(2, 3), keepdims=True)) + 1e-12)
    return f.reshape(len(f), -1) / np.sqrt(len(channels))


def autofocus_select(
    features: list[NDArray],
    penalties: list[NDArray] | None = None,
    data_costs: list[NDArray] | None = None,
    metric: str = "coherence",
    prior_weight: float = 0.0,
    data_weight: float = 0.0,
    n_iter: int = 6,
) -> list[int]:
    """Pick one candidate per pose to maximise cross-pose image agreement.

    Parameters
    ----------
    features
        Per pose ``(C_k, D)`` unit-norm features (:func:`focus_features`),
        row 0 the identity.
    penalties, data_costs
        Optional per-pose ``(C_k,)`` prior penalties and data costs (lower is
        better), weighted by ``prior_weight`` and ``data_weight``.
    metric
        ``"coherence"``: :math:`\\langle F_k, \\sum_{l\\ne k} F_l\\rangle`
        (the quadratic sharpness of :math:`\\sum_k F_k`), or ``"entropy"``:
        the negative Shannon entropy of the normalised intensity
        :math:`(\\sum_k F_k)^2`.

    Coordinate ascent from the identity; each sweep updates the poses in
    turn and stops when nothing changes. Every update cannot lower the
    objective, so the ascent terminates.
    """
    K = len(features)
    idx = [0] * K

    def extra(k: int) -> NDArray:
        e = np.zeros(len(features[k]))
        if penalties is not None and prior_weight > 0:
            e = e + prior_weight * penalties[k]
        if data_costs is not None and data_weight > 0:
            e = e + data_weight * data_costs[k]
        return e

    extras = [extra(k) for k in range(K)]
    for _ in range(n_iter):
        changed = False
        for k in range(K):
            others = sum(features[j][idx[j]] for j in range(K) if j != k)
            if K == 1:
                others = np.zeros(features[k].shape[1])
            if metric == "coherence":
                f = features[k] @ others
            elif metric == "entropy":
                inten = (features[k] + others[None]) ** 2
                p = inten / (inten.sum(1, keepdims=True) + 1e-300)
                f = (p * np.log(p + 1e-300)).sum(1)
            else:
                raise ValueError(f"unknown metric {metric!r}")
            best = int(np.argmax(f - extras[k]))
            if best != idx[k]:
                idx[k] = best
                changed = True
        if not changed:
            break
    return idx


# ---------------------------------------------------------------------------
# Whole-room drivers
# ---------------------------------------------------------------------------


def candidate_thetas(sigma_theta_deg: float, n_side: int = 2) -> list[float]:
    """Rotation grid ``{-2, -1, 0, 1, 2} * sigma_theta`` (only 0 when ``sigma_theta`` is 0)."""
    if sigma_theta_deg <= 0:
        return [0.0]
    return [float(sigma_theta_deg * i) for i in range(-n_side, n_side + 1)]


def analyse_room(
    room: ArchiveRoom,
    radius: int,
    thetas_deg: list[float],
    deconvolver: TikhonovDeconvolver | None = None,
    focus: bool = True,
    huber_rel: float = 0.05,
) -> list[CandidateSet]:
    """:func:`evaluate_candidates` for every pose of ``room`` (its cells are the assumed ones)."""
    grid = room.mask.shape
    T = room.recordings.shape[-1]
    if deconvolver is None and focus:
        deconvolver = TikhonovDeconvolver(room.drive, T, lam=1e-2)
    devs = _devices(room)
    out = []
    for k in range(len(devs)):
        cands = rigid_candidates(devs[k], radius, thetas_deg, grid)
        out.append(
            evaluate_candidates(
                grid,
                room.recordings[k],
                room.drive,
                cands,
                room.dt,
                deconvolver,
                focus=focus,
                huber_rel=huber_rel,
            )
        )
    return out


def apply_selection(room: ArchiveRoom, csets: list[CandidateSet], idx: list[int]) -> ArchiveRoom:
    """The room with each pose's cells replaced by its selected candidate."""
    devs = np.stack([cs.candidates.devices[i] for cs, i in zip(csets, idx)])
    return _with_devices(room, devs)


def refine_room_rigid(
    room: ArchiveRoom,
    radius: int,
    thetas_deg: list[float],
    loss: str = "l2",
) -> tuple[ArchiveRoom, list[int]]:
    """Rigid (3-DOF per pose) refinement of every pose against the empty-box model."""
    cs = analyse_room(room, radius, thetas_deg, focus=False)
    idx = [select_misfit(c, loss) for c in cs]
    return apply_selection(room, cs, idx), idx


def refine_room_rigid_polish(
    room: ArchiveRoom,
    radius: int,
    thetas_deg: list[float],
    loss: str = "l2",
    polish_radius: int = 1,
) -> tuple[ArchiveRoom, list]:
    """Rigid refinement, then a per-device joint least-squares polish around it.

    The rigid search fixes the bulk of a placement error with 3 unknowns per
    pose; the polish (``pose_refine.refine_pose_joint`` within
    ``polish_radius`` of every refined device) absorbs what a rigid model on
    a coarse rotation grid, rounded to cells, cannot: rotation steps between
    grid values and the up-to-half-cell rounding of each device. Returns the
    room and the per-pose ``RefinedPose`` of the polish (whose ``incident``
    fields feed ``compute_pose_images``).
    """
    from .pose_refine import refine_room

    rigid, _ = refine_room_rigid(room, radius, thetas_deg, loss)
    return refine_room(rigid, polish_radius, method="joint")


def autofocus_room(
    room: ArchiveRoom,
    radius: int,
    thetas_deg: list[float],
    sigma_t: float,
    sigma_theta_deg: float,
    metric: str = "coherence",
    channels: tuple[int, ...] = (0, 1),
    smooth: float = 1.0,
    prior_weight: float = 0.0,
    data_weight: float = 0.0,
    data_loss: str = "l2",
) -> tuple[ArchiveRoom, list[int]]:
    """Autofocus every pose of ``room`` (no map model; see :func:`autofocus_select`).

    ``data_weight > 0`` adds the log misfit (relative to the identity) of
    ``data_loss``, combining autofocus with the empty-box fit.
    """
    cs = analyse_room(room, radius, thetas_deg)
    feats = [focus_features(c.focus, channels, smooth) for c in cs if c.focus is not None]
    pens = [prior_penalty(c.candidates.params, sigma_t, sigma_theta_deg) for c in cs]
    data = [relative_cost(c, data_loss) for c in cs]
    idx = autofocus_select(
        feats, pens, data, metric=metric, prior_weight=prior_weight, data_weight=data_weight
    )
    return apply_selection(room, cs, idx), idx


def device_error(assumed: ArchiveRoom, truth: ArchiveRoom) -> NDArray[np.float64]:
    """Euclidean error ``(K, 1 + M)`` of every device, in cells."""
    return np.linalg.norm((_devices(assumed) - _devices(truth)).astype(np.float64), axis=-1)


__all__ = [
    "rigid_transform",
    "perturb_poses_rigid",
    "RigidCandidates",
    "rigid_candidates",
    "huber_cost",
    "CandidateSet",
    "evaluate_candidates",
    "prior_penalty",
    "select_misfit",
    "focus_features",
    "autofocus_select",
    "candidate_thetas",
    "analyse_room",
    "apply_selection",
    "refine_room_rigid",
    "refine_room_rigid_polish",
    "autofocus_room",
    "device_error",
]
