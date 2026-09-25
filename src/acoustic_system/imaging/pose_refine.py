"""Pose refinement from the known background (plan Task 6.6.2).

When the laptop positions are only roughly known (6.6.1), both halves of
the imaging chain break: the empty-room background
:math:`y^{(0)}(\\hat{\\mathbf{s}}, \\hat{\\mathbf{m}})` is simulated at the
wrong cells, so the residual :math:`r = y - y^{(0)}` keeps a large
direct-path error from the direct arrival on, and every travel time is
wrong.

The residual also says how to fix it. At the true cells the noise-free
residual is *exactly zero until the first scattered arrival* :math:`t_1`
(``docs/imaging.md`` §1). A plain least-squares fit of the whole
recording mixes this with the unknown scattering (the p = 0 obstacles
scatter so strongly that the residual energy after :math:`t_1` often
exceeds the recording's own), and searched cell by cell it stalls in
local minima of the oscillating chirp misfit. The alternating search
below therefore uses an objective that rewards a *long quiet start*,

.. math::
    J(\\hat{\\mathbf{s}}, \\hat{\\mathbf{m}}) = \\sum_m \\sum_{n=0}^{T-1}
        \\log\\Bigl(\\epsilon E_m + \\sum_{j \\le n}
        \\bigl(y_m[j] - G_{\\hat{\\mathbf{s}}\\to\\hat{\\mathbf{m}}_m}[j]\\bigr)^2\\Bigr),
    \\qquad E_m = \\sum_j y_m[j]^2,

where :math:`G_{\\mathbf{a}\\to\\mathbf{b}}` is the engine's empty-box
recording at :math:`\\mathbf{b}` for the drive played at :math:`\\mathbf{a}`.
Each sample before the residual onset costs :math:`\\log\\epsilon E` (very
negative); after it, the log of the cumulative residual energy, which
depends only weakly on the pose. Minimising :math:`J` therefore maximises
the quiet time and, among equal quiet times, the early fit.

**Identifiability.** Before the first wall echo, the direct wave from
:math:`\\mathbf{s}` only fixes the distance :math:`\\lVert\\mathbf{s} -
\\mathbf{m}\\rVert`; a pose translated as a whole, or a mic moved along
the circle around the source, looks the same. Outer-wall echoes break the
symmetry, but only if they arrive before :math:`t_1`. On the v2 training
rooms the scattered onset trails the direct arrival by a median of only 4
lags (39 % of the traces lead it: the FDTD precursor, or an obstacle near
the direct path), so poses are rarely recovered *exactly*. What the
refinement does recover is a pose whose empty-box response matches the
data, which is what the background subtraction and the travel times
need: on training rooms it restores most of the imaging quality lost to
pose error, even where the cells stay wrong.

**Two searches.** :func:`refine_pose_joint` minimises the plain
least-squares misfit :math:`\\sum_m \\lVert y_m - G_{\\hat{\\mathbf{s}}\\to
\\hat{\\mathbf{m}}_m}\\rVert^2` exhaustively over the source window (one
simulation per candidate source; the mics then separate). With the true
obstacle map in place of the empty box (``mask=``) it puts 93 % of devices
on their exact cell; with the empty box it is the stronger of the two on
the imaging metric.
:func:`refine_pose` is the cheap alternative below.

**Alternating search.** One simulation from a source gives
:math:`G_{\\mathbf{s}\\to\\mathbf{x}}` at *every* cell :math:`\\mathbf{x}`,
so the best mic cells for a fixed source are an exhaustive search over a
window around the assumed cell. The discrete scheme is reciprocal (the
5-point Laplacian with Dirichlet walls is symmetric, and injection and
recording act on the same :math:`p^{n+1}`), so
:math:`G_{\\mathbf{a}\\to\\mathbf{b}} = G_{\\mathbf{b}\\to\\mathbf{a}}`
(checked to :math:`10^{-6}` relative in ``tests/imaging``), and one
simulation from each estimated mic gives the cost of every candidate
source cell. Alternating the two exhaustive searches costs :math:`M` or
:math:`M + 1` simulations per iteration and stops when nothing moves. The
window radius encodes the pose prior (the evaluation uses
:math:`R = \\lceil 2\\sigma \\rceil`); it also rules out mirror-image
poses of the symmetric box.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ..simulation.setup import Driver
from ..simulation.simulate import Simulate
from .ir import SampledWaveform, empty_room_response
from .pipeline import ArchiveRoom


@dataclass
class RefinedPose:
    """Result of :func:`refine_pose` for one pose."""

    source: NDArray[np.int64]  # (2,)
    mics: NDArray[np.int64]  # (M, 2)
    cost: float  # J at the estimate
    cost_assumed: float  # J at the assumed cells
    iterations: int
    incident: tuple[NDArray[np.float64], NDArray[np.float32]]  # (y0, P) at the estimate


def window_cells(center: NDArray, radius: int, grid_shape: tuple[int, int]) -> NDArray[np.int64]:
    """Interior cells within Chebyshev ``radius`` of ``center`` (``(n, 2)``)."""
    c = np.asarray(center, dtype=np.int64)
    lo0, hi0 = max(1, int(c[0]) - radius), min(grid_shape[0] - 2, int(c[0]) + radius)
    lo1, hi1 = max(1, int(c[1]) - radius), min(grid_shape[1] - 2, int(c[1]) + radius)
    ii, jj = np.meshgrid(np.arange(lo0, hi0 + 1), np.arange(lo1, hi1 + 1), indexing="ij")
    return np.stack([ii.ravel(), jj.ravel()], 1)


def quiet_cost(target: NDArray, traces: NDArray, eps: float = 1e-8) -> NDArray[np.float64]:
    """:math:`\\sum_n \\log(\\epsilon E + C_n)` for each column of ``traces`` ``(T, n)``.

    :math:`C_n = \\sum_{j \\le n} (y_j - g_j)^2` is the cumulative residual
    energy of the candidate trace :math:`g` against the recording
    :math:`y` (``target``, ``(T,)``) and :math:`E = \\sum_j y_j^2`.
    """
    y = np.asarray(target, dtype=np.float64)
    r = y[:, None] - np.asarray(traces, dtype=np.float64)
    c = np.cumsum(r**2, axis=0)
    e = float(np.sum(y**2)) + 1e-300
    return np.log(eps * e + c).sum(0)


def _cell_costs(field: NDArray, target: NDArray, cells: NDArray, eps: float) -> NDArray:
    return quiet_cost(target, field[:, cells[:, 0], cells[:, 1]], eps)


def refine_pose(
    grid_shape: tuple[int, int],
    recordings: NDArray,
    drive: NDArray,
    source: NDArray,
    mics: NDArray,
    radius: int,
    eps: float = 1e-8,
    max_iter: int = 6,
) -> RefinedPose:
    """Refine one pose (source + ``M`` mics) by alternating exhaustive searches.

    Parameters
    ----------
    recordings
        ``(M, T)`` recorded pressure at the true (unknown) cells.
    source, mics
        Assumed cells ``(2,)`` and ``(M, 2)``; the search stays within
        ``radius`` (Chebyshev) of them.
    eps
        Floor :math:`\\epsilon` of the cumulative residual energy, relative
        to the recording energy.
    """
    y = np.asarray(recordings, dtype=np.float64)
    M = y.shape[0]
    s0 = np.asarray(source, dtype=np.int64)
    m0 = np.asarray(mics, dtype=np.int64).reshape(M, 2)
    src_cells = window_cells(s0, radius, grid_shape)
    mic_cells = [window_cells(m0[i], radius, grid_shape) for i in range(M)]

    def cost_at(field: NDArray, mic_pos: NDArray) -> float:
        return float(sum(_cell_costs(field, y[i], mic_pos[i : i + 1], eps)[0] for i in range(M)))

    s, m = s0.copy(), m0.copy()
    _, P = empty_room_response(grid_shape, s, drive, m, return_field=True)
    assert P is not None
    c_assumed = cost_at(P, m0)
    it = 0
    for it in range(1, max_iter + 1):
        m_new = np.stack(
            [
                mic_cells[i][int(np.argmin(_cell_costs(P, y[i], mic_cells[i], eps)))]
                for i in range(M)
            ]
        )
        # Reciprocity: a simulation from each mic scores every candidate source cell.
        cost = np.zeros(len(src_cells))
        for i in range(M):
            _, G = empty_room_response(grid_shape, m_new[i], drive, m_new[i], return_field=True)
            assert G is not None
            cost += _cell_costs(G, y[i], src_cells, eps)
        s_new = src_cells[int(np.argmin(cost))]
        source_moved = not np.array_equal(s_new, s)
        moved = source_moved or not np.array_equal(m_new, m)
        s, m = s_new, m_new
        if source_moved:  # the incident field depends on the source only
            _, P = empty_room_response(grid_shape, s, drive, m, return_field=True)
            assert P is not None
        if not moved:
            break
    y0 = P[:, m[:, 0], m[:, 1]].astype(np.float64).T
    return RefinedPose(s, m, cost_at(P, m), c_assumed, it, (y0, P))


def _field(
    grid_shape: tuple[int, int], source: NDArray, drive: NDArray, mask: NDArray | None
) -> NDArray[np.float32]:
    """Incident field ``(T, H, W)`` of ``drive`` at ``source`` (empty box, or with ``mask``)."""
    if mask is None:
        _, P = empty_room_response(grid_shape, source, drive, [source], return_field=True)
        assert P is not None
        return P
    sim = Simulate(grid_shape=grid_shape)
    sim.set_obstacle_mask(np.asarray(mask, dtype=bool))
    sim.set_drivers(
        [Driver(tuple(int(c) for c in source), SampledWaveform(np.asarray(drive), sim.timestep))]
    )
    out = np.empty((len(drive),) + tuple(grid_shape), dtype=np.float32)
    for n in range(len(drive)):
        sim.step()
        out[n] = sim.p
    return out


def refine_pose_joint(
    grid_shape: tuple[int, int],
    recordings: NDArray,
    drive: NDArray,
    source: NDArray,
    mics: NDArray,
    radius: int,
    mask: NDArray | None = None,
) -> RefinedPose:
    """Exhaustive joint least-squares fit of one pose to the empty-box model.

    .. math::
        (\\hat{\\mathbf{s}}, \\hat{\\mathbf{m}}) = \\arg\\min \\sum_m
            \\lVert y_m - G_{\\hat{\\mathbf{s}}\\to\\hat{\\mathbf{m}}_m} \\rVert^2

    over all source cells in the window and, for each, the best cell of
    every mic (the cost separates over mics once the source is fixed).
    One simulation per candidate source, so :math:`(2R+1)^2` in all. Unlike
    :func:`refine_pose`, this cannot stall in a coordinate-wise local
    minimum. ``mask`` replaces the empty box by a room with those obstacles
    (with the true map the fit is exact at the true cells; the
    returned ``incident`` is then that room's field, not the empty box's).
    """
    y = np.asarray(recordings, dtype=np.float64)
    M = y.shape[0]
    s0 = np.asarray(source, dtype=np.int64)
    m0 = np.asarray(mics, dtype=np.int64).reshape(M, 2)
    src_cells = window_cells(s0, radius, grid_shape)
    mic_cells = [window_cells(m0[i], radius, grid_shape) for i in range(M)]
    best = (np.inf, s0, m0, None)
    c_assumed = float("nan")
    for s in src_cells:
        P = _field(grid_shape, s, drive, mask)
        tot = 0.0
        picks = []
        for i in range(M):
            tr = P[:, mic_cells[i][:, 0], mic_cells[i][:, 1]].astype(np.float64)
            c = ((tr - y[i][:, None]) ** 2).sum(0)
            j = int(np.argmin(c))
            tot += float(c[j])
            picks.append(mic_cells[i][j])
        if np.array_equal(s, s0):
            c_assumed = float(sum(((P[:, m0[i, 0], m0[i, 1]] - y[i]) ** 2).sum() for i in range(M)))
        if tot < best[0]:
            best = (tot, s.copy(), np.stack(picks), P)
    tot, s, m, P = best
    assert P is not None
    y0 = P[:, m[:, 0], m[:, 1]].astype(np.float64).T
    return RefinedPose(s, m, float(tot), c_assumed, 1, (y0, P))


def refine_room(
    room: ArchiveRoom,
    radius: int,
    eps: float = 1e-8,
    max_iter: int = 6,
    method: str = "alternating",
) -> tuple[ArchiveRoom, list[RefinedPose]]:
    """Refine every pose of ``room`` (its stored cells are the assumed ones).

    ``method="alternating"`` uses :func:`refine_pose` (quiet-start cost,
    cheap); ``method="joint"`` uses :func:`refine_pose_joint`.
    """
    K = len(room.sources)
    if method == "alternating":
        out = [
            refine_pose(
                room.mask.shape,
                room.recordings[k],
                room.drive,
                room.sources[k],
                room.mics[k],
                radius,
                eps,
                max_iter,
            )
            for k in range(K)
        ]
    elif method == "joint":
        out = [
            refine_pose_joint(
                room.mask.shape,
                room.recordings[k],
                room.drive,
                room.sources[k],
                room.mics[k],
                radius,
            )
            for k in range(K)
        ]
    else:
        raise ValueError(f"unknown method {method!r}")
    refined = replace(
        room,
        sources=np.stack([r.source for r in out]),
        mics=np.stack([r.mics for r in out]),
    )
    return refined, out


__all__ = [
    "RefinedPose",
    "window_cells",
    "quiet_cost",
    "refine_pose",
    "refine_pose_joint",
    "refine_room",
]
