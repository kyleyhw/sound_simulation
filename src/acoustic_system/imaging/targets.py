"""Observable sensing targets (plan Task 6.1).

The Phase 2 target, the filled obstacle mask, is not fully observable
(plan audit A2): with :math:`p = 0` held inside an obstacle, its interior
never interacts with the field, and a surface in the acoustic shadow of
another obstacle scatters almost nothing back. This module derives three
targets from a room mask and the device poses.

Illuminated boundary (6.1.1)
----------------------------
A surface cell (an obstacle cell with a 4-neighbour in air) can return an
echo only if sound reaches it from the source *and* can travel from it to a
mic. Paths are straight rays (geometric acoustics) either direct or with one
specular bounce off the outer walls (first order). With the mirror plane of
the p = 0 wall at index :math:`0` and :math:`N-1`, the image of
:math:`(i, j)` is :math:`(-i, j)`, :math:`(2(N-1) - i, j)`, and likewise in
:math:`j`. A bounce path from image :math:`\\mathbf{s}'` to cell
:math:`\\mathbf{x}` is the straight segment in the unfolded plane; folding
the out-of-room part back across the wall gives the physical path, which
must avoid obstacle cells. A pose illuminates :math:`\\mathbf{x}` when the
source reaches it and at least one of the pose's mics sees it
(``mode="pose"``); ``mode="any"`` needs any single device. The union over
poses is the target. Higher-order bounces, diffraction and paths via other
obstacles are ignored, so the target is a conservative geometric proxy.

Room outline (6.1.2)
--------------------
:func:`obstacle_polygons` traces each obstacle's boundary as a closed
polygon (marching squares at the 0.5 level, vertices on half-cell
positions) and simplifies it with Douglas-Peucker.
:func:`free_space_outline` traces the free region reachable from the
devices, i.e. the room as a mapper would draw it: its outer outline plus
holes for obstacles it encloses. :func:`rasterize_polygons` maps polygons
back to cells (even-odd rule), so a polygon predictor can be scored with
the same pixel metrics.

Signed distance field (6.1.3)
-----------------------------
.. math::
    \\phi(\\mathbf{x}) = \\begin{cases}
      d(\\mathbf{x}, \\text{obstacles}) - \\tfrac12 & \\mathbf{x} \\text{ free} \\\\
      -\\bigl(d(\\mathbf{x}, \\text{air}) - \\tfrac12\\bigr) & \\mathbf{x} \\text{ occupied,}
    \\end{cases}

with Euclidean distances between cell centres, so the zero level set sits on
the cell faces. It is smooth and dense (every pixel carries a regression
value), unlike the 7 %-positive occupancy mask.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..learning.metrics import boundary

# ---------------------------------------------------------------------------
# 6.1.1 Illuminated boundary
# ---------------------------------------------------------------------------


def surface_cells(mask: NDArray) -> NDArray[np.bool_]:
    """Obstacle cells with at least one 4-neighbour in air."""
    return boundary(np.asarray(mask, dtype=bool))


def _images(p: NDArray, n: tuple[int, int]) -> list[NDArray]:
    """The point itself and its four first-order images across the outer walls."""
    i, j = float(p[0]), float(p[1])
    ni, nj = n[0] - 1, n[1] - 1
    return [
        np.array([i, j]),
        np.array([-i, j]),
        np.array([2 * ni - i, j]),
        np.array([i, -j]),
        np.array([i, 2 * nj - j]),
    ]


def ray_visible(
    mask: NDArray,
    origin: NDArray,
    cells: NDArray,
    step: float = 0.25,
    end_tolerance: float = 0.75,
) -> NDArray[np.bool_]:
    """Which ``cells`` a straight (possibly unfolded) ray from ``origin`` reaches.

    ``origin`` may lie outside the grid (a first-order image); the ray is then
    folded back across the outer wall. Folding acts per axis, so a ray that
    crosses two walls near a corner is folded twice (a corner bounce,
    physically a valid second-order path). The ray is sampled every
    ``step`` cells; samples within ``end_tolerance`` of the target cell
    centre are not tested (the target is itself an obstacle cell). A path
    is rejected if a folded sample hits an obstacle cell.
    """
    m = np.asarray(mask, dtype=bool)
    ni, nj = m.shape
    cells = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
    if cells.size == 0:
        return np.zeros(0, dtype=bool)
    o = np.asarray(origin, dtype=np.float64)
    d = cells - o[None, :]
    length = np.hypot(d[:, 0], d[:, 1])
    n_s = int(np.ceil(length.max() / step)) + 1
    t = np.linspace(0.0, 1.0, n_s)[None, :]
    pi = o[0] + t * d[:, :1]
    pj = o[1] + t * d[:, 1:]
    # Fold first-order excursions back into the room.
    pi = np.where(pi < 0, -pi, np.where(pi > ni - 1, 2 * (ni - 1) - pi, pi))
    pj = np.where(pj < 0, -pj, np.where(pj > nj - 1, 2 * (nj - 1) - pj, pj))
    ii = np.rint(pi).astype(np.int64)
    jj = np.rint(pj).astype(np.int64)
    inside = (ii >= 0) & (ii < ni) & (jj >= 0) & (jj < nj)
    blocked = ~inside | m[np.clip(ii, 0, ni - 1), np.clip(jj, 0, nj - 1)]
    near_end = (1.0 - t) * length[:, None] < end_tolerance
    return ~np.any(blocked & ~near_end, axis=1)


def reachable_from(
    mask: NDArray, device: NDArray, cells: NDArray, first_order: bool = True
) -> NDArray[np.bool_]:
    """Cells reached from ``device`` directly or via one outer-wall bounce."""
    m = np.asarray(mask, dtype=bool)
    origins = _images(np.asarray(device), m.shape) if first_order else [np.asarray(device, float)]
    vis = np.zeros(len(cells), dtype=bool)
    for o in origins:
        todo = ~vis
        if not todo.any():
            break
        vis[todo] |= ray_visible(m, o, np.asarray(cells)[todo])
    return vis


def illuminated_boundary(
    mask: NDArray,
    source_positions: NDArray,
    mic_positions: NDArray,
    mode: str = "pose",
    first_order: bool = True,
) -> NDArray[np.bool_]:
    """Surface cells an echo can come back from (see module docstring).

    Parameters
    ----------
    source_positions
        ``(K, 2)`` source cells.
    mic_positions
        ``(K, M, 2)`` mic cells.
    mode
        ``"pose"``: reached by the pose's source and seen by one of its mics;
        ``"any"``: reached from any single device position.
    """
    m = np.asarray(mask, dtype=bool)
    surf = surface_cells(m)
    cells = np.argwhere(surf)
    out = np.zeros_like(m)
    if cells.size == 0:
        return out
    src = np.asarray(source_positions).reshape(-1, 2)
    mics = np.asarray(mic_positions).reshape(len(src), -1, 2)
    hit = np.zeros(len(cells), dtype=bool)
    for k in range(len(src)):
        from_src = reachable_from(m, src[k], cells, first_order)
        from_mic = np.zeros(len(cells), dtype=bool)
        for q in mics[k]:
            if mode == "pose":
                sel = from_src & ~from_mic
                if sel.any():
                    from_mic[sel] |= reachable_from(m, q, cells[sel], first_order)
            else:
                from_mic |= reachable_from(m, q, cells, first_order)
        if mode == "pose":
            hit |= from_src & from_mic
        elif mode == "any":
            hit |= from_src | from_mic
        else:
            raise ValueError(f"mode must be 'pose' or 'any', got {mode!r}")
    out[cells[:, 0], cells[:, 1]] = hit
    return out


# ---------------------------------------------------------------------------
# 6.1.2 Polygons
# ---------------------------------------------------------------------------


def _douglas_peucker(pts: NDArray, tol: float) -> NDArray:
    """Douglas-Peucker simplification of an open polyline."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    nab = float(np.hypot(*ab))
    rel = pts - a
    if nab == 0:
        dist = np.hypot(rel[:, 0], rel[:, 1])
    else:
        dist = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / nab
    k = int(np.argmax(dist))
    if dist[k] <= tol:
        return np.stack([a, b])
    left = _douglas_peucker(pts[: k + 1], tol)
    right = _douglas_peucker(pts[k:], tol)
    return np.concatenate([left[:-1], right])


def simplify_polygon(poly: NDArray, tol: float = 0.25) -> NDArray:
    """Simplify a closed polygon (first vertex not repeated) within ``tol`` cells."""
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 4:
        return p
    # Split at the vertex farthest from vertex 0 so both halves are open chains.
    far = int(np.argmax(np.hypot(*(p - p[0]).T)))
    ring = np.concatenate([p, p[:1]])
    a = _douglas_peucker(ring[: far + 1], tol)
    b = _douglas_peucker(ring[far:], tol)
    return np.concatenate([a[:-1], b[:-1]])


def _trace(region: NDArray) -> list[NDArray]:
    """Closed contours (vertex arrays, ``(i, j)``) of a boolean region."""
    import contourpy

    z = np.pad(np.asarray(region, dtype=np.float64), 1)
    gen = contourpy.contour_generator(z=z, line_type="Separate")
    lines = gen.lines(0.5)
    polys = []
    for ln in lines:
        arr = np.asarray(ln, dtype=np.float64)
        # contourpy returns (x, y) = (column, row) in padded coordinates.
        ij = np.stack([arr[:, 1] - 1.0, arr[:, 0] - 1.0], axis=1)
        if len(ij) > 1 and np.allclose(ij[0], ij[-1]):
            ij = ij[:-1]
        if len(ij) >= 3:
            polys.append(ij)
    return polys


def obstacle_polygons(mask: NDArray, tol: float = 0.25) -> list[NDArray]:
    """One simplified closed polygon per obstacle contour (outer and hole rings)."""
    return [simplify_polygon(p, tol) for p in _trace(np.asarray(mask, dtype=bool))]


def free_space_outline(
    mask: NDArray, seeds: Sequence[Sequence[int]], tol: float = 0.25
) -> tuple[NDArray[np.bool_], list[NDArray]]:
    """Free region 4-connected to ``seeds`` and its simplified outline polygons.

    The outer p = 0 wall cells (index 0 and N-1) are treated as walls.
    Returns ``(region, polygons)``.
    """
    m = np.asarray(mask, dtype=bool).copy()
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    lab, _ = ndimage.label(~m)
    ids = {int(lab[tuple(int(c) for c in s)]) for s in seeds}
    ids.discard(0)
    region = np.isin(lab, list(ids)) if ids else np.zeros_like(m)
    return region, [simplify_polygon(p, tol) for p in _trace(region)]


def rasterize_polygons(polys: Sequence[NDArray], shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Cells whose centres fall inside the polygons (even-odd rule)."""
    from matplotlib.path import Path

    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    pts = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(np.float64)
    inside = np.zeros(len(pts), dtype=bool)
    for p in polys:
        if len(p) >= 3:
            inside ^= Path(np.asarray(p, dtype=np.float64)).contains_points(pts)
    return inside.reshape(shape)


# ---------------------------------------------------------------------------
# 6.1.3 Signed distance field
# ---------------------------------------------------------------------------


def signed_distance(
    mask: NDArray, include_walls: bool = False, truncate: float | None = None
) -> NDArray[np.float64]:
    """Signed distance to the obstacle surface (positive in air), in cells.

    ``include_walls`` counts the outer p = 0 wall cells as obstacles;
    ``truncate`` clips :math:`|\\phi|` (a TSDF). A room with no obstacle
    (and ``include_walls=False``) returns ``+inf`` everywhere.
    """
    m = np.asarray(mask, dtype=bool).copy()
    if include_walls:
        m[0, :] = m[-1, :] = True
        m[:, 0] = m[:, -1] = True
    if not m.any():
        out = np.full(m.shape, np.inf)
    elif m.all():
        out = np.full(m.shape, -np.inf)
    else:
        d_out = ndimage.distance_transform_edt(~m)
        d_in = ndimage.distance_transform_edt(m)
        out = np.where(m, -(np.asarray(d_in) - 0.5), np.asarray(d_out) - 0.5)
    if truncate is not None:
        out = np.clip(out, -truncate, truncate)
    return out


__all__ = [
    "surface_cells",
    "ray_visible",
    "reachable_from",
    "illuminated_boundary",
    "simplify_polygon",
    "obstacle_polygons",
    "free_space_outline",
    "rasterize_polygons",
    "signed_distance",
]
