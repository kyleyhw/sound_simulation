"""Reflector localisation from echo times (plan Task 6.2.2).

Image-source picture: a specular reflection off a planar surface arrives
as if emitted by the mirror image of the source. For a source
:math:`\\mathbf{s}` and a mic :math:`\\mathbf{m}`, an echo at delay
:math:`\\tau` (after emission) constrains the reflecting point to the
ellipse with foci :math:`\\mathbf{s}, \\mathbf{m}`

.. math::
    \\mathcal{E}(\\tau) = \\{\\mathbf{x} : \\lVert\\mathbf{x}-\\mathbf{s}\\rVert
        + \\lVert\\mathbf{x}-\\mathbf{m}\\rVert = c\\tau\\},

and the reflecting plane is tangent to it. When source and mic are
co-located (a laptop, a few cm apart) the ellipse is a circle of radius
:math:`c\\tau/2`. The v2 archives place the source and the mic pair
independently, so the general ellipse is used.

Two evidence maps are built from *picked* echoes (sparse, unlike the dense
delay-and-sum of ``backprojection.py``):

* **Echo ellipses**: every picked echo :math:`(\\tau_e, a_e)` adds
  :math:`a_e \\exp(-(t(\\mathbf{x}) - \\tau_e)^2 / 2\\sigma_t^2)` to the
  pose's map; stacking poses intersects the ellipses.
* **Free-space carving**: the *first* scattered arrival :math:`\\tau_1`
  bounds the nearest scatterer, so every pixel strictly inside
  :math:`\\mathcal{E}(\\tau_1)` is free (an obstacle there would have
  echoed earlier). The count of (pose, mic) ellipses containing a pixel is
  negative evidence for occupancy. This is the information that a filled
  occupancy mask can use most directly, because it clears whole regions
  rather than lighting up surfaces.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .backprojection import travel_lags


def pick_echoes(
    env: NDArray,
    max_echoes: int = 6,
    rel_threshold: float = 0.2,
    min_separation: int = 6,
    start: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Strongest local maxima of an envelope, returned in lag order.

    A lag :math:`j \\ge` ``start`` qualifies when it is a local maximum and
    :math:`e[j] \\ge` ``rel_threshold`` :math:`\\cdot \\max e`; peaks closer
    than ``min_separation`` lags to a stronger one are suppressed.
    """
    e = np.asarray(env, dtype=np.float64)
    if e.size < 3 or not np.any(e > 0):
        return np.zeros(0, np.int64), np.zeros(0)
    ref = float(e[start:].max()) if start < e.size else 0.0
    if ref <= 0:
        return np.zeros(0, np.int64), np.zeros(0)
    i = np.arange(max(start, 1), e.size - 1)
    peak = (e[i] >= e[i - 1]) & (e[i] > e[i + 1]) & (e[i] >= rel_threshold * ref)
    cand = i[peak]
    cand = cand[np.argsort(-e[cand], kind="stable")]
    picked: list[int] = []
    for j in cand:
        if all(abs(int(j) - p) >= min_separation for p in picked):
            picked.append(int(j))
        if len(picked) >= max_echoes:
            break
    lags = np.array(sorted(picked), dtype=np.int64)
    return lags, e[lags]


def first_arrival(trace: NDArray, rel_threshold: float = 0.1) -> int | None:
    """First lag at which :math:`|x|` reaches ``rel_threshold`` of its maximum.

    Works on an envelope or on the raw scattered residual. The noise-free
    FDTD residual is exactly zero until the first scattered wave arrives
    (up to a small numerical precursor a few steps ahead of the wavefront),
    so a low threshold on the raw residual gives a sharp onset.
    """
    e = np.abs(np.asarray(trace, dtype=np.float64))
    if e.size == 0 or e.max() <= 0:
        return None
    return int(np.argmax(e >= rel_threshold * e.max()))


def ellipse_evidence(
    grid_shape: tuple[int, int],
    source: NDArray,
    mic: NDArray,
    lags: NDArray,
    amps: NDArray,
    dt: float,
    sigma_lag: float = 2.0,
    c: float = 1.0,
    dx: float = 1.0,
    lag_offset: float = 0.0,
) -> NDArray[np.float64]:
    """Sum of Gaussian-blurred echo ellipses for one (source, mic) pair."""
    t, _, _ = travel_lags(grid_shape, np.asarray(source), np.asarray(mic), dt, c, dx)
    out = np.zeros(grid_shape, dtype=np.float64)
    for lag, a in zip(np.asarray(lags, float), np.asarray(amps, float)):
        out += a * np.exp(-0.5 * ((t + lag_offset - lag) / sigma_lag) ** 2)
    return out


def image_source_maps(
    grid_shape: tuple[int, int],
    source_positions: NDArray,
    mic_positions: NDArray,
    envs: NDArray,
    dt: float,
    max_echoes: int = 6,
    rel_threshold: float = 0.2,
    min_separation: int = 6,
    sigma_lag: float = 2.0,
    c: float = 1.0,
    dx: float = 1.0,
    lag_offset: float = 0.0,
) -> NDArray[np.float64]:
    """Per-pose echo-ellipse evidence ``(K, H, W)`` from envelopes ``(K, M, L)``.

    Echo amplitudes are normalised by the strongest echo of each trace, so
    every (pose, mic) pair contributes at most one unit per echo.
    """
    envs = np.asarray(envs, dtype=np.float64)
    K, M, _ = envs.shape
    maps = np.zeros((K,) + tuple(grid_shape), dtype=np.float64)
    for k in range(K):
        for m in range(M):
            lags, amps = pick_echoes(envs[k, m], max_echoes, rel_threshold, min_separation)
            if lags.size == 0:
                continue
            maps[k] += ellipse_evidence(
                grid_shape,
                source_positions[k],
                mic_positions[k, m],
                lags,
                amps / amps.max(),
                dt,
                sigma_lag,
                c,
                dx,
                lag_offset,
            )
    return maps


def carve_free_space(
    grid_shape: tuple[int, int],
    source_positions: NDArray,
    mic_positions: NDArray,
    traces: NDArray,
    dt: float,
    rel_threshold: float = 1e-3,
    lag_offset: float = 0.0,
    c: float = 1.0,
    dx: float = 1.0,
) -> NDArray[np.float64]:
    """Count of first-arrival ellipses that contain each pixel.

    ``traces`` ``(K, M, L)`` are scattered residuals (raw or envelopes). A
    pixel whose travel lag is below the detected onset plus ``lag_offset``
    (a correction for the detector's lead or lag, fitted on training
    rooms) lies inside the ellipse, hence is free for that (pose, mic) pair.
    """
    traces = np.asarray(traces, dtype=np.float64)
    K, M, _ = traces.shape
    count = np.zeros(grid_shape, dtype=np.float64)
    for k in range(K):
        for m in range(M):
            j1 = first_arrival(traces[k, m], rel_threshold)
            if j1 is None:
                continue
            t, _, _ = travel_lags(
                grid_shape,
                np.asarray(source_positions[k]),
                np.asarray(mic_positions[k, m]),
                dt,
                c,
                dx,
            )
            count += (t < j1 + lag_offset).astype(np.float64)
    return count


__all__ = [
    "pick_echoes",
    "first_arrival",
    "ellipse_evidence",
    "image_source_maps",
    "carve_free_space",
]
