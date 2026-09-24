"""Synthetic-aperture delay-and-sum imaging (plan Task 6.2.3).

Each pose :math:`k` contributes one source :math:`\\mathbf{s}_k` and two
mics :math:`\\mathbf{m}_{k,1}, \\mathbf{m}_{k,2}`. Carrying the laptop to
:math:`K` spots turns :math:`2K` bistatic pairs into one virtual array.
For a scatterer at pixel :math:`\\mathbf{x}`, the single-scattering echo
arrives at the exact 2D travel time

.. math::
    t_{km}(\\mathbf{x}) = \\frac{\\lVert \\mathbf{x} - \\mathbf{s}_k \\rVert
        + \\lVert \\mathbf{x} - \\mathbf{m}_{km} \\rVert}{c},

so the delay-and-sum (Kirchhoff-type) migration image is

.. math::
    I(\\mathbf{x}) = \\sum_{k=1}^{K} \\sum_{m} w_{km}(\\mathbf{x})\\;
        a_{km}\\bigl(t_{km}(\\mathbf{x})\\bigr),

where :math:`a_{km}` is the scattered impulse response (direct path and
outer-wall reverberation removed, see ``ir.py``) after a detection filter:

* ``"envelope"`` (default): the Hilbert envelope :math:`|h + j\\mathcal{H}h|`.
  Robust to the :math:`\\pi/4` phase of the 2D Green's function and to the
  polarity of the reflector (:math:`\\Gamma = -1` for the p = 0 obstacles).
* ``"signed"``: :math:`-h` (soft reflectors invert the pulse), coherent
  summation; sharper but sensitive to phase.
* ``"kirchhoff"``: :math:`-\\mathcal{F}^{-1}[(j\\omega)^{1/2} H]`, the 2D
  Kirchhoff migration filter that undoes the half-integration of the 2D
  Green's function (:math:`\\hat G \\propto (j\\omega)^{-1/2}` in the far
  field), followed by a coherent sum.

The weight :math:`w = \\sqrt{r_s r_m}` (``spreading=True``) compensates the
:math:`1/\\sqrt{r}` cylindrical spreading of each leg. An optional time gate
keeps only lags up to ``max_lag`` so late, multiply scattered energy does
not smear across the image.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .ir import envelope


def travel_lags(
    grid_shape: tuple[int, int],
    source: NDArray,
    mic: NDArray,
    dt: float,
    c: float = 1.0,
    dx: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Travel lag (in steps) and leg lengths from ``source`` via each pixel to ``mic``."""
    ii, jj = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing="ij")
    rs = dx * np.hypot(ii - source[0], jj - source[1])
    rm = dx * np.hypot(ii - mic[0], jj - mic[1])
    return (rs + rm) / (c * dt), rs, rm


def detection_filter(h: NDArray, kind: str = "envelope", dt: float = 1.0) -> NDArray[np.float64]:
    """Apply the detection filter named ``kind`` along the last axis."""
    h = np.asarray(h, dtype=np.float64)
    if kind == "envelope":
        return envelope(h)
    if kind == "signed":
        return -h
    if kind == "kirchhoff":
        n = h.shape[-1]
        n_fft = 2 * n
        H = np.fft.rfft(h, n_fft, axis=-1)
        w = 2 * np.pi * np.fft.rfftfreq(n_fft, d=dt)
        return -np.fft.irfft(H * np.sqrt(1j * w), n_fft, axis=-1)[..., :n]
    raise ValueError(f"unknown detection filter {kind!r}")


def sample_trace(trace: NDArray, lag: NDArray) -> NDArray[np.float64]:
    """Linear interpolation of ``trace`` at fractional ``lag`` (zero outside)."""
    n = trace.shape[-1]
    i0 = np.floor(lag).astype(np.int64)
    frac = lag - i0
    ok = (i0 >= 0) & (i0 + 1 < n)
    i0c = np.clip(i0, 0, n - 2)
    val = (1 - frac) * trace[i0c] + frac * trace[i0c + 1]
    return np.where(ok, val, 0.0)


def backproject(
    grid_shape: tuple[int, int],
    source_positions: NDArray,
    mic_positions: NDArray,
    irs: NDArray,
    dt: float,
    c: float = 1.0,
    dx: float = 1.0,
    kind: str = "envelope",
    spreading: bool = True,
    max_lag: int | None = None,
    normalise_traces: bool = True,
    lag_offset: float = 0.0,
) -> NDArray[np.float64]:
    """Delay-and-sum image over all poses and mics.

    Parameters
    ----------
    source_positions
        ``(K, 2)`` source cells.
    mic_positions
        ``(K, M, 2)`` mic cells.
    irs
        ``(K, M, L)`` scattered impulse responses (lag axis last).
    max_lag
        Ignore lags beyond this (a time gate); ``None`` keeps all.
    normalise_traces
        Divide each filtered trace by its RMS so every (pose, mic) pair
        votes with equal weight.
    lag_offset
        Constant added to the travel lag, e.g. to align the envelope peak of
        the band-limited wavelet with the geometric arrival.
    """
    irs = np.asarray(irs, dtype=np.float64)
    K, M, L = irs.shape
    filt = detection_filter(irs, kind, dt)
    if max_lag is not None and max_lag < L:
        filt[..., max_lag:] = 0.0
    img = np.zeros(grid_shape, dtype=np.float64)
    for k in range(K):
        for m in range(M):
            tr = filt[k, m]
            if normalise_traces:
                rms = float(np.sqrt(np.mean(tr**2)))
                if rms <= 0:
                    continue
                tr = tr / rms
            lag, rs, rm = travel_lags(
                grid_shape,
                np.asarray(source_positions[k]),
                np.asarray(mic_positions[k, m]),
                dt,
                c,
                dx,
            )
            val = sample_trace(tr, lag + lag_offset)
            if spreading:
                val = val * np.sqrt(np.maximum(rs, dx) * np.maximum(rm, dx))
            img += val
    return img


__all__ = ["travel_lags", "detection_filter", "sample_trace", "backproject"]
