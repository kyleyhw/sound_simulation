"""Passive imaging with an unknown source signal (plan Task 6.6.4).

The active imagers deconvolve the *known* chirp. Here the source waveform
:math:`s` is unknown (the device positions, including the speaker's, are
still known). For two mics, :math:`Y_i(f) = H_i(f) S(f)`, and the
phase-transform cross-spectrum

.. math::
    \\Phi(f) = \\frac{Y_1(f) Y_2^*(f)}{|Y_1(f) Y_2^*(f)|}
             = \\frac{H_1(f) H_2^*(f)}{|H_1(f) H_2^*(f)|}

does not depend on :math:`S` wherever :math:`S(f) \\ne 0` (GCC-PHAT). Its
inverse transform :math:`g(\\tau)` peaks at the arrival-time differences
between the paths into mic 1 and into mic 2. The strongest are the
direct-direct peak and the *direct-scattered* cross terms: a scatterer at
:math:`\\mathbf{x}` puts energy at

.. math::
    \\tau_1(\\mathbf{x}) = t(\\mathbf{s}\\to\\mathbf{x}\\to\\mathbf{m}_1) -
        t(\\mathbf{s}\\to\\mathbf{m}_2), \\qquad
    \\tau_2(\\mathbf{x}) = t(\\mathbf{s}\\to\\mathbf{m}_1) -
        t(\\mathbf{s}\\to\\mathbf{x}\\to\\mathbf{m}_2),

so the direct wave at the other mic serves as the reference signal that
the unknown source denies us. The interferometric image migrates the
envelope of :math:`g` along both:

.. math::
    I(\\mathbf{x}) = \\sum_k \\sum_{m < m'} \\bigl|g_{k,mm'}\\bigr|
        \\bigl(\\tau_1(\\mathbf{x})\\bigr) + \\bigl|g_{k,mm'}\\bigr|
        \\bigl(\\tau_2(\\mathbf{x})\\bigr).

**Background.** The outer box also scatters. Its GCC-PHAT, computed from
the engine's empty-box *impulse* response (a probe of our own choosing;
the phase transform removes the probe spectrum), is subtracted from
:math:`|g|` before migration (``background=True``). Only the band where
the recordings themselves carry energy is used; it is read from the data,
not from the source.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .backprojection import sample_trace, travel_lags
from .ir import empty_room_response, envelope


def band_mask(spec: NDArray, rel: float = 1e-3) -> NDArray[np.bool_]:
    """Frequency bins whose magnitude exceeds ``rel`` of the maximum."""
    a = np.abs(spec)
    return a >= rel * a.max()


def gcc_phat(
    y1: NDArray, y2: NDArray, n_fft: int, band: NDArray | None = None, rel: float = 1e-3
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """GCC-PHAT of two traces, circularly shifted so lag 0 is at index ``n_fft // 2``.

    Returns ``(g, band)``; ``g[n_fft // 2 + tau]`` is the correlation at a
    delay of mic 1 relative to mic 2 of ``tau`` samples. ``band`` defaults
    to the bins where :math:`|Y_1 Y_2^*|` exceeds ``rel`` of its maximum.
    """
    Y1 = np.fft.rfft(np.asarray(y1, dtype=np.float64), n_fft)
    Y2 = np.fft.rfft(np.asarray(y2, dtype=np.float64), n_fft)
    X = Y1 * np.conj(Y2)
    if band is None:
        band = band_mask(X, rel)
    phi = np.where(band, X / (np.abs(X) + 1e-300), 0.0)
    g = np.fft.irfft(phi, n_fft)
    return np.roll(g, n_fft // 2), band


def empty_box_impulse(
    grid_shape: tuple[int, int], source: NDArray, mics: NDArray, n_steps: int
) -> NDArray[np.float64]:
    """Engine impulse response of the empty box, ``(M, n_steps)`` (a unit kick at step 0)."""
    probe = np.zeros(n_steps)
    probe[0] = 1.0
    rec, _ = empty_room_response(grid_shape, source, probe, mics)
    return rec


def interferometric_image(
    grid_shape: tuple[int, int],
    source: NDArray,
    mics: NDArray,
    traces: NDArray,
    dt: float,
    n_fft: int,
) -> NDArray[np.float64]:
    """Migrate the GCC envelopes ``traces`` ``(n_pairs, n_fft)`` of one pose.

    Pairs are ordered ``(0, 1), (0, 2), ..., (1, 2), ...``.
    """
    mics = np.asarray(mics).reshape(-1, 2)
    src = np.asarray(source, dtype=np.float64)
    lag_scat = [travel_lags(grid_shape, src, m, dt)[0] for m in mics]
    lag_dir = [float(np.hypot(*(np.asarray(m, dtype=np.float64) - src)) / dt) for m in mics]
    img = np.zeros(grid_shape)
    c = n_fft // 2
    p = 0
    for a in range(len(mics)):
        for b in range(a + 1, len(mics)):
            tr = traces[p]
            img += sample_trace(tr, c + lag_scat[a] - lag_dir[b])
            img += sample_trace(tr, c + lag_dir[a] - lag_scat[b])
            p += 1
    return img


def passive_images(
    grid_shape: tuple[int, int],
    sources: NDArray,
    mics: NDArray,
    recordings: NDArray,
    dt: float,
    n_fft: int = 1024,
    rel: float = 1e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interferometric images without and with empty-box background removal.

    Parameters
    ----------
    recordings
        ``(K, M, T)`` recordings; the source signal is never used.

    Returns
    -------
    raw, background_removed
        Two ``(H, W)`` images summed over poses. Each pose's GCC envelope is
        divided by its RMS, so every pose votes with equal weight.
    """
    recordings = np.asarray(recordings, dtype=np.float64)
    K, M, T = recordings.shape
    raw = np.zeros(grid_shape)
    sub = np.zeros(grid_shape)
    for k in range(K):
        h0 = empty_box_impulse(grid_shape, sources[k], mics[k], T)
        env_raw, env_sub = [], []
        for a in range(M):
            for b in range(a + 1, M):
                g, band = gcc_phat(recordings[k, a], recordings[k, b], n_fft, rel=rel)
                g0, _ = gcc_phat(h0[a], h0[b], n_fft, band=band)
                e = envelope(g)
                e0 = envelope(g0)
                env_raw.append(e / (np.sqrt(np.mean(e**2)) + 1e-30))
                d = np.maximum(e - e0, 0.0)
                env_sub.append(d / (np.sqrt(np.mean(d**2)) + 1e-30))
        raw += interferometric_image(grid_shape, sources[k], mics[k], np.array(env_raw), dt, n_fft)
        sub += interferometric_image(grid_shape, sources[k], mics[k], np.array(env_sub), dt, n_fft)
    return raw, sub


__all__ = [
    "band_mask",
    "gcc_phat",
    "empty_box_impulse",
    "interferometric_image",
    "passive_images",
]
