"""Impulse-response recovery from chirp recordings (plan Task 6.2.1).

The active-sensing archives store, per pose, the pressure :math:`y_m[n]`
at microphone :math:`m` and the source audio :math:`u(t)`. The engine adds
the driver value :math:`s_k = A\\,u(k\\Delta t)` to :math:`p^{k+1}` at the
source cell (soft source) and the recorder stores :math:`p^{n+1}` after
step :math:`n`. The engine is linear and time invariant, so

.. math::
    y_m[n] = \\sum_{k=0}^{n} h_m[n-k]\\, s_k,
    \\qquad 0 \\le n < T,

where :math:`h_m[j]` is the discrete impulse response from the source cell
to mic :math:`m` at lag :math:`j` (a unit injection at step :math:`k`
appears at the mic :math:`j` steps later, i.e. after a physical delay
:math:`j\\,\\Delta t`). Recovering :math:`h_m` from :math:`(y_m, s)` is a
deconvolution. Two estimators are provided:

* :func:`wiener_deconvolve`: regularised spectral division
  :math:`\\hat H = Y S^* / (|S|^2 + \\lambda \\max |S|^2)` on a zero-padded
  FFT grid. It treats the unrecorded tail :math:`y[n \\ge T]` as zero,
  which is exact only when the recording outlasts the chirp plus the
  decay.
* :func:`tikhonov_deconvolve`: the same Wiener (white-prior) estimator
  solved on the *truncated* convolution, :math:`\\hat h = (S^\\top S +
  \\lambda I)^{-1} S^\\top y` with :math:`S` the :math:`T \\times L` lower
  triangular Toeplitz matrix of :math:`s`. The v2 chirp plays for the
  whole 400-step window, so a late echo of the high-frequency end of the
  sweep is never recorded; this estimator uses exactly the samples that
  exist. The operator is the same for every recording that shares a
  source, so it is built once and applied as one matrix product.

Direct-path removal
-------------------
The direct sound dominates the recording. Two ways to remove it:

* :func:`free_field_green_2d` evaluates the analytic 2D Green's function
  of the engine's soft source at the known source-mic distance
  :math:`r`. With :math:`p_{tt} = c^2 \\nabla^2 p + q` and
  :math:`q = (\\Delta x^2/\\Delta t^2)\\, s(t)\\, \\delta(\\mathbf{x})`,
  the step response of the 2D wave operator is
  :math:`F(t) = \\operatorname{arccosh}(ct/r) / (2\\pi c^2)` for
  :math:`ct > r`, so the discrete impulse response is
  :math:`h[j] = (\\Delta x^2/\\Delta t^2)\\,[F(j\\Delta t) - F((j-1)\\Delta t)]`
  (``scripts/verify_physics.py`` checks the same relation, correlation
  0.994). It ignores grid dispersion, which is large near the top of the
  v2 band (:math:`\\lambda \\approx 2.2` cells).
* :func:`empty_room_response` runs the engine itself in the obstacle-free
  box. This is the exact discrete incident field, direct path *and* the
  known outer-wall reverberation, so :math:`y - y_\\text{empty}` is the
  field scattered by the interior obstacles alone. This is the
  "equivalent" used by the imaging methods (the outer box is identical in
  every room, so it is known background, not target).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..simulation.setup import Driver
from ..simulation.simulate import Simulate
from ..simulation.waveforms import Waveform


@dataclass
class SampledWaveform(Waveform):
    """A waveform given by its values at the engine's injection times.

    ``values[n]`` is returned for :math:`t \\in [n\\Delta t - \\Delta t/2,
    n\\Delta t + \\Delta t/2)`, so ``Simulate`` injects exactly
    ``values[n]`` at step ``n`` (``time`` accumulates in exact multiples of
    ``dt`` for the dyadic steps used here, and rounding makes it robust
    otherwise). Zero outside the array.
    """

    values: NDArray[np.float64] | None = None
    dt: float = 0.5

    def __call__(self, t: float) -> float:
        v = self.values
        if v is None:
            return 0.0
        n = int(round(t / self.dt))
        if 0 <= n < v.shape[0]:
            return float(v[n])
        return 0.0


def source_drive(
    source: NDArray, n_steps: int, dt: float, fs: float, amplitude: float
) -> NDArray[np.float64]:
    """Per-step drive :math:`s_n = A\\,u(n\\Delta t)` of an archived source.

    Reproduces ``AudioFileWaveform``: linear interpolation of the stored
    audio (``fs`` samples per unit time) at :math:`t_n = n\\Delta t`, zero
    past the end. For the v2 archives :math:`f_s\\Delta t = 100` is an
    integer, so this is an exact decimation.
    """
    u = np.asarray(source, dtype=np.float64)
    idx_f = np.arange(n_steps) * dt * fs
    i0 = np.floor(idx_f).astype(np.int64)
    frac = idx_f - i0
    a = np.where(i0 < u.size, u[np.minimum(i0, u.size - 1)], 0.0)
    b = np.where(i0 + 1 < u.size, u[np.minimum(i0 + 1, u.size - 1)], 0.0)
    return amplitude * ((1.0 - frac) * a + frac * b)


def convolution_matrix(s: NDArray, n_out: int, n_lags: int) -> NDArray[np.float64]:
    """Lower-triangular Toeplitz :math:`S_{nj} = s_{n-j}` (``n_out x n_lags``)."""
    s = np.asarray(s, dtype=np.float64)
    n = np.arange(n_out)[:, None]
    j = np.arange(n_lags)[None, :]
    k = n - j
    valid = (k >= 0) & (k < s.size)
    return np.where(valid, s[np.clip(k, 0, s.size - 1)], 0.0)


class TikhonovDeconvolver:
    """Regularised least-squares deconvolution for a fixed source.

    Solves :math:`\\min_h \\lVert y - S h \\rVert^2 + \\lambda' \\lVert h
    \\rVert^2` with :math:`\\lambda' = \\lambda \\cdot \\operatorname{tr}
    (S^\\top S)/L`, i.e. :math:`\\lambda` is relative to the mean diagonal.
    This is the Wiener filter for white :math:`h` with noise-to-signal
    ratio :math:`\\lambda`. The resolvent :math:`R = (S^\\top S +
    \\lambda' I)^{-1} S^\\top` is precomputed, so a batch of recordings
    costs one matrix product.
    """

    def __init__(self, s: NDArray, n_out: int, n_lags: int | None = None, lam: float = 1e-3):
        n_lags = n_out if n_lags is None else int(n_lags)
        S = convolution_matrix(s, n_out, n_lags)
        gram = S.T @ S
        reg = lam * np.trace(gram) / n_lags
        self.S = S
        self.R = np.linalg.solve(gram + reg * np.eye(n_lags), S.T)

    def __call__(self, y: NDArray) -> NDArray[np.float64]:
        """Deconvolve along the last axis (any leading batch shape)."""
        y = np.asarray(y, dtype=np.float64)
        return y @ self.R.T


def tikhonov_deconvolve(y: NDArray, s: NDArray, lam: float = 1e-3) -> NDArray[np.float64]:
    """One-shot :class:`TikhonovDeconvolver` over the last axis of ``y``."""
    y = np.asarray(y, dtype=np.float64)
    return TikhonovDeconvolver(s, y.shape[-1], lam=lam)(y)


def wiener_deconvolve(
    y: NDArray, s: NDArray, lam: float = 1e-3, n_fft: int | None = None
) -> NDArray[np.float64]:
    """Regularised spectral division along the last axis.

    .. math::
        \\hat h = \\mathcal{F}^{-1}\\Bigl[\\frac{Y(f)\\,S^*(f)}
        {|S(f)|^2 + \\lambda \\max_f |S(f)|^2}\\Bigr],

    with both signals zero-padded to ``n_fft`` (default: the next power of
    two :math:`\\ge 2T`, so the circular convolution is linear). Returns the
    first :math:`T` lags. Frequencies where the chirp has little energy are
    suppressed by the :math:`\\lambda` floor instead of being amplified.
    """
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = y.shape[-1]
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(2 * T, s.size + T))))
    Y = np.fft.rfft(y, n_fft, axis=-1)
    S = np.fft.rfft(s, n_fft)
    p = np.abs(S) ** 2
    H = Y * np.conj(S) / (p + lam * p.max())
    return np.fft.irfft(H, n_fft, axis=-1)[..., :T]


def matched_filter(y: NDArray, s: NDArray) -> NDArray[np.float64]:
    """Pulse compression: cross-correlation :math:`\\sum_n y[n+j]\\, s[n]`.

    The matched filter maximises the output SNR for a known waveform in
    white noise. Its output is :math:`h * R_{ss}`, with the chirp
    autocorrelation :math:`R_{ss}` as the (band-limited) point-spread
    function; returned for lags :math:`0 \\le j < T`.
    """
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    T = y.shape[-1]
    n_fft = 1 << int(np.ceil(np.log2(T + s.size)))
    Y = np.fft.rfft(y, n_fft, axis=-1)
    S = np.fft.rfft(s, n_fft)
    return np.fft.irfft(Y * np.conj(S), n_fft, axis=-1)[..., :T]


def envelope(x: NDArray, axis: int = -1) -> NDArray[np.float64]:
    """Hilbert envelope :math:`|x + j\\mathcal{H}x|` along ``axis``."""
    from scipy.signal import hilbert

    return np.abs(hilbert(np.asarray(x, dtype=np.float64), axis=axis))


def free_field_green_2d(
    r: float,
    n_lags: int,
    dt: float,
    c: float = 1.0,
    dx: float = 1.0,
) -> NDArray[np.float64]:
    """Discrete impulse response of the 2D free field at distance ``r``.

    .. math::
        h[j] = \\frac{\\Delta x^2}{\\Delta t^2}\\bigl[F(j\\Delta t) -
        F((j-1)\\Delta t)\\bigr], \\qquad
        F(t) = \\frac{\\operatorname{arccosh}(\\max(ct/r, 1))}{2\\pi c^2},

    for lags :math:`j = 0 \\dots n_\\text{lags}-1` (``r`` in the same length
    unit as ``dx``). The :math:`1/\\sqrt{t^2 - r^2/c^2}` singularity of the
    2D Green's function is integrated exactly per step, and the slowly
    decaying wake behind the wavefront, the signature of 2D propagation,
    is kept. ``r`` must be positive; a mic on the source cell has no
    continuum analogue.
    """
    if r <= 0:
        raise ValueError("r must be positive")
    t = np.arange(-1, n_lags) * dt
    F = np.arccosh(np.maximum(c * t / r, 1.0)) / (2.0 * np.pi * c * c)
    return (dx * dx / (dt * dt)) * np.diff(F)


def remove_direct_path_free_field(
    y: NDArray,
    s: NDArray,
    r: float,
    dt: float,
    c: float = 1.0,
    dx: float = 1.0,
    fit_window: int | None = None,
) -> NDArray[np.float64]:
    """Subtract the modelled direct sound :math:`a\\,(g_r * s)` from a recording.

    With ``fit_window`` the gain :math:`a = \\langle y, d\\rangle/\\langle d,
    d\\rangle` is fitted on the first ``fit_window`` samples (choose a window
    that ends before the first possible echo), which absorbs the few-percent
    amplitude error of the continuum model; otherwise :math:`a = 1`.
    """
    y = np.asarray(y, dtype=np.float64)
    T = y.shape[-1]
    g = free_field_green_2d(r, T, dt, c, dx)
    d = np.convolve(np.asarray(s, dtype=np.float64), g)[:T]
    a = 1.0
    if fit_window is not None:
        w = slice(0, int(fit_window))
        den = float(d[w] @ d[w])
        a = float(y[w] @ d[w]) / den if den > 0 else 1.0
    return y - a * d


def empty_room_response(
    grid_shape: tuple[int, int],
    source_pos: ArrayLike,
    drive: NDArray,
    mic_pos: ArrayLike,
    courant: float = 0.5,
    return_field: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float32] | None]:
    """Record the obstacle-free box with the engine (direct + outer walls).

    Returns ``(rec, field)``: ``rec`` has shape ``(n_mics, T)`` and equals
    what the archive would contain for an empty room; ``field`` (only with
    ``return_field``) is the full incident wavefield ``(T, *grid)`` with
    ``field[n] = p^{n+1}``, which the time-reversal imager correlates
    against.
    """
    drive = np.asarray(drive, dtype=np.float64)
    T = drive.shape[0]
    sim = Simulate(grid_shape=grid_shape, courant=courant)
    sim.set_drivers(
        [
            Driver(
                position=tuple(int(c) for c in np.asarray(source_pos).ravel()),
                waveform=SampledWaveform(drive, sim.timestep),
            )
        ]
    )
    mics = [tuple(int(c) for c in m) for m in np.asarray(mic_pos).reshape(-1, len(grid_shape))]
    rec = np.zeros((len(mics), T), dtype=np.float64)
    field = np.empty((T,) + tuple(grid_shape), dtype=np.float32) if return_field else None
    for n in range(T):
        sim.step()
        p = sim.p
        for i, m in enumerate(mics):
            rec[i, n] = p[m]
        if field is not None:
            field[n] = p
    return rec, field


__all__ = [
    "SampledWaveform",
    "source_drive",
    "convolution_matrix",
    "TikhonovDeconvolver",
    "tikhonov_deconvolve",
    "wiener_deconvolve",
    "matched_filter",
    "envelope",
    "free_field_green_2d",
    "remove_direct_path_free_field",
    "empty_room_response",
]
