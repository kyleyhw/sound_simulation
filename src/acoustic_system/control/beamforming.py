"""Classical sound-zone controllers and broadband FIR design (plan 7.2).

All narrowband designs work per frequency on the transfer matrices
:math:`H_B` (bright-zone points x speakers) and :math:`H_D` (dark zone)
from :mod:`.transfer`, and return weights :math:`q(f)` of shape (F, S).

* **Delay-and-sum** (7.2.1): no room model. Each speaker is delayed so all
  direct-path arrivals line up at a focus point (or form a plane wave in a
  steering direction), with equal gains:
  :math:`q_s = e^{+j\\omega r_s / c}` (focus) or
  :math:`q_s = e^{-j\\omega\\, \\mathbf{x}_s\\cdot\\mathbf{u}/c}` (steer).
* **Pressure matching** (7.2.2): reproduce a target field :math:`p_T` in
  the bright zone and silence in the dark zone, regularised least squares

  .. math::
      \\min_q \\|H_B q - p_T\\|^2 + \\kappa \\|H_D q\\|^2 + \\lambda \\|q\\|^2
      \\;\\Rightarrow\\;
      q = (H_B^H H_B + \\kappa H_D^H H_D + \\lambda I)^{-1} H_B^H p_T.

* **Acoustic contrast control** (7.2.3, Choi & Kim 2002): maximise the
  contrast directly. With :math:`R_B = H_B^H H_B / M_B`,
  :math:`R_D = H_D^H H_D / M_D`,

  .. math::
      \\max_q \\frac{q^H R_B q}{q^H (R_D + \\delta I) q}
      \\;\\Rightarrow\\; R_B q = \\mu_{\\max} (R_D + \\delta I) q,

  the principal generalised eigenvector. Tikhonov :math:`\\delta` (relative
  to :math:`\\operatorname{tr} R_D / S`) bounds the array effort and the
  sensitivity to model error.
* **Time-reversal focusing** (7.2.4): play back the time-reversed impulse
  response from the focus to each speaker. By reciprocity this is the
  matched filter :math:`q_s = H_{f s}^*`, which maximises the pressure at
  the focus for a given drive energy (in a room it uses the reflections).
* **Broadband FIR design** (7.2.5) by frequency sampling. On the
  :math:`L`-point FFT grid :math:`f_k = k f_s / L`, the in-band bins take the
  weights (tapered at the band edges), a modelling delay :math:`\\tau`
  makes the filters causal, and a window tapers the time-aliased ends:

  .. math::
      c_s[n] = w[n]\\, \\mathcal{F}^{-1}_L\\bigl[g_k\\, q_s(f_k)\\,
               e^{-j 2\\pi f_k \\tau}\\bigr][n].

  Narrowband weights have an arbitrary phase at each frequency, so before
  the FIR design every method is normalised (:func:`normalise_to_reference`):
  the array reproduces, at a reference bright point, the level and phase
  that a single reference speaker would give. The FIR's realised response
  is then smooth in frequency, and the effort is relative to that speaker.

:func:`verify_fir` runs the filters through the time-domain engine and
compares the measured broadband contrast with the frequency-domain
prediction from the transfer functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.signal import fftconvolve
from scipy.signal.windows import tukey

from .metrics import array_effort_db, contrast_spectrum_db, energy_contrast_db, zone_energy
from .transfer import Point, Room, TransferSet, dtft, simulate_drives

# --------------------------------------------------------------------------- #
# Narrowband controllers
# --------------------------------------------------------------------------- #


def delay_and_sum(
    speakers: np.ndarray,
    freqs: np.ndarray,
    *,
    focus: "Point | None" = None,
    direction: "Point | None" = None,
    c: float = 343.0,
) -> np.ndarray:
    """Equal-gain delay-and-sum weights (F, S), focused on a point or steered.

    ``focus`` compensates the direct-path delays :math:`r_s / c` so the
    arrivals coincide there; ``direction`` (a vector, normalised here)
    delays for a plane wave travelling along it. Unit norm per frequency.
    """
    speakers = np.asarray(speakers, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    if (focus is None) == (direction is None):
        raise ValueError("give exactly one of focus or direction")
    if focus is not None:
        r = np.linalg.norm(speakers - np.asarray(focus, dtype=np.float64), axis=1)
        tau = -(r - r.min()) / c  # the farthest speaker fires first
    else:
        assert direction is not None
        u = np.asarray(direction, dtype=np.float64)
        u = u / np.linalg.norm(u)
        proj = speakers @ u
        tau = (proj - proj.min()) / c
    q = np.exp(-2j * np.pi * np.outer(freqs, tau))
    return q / np.sqrt(len(speakers))


def time_reversal(H_focus: np.ndarray) -> np.ndarray:
    """Matched-filter weights :math:`q = H_{f,:}^* / \\|H_{f,:}\\|` (F, S).

    ``H_focus`` is (F, S): the transfer functions to the focus point.
    """
    n = np.linalg.norm(H_focus, axis=-1, keepdims=True)
    return np.conj(H_focus) / np.maximum(n, 1e-300)


def pressure_matching(
    Hb: np.ndarray,
    Hd: np.ndarray,
    target: np.ndarray,
    *,
    reg: float = 1e-3,
    dark_weight: float = 1.0,
) -> np.ndarray:
    """Regularised least-squares weights (F, S) for bright-zone target ``target`` (F, M_B).

    ``reg`` is :math:`\\lambda` relative to
    :math:`\\operatorname{tr}(H_B^H H_B + \\kappa H_D^H H_D)/S`;
    ``dark_weight`` is :math:`\\kappa`.
    """
    F, _, S = Hb.shape
    q = np.empty((F, S), dtype=np.complex128)
    eye = np.eye(S)
    for k in range(F):
        A = Hb[k].conj().T @ Hb[k] + dark_weight * (Hd[k].conj().T @ Hd[k])
        lam = reg * np.real(np.trace(A)) / S
        q[k] = np.linalg.solve(A + lam * eye, Hb[k].conj().T @ target[k])
    return q


def acoustic_contrast_control(Hb: np.ndarray, Hd: np.ndarray, *, reg: float = 1e-3) -> np.ndarray:
    """ACC weights (F, S): principal generalised eigenvector, unit norm.

    ``reg`` is :math:`\\delta` relative to :math:`\\operatorname{tr} R_D / S`.
    """
    F, Mb, S = Hb.shape
    Md = Hd.shape[1]
    q = np.empty((F, S), dtype=np.complex128)
    eye = np.eye(S)
    for k in range(F):
        Rb = Hb[k].conj().T @ Hb[k] / Mb
        Rd = Hd[k].conj().T @ Hd[k] / Md
        delta = reg * np.real(np.trace(Rd)) / S
        _, vec = eigh(Rb, Rd + delta * eye, subset_by_index=[S - 1, S - 1])
        v = vec[:, 0]
        q[k] = v / np.linalg.norm(v)
    return q


def normalise_to_reference(
    q: np.ndarray, Hb: np.ndarray, ref_speaker: int = 0, ref_point: int = 0
) -> np.ndarray:
    """Scale and rotate ``q`` so the array matches a single reference speaker.

    After normalisation the mean squared bright-zone pressure equals that of
    speaker ``ref_speaker`` driven alone at unit gain, and the phase at
    bright point ``ref_point`` equals that speaker's. Contrast is unchanged;
    the FIR design gets a smooth, delay-like phase, and
    :func:`metrics.array_effort_db` becomes :math:`10\\log_{10}\\|q\\|^2`.
    """
    eb = zone_energy(Hb, q)
    e_ref = np.mean(np.abs(Hb[:, :, ref_speaker]) ** 2, axis=-1)
    scale = np.sqrt(e_ref / np.maximum(eb, 1e-300))
    p_arr = np.einsum("fs,fs->f", Hb[:, ref_point, :], q)
    p_ref = Hb[:, ref_point, ref_speaker]
    rot = np.exp(1j * (np.angle(p_ref) - np.angle(p_arr)))
    return q * (scale * rot)[:, None]


# --------------------------------------------------------------------------- #
# Broadband FIR design (frequency sampling)
# --------------------------------------------------------------------------- #


def design_grid(n_taps: int, dt: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """In-band FFT bins and their frequencies for an ``n_taps`` FIR at step ``dt``."""
    f = np.fft.rfftfreq(int(n_taps), dt)
    bins = np.where((f >= band[0]) & (f <= band[1]))[0]
    return bins, f[bins]


def fir_from_weights(
    q: np.ndarray,
    bins: np.ndarray,
    n_taps: int,
    dt: float,
    *,
    delay: float | None = None,
    window: str = "hann",
    taper_bins: int = 4,
    n_fft: int | None = None,
) -> np.ndarray:
    """FIR filters (..., L) from weights ``q`` (F, ...) on the in-band ``bins``.

    ``bins`` index the ``n_fft``-point grid (default ``n_fft = L``, plain
    frequency sampling). With ``n_fft > L`` the design is oversampled: the
    ``n_fft``-point impulse response is truncated to its first L samples,
    so a long room-inverse tail is cut instead of time-aliased onto the
    filter's start. ``delay`` is the modelling delay in samples (default
    ``L/2``); ``taper_bins`` raised-cosine bins at each band edge avoid a
    brick-wall band edge (long ringing); ``window`` is ``"hann"`` (whole
    filter, as in ``web/src/control/ctc.ts``), ``"tukey"`` (flat, 10 %
    cosine ends; keeps the filters' causal tails) or ``"none"``.
    """
    L = int(n_taps)
    N = L if n_fft is None else int(n_fft)
    if N < L:
        raise ValueError("n_fft must be >= n_taps")
    tau = L / 2 if delay is None else float(delay)
    F = len(bins)
    g = np.ones(F)
    t = min(int(taper_bins), F // 2)
    if t > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(t) + 1) / (t + 1))
        g[:t] = ramp
        g[F - t :] = ramp[::-1]
    f = bins / (N * dt)
    ph = np.exp(-2j * np.pi * f * tau * dt)
    extra = q.shape[1:]
    spec = np.zeros((N // 2 + 1,) + extra, dtype=np.complex128)
    shape = (F,) + (1,) * len(extra)
    spec[bins] = q * (g * ph).reshape(shape)
    taps = np.fft.irfft(spec, n=N, axis=0)[:L]
    if window == "hann":
        taps = taps * np.hanning(L).reshape((L,) + (1,) * len(extra))
    elif window == "tukey":
        taps = taps * tukey(L, 0.2).reshape((L,) + (1,) * len(extra))
    elif window != "none":
        raise ValueError(f"window must be 'hann', 'tukey' or 'none', got {window!r}")
    return np.moveaxis(taps, 0, -1)


def fir_response(taps: np.ndarray, freqs: np.ndarray, dt: float) -> np.ndarray:
    """Frequency response (F, ...) of filters (..., L) at ``freqs`` [Hz]."""
    R = dtft(taps, freqs, dt)
    return np.moveaxis(R, -1, 0)


def apply_fir(taps: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Filter the programme ``x`` (T,) through ``taps`` (..., L): (..., T + L - 1)."""
    x = np.asarray(x, dtype=np.float64)
    return fftconvolve(taps, x.reshape((1,) * (taps.ndim - 1) + (-1,)), axes=-1)


def time_reversal_fir(ir_focus: np.ndarray, n_taps: int) -> np.ndarray:
    """Broadband time-reversal filters (S, L): the first L samples of each IR, reversed.

    ``ir_focus`` is (S, N), the impulse responses from each speaker to the
    focus (``TransferSet.impulse_responses()[:, m]``). Unit total energy.
    """
    L = int(n_taps)
    h = np.zeros((ir_focus.shape[0], L))
    n = min(L, ir_focus.shape[1])
    h[:, :n] = ir_focus[:, :n]
    taps = h[:, ::-1]
    return taps / np.sqrt(np.sum(taps**2))


@dataclass
class BroadbandDesign:
    """A broadband sound-zone design and its frequency-domain predictions."""

    method: str
    taps: np.ndarray  # (S, L)
    freqs: np.ndarray  # design bins [Hz]
    q: np.ndarray  # (F, S) normalised narrowband weights at the bins
    ideal_db: np.ndarray  # (F,) contrast of q itself
    fir_db: np.ndarray  # (F,) contrast realised by the FIR at the same bins
    effort_db: np.ndarray  # (F,) array effort of q re the reference speaker


def design_broadband(
    ts: TransferSet,
    bright: np.ndarray,
    dark: np.ndarray,
    *,
    method: str = "acc",
    n_taps: int = 2048,
    band: tuple[float, float] = (300.0, 1500.0),
    reg: float = 1e-3,
    ref_speaker: int | None = None,
    ref_point: int = 0,
    focus: "Point | None" = None,
    delay: float | None = None,
    window: str = "tukey",
    oversample: int = 2,
) -> BroadbandDesign:
    """Design broadband FIR filters for one of the controllers.

    ``bright`` and ``dark`` index ``ts.points``; ``ref_point`` indexes
    ``bright`` (default: the first bright point, e.g. the zone centre).
    Methods: ``"acc"``, ``"pm"`` (target: the reference speaker's own field
    in the bright zone, silence in the dark zone), ``"das"`` (focused on
    ``focus`` or the reference bright point) and ``"tr"``.

    Defaults (L = 2048 taps = 75 ms at 27.4 kHz, modelling delay L/4, Tukey
    window, 2x oversampled design grid) were chosen on the absorbing and
    reverberant test rooms: the room inverse is mostly causal with a long
    reverberant tail, so a short pre-delay and a flat window keep more of
    it than the L/2 + Hann convention used for crosstalk cancellation.
    """
    S = len(ts.speakers)
    ref_speaker = S // 2 if ref_speaker is None else int(ref_speaker)
    n_fft = int(n_taps) * int(oversample)
    delay = n_taps / 4 if delay is None else delay
    bins, f = design_grid(n_fft, ts.dt, band)
    H = ts.at(f)
    Hb, Hd = H[:, bright], H[:, dark]
    if method == "acc":
        q = acoustic_contrast_control(Hb, Hd, reg=reg)
    elif method == "pm":
        q = pressure_matching(Hb, Hd, Hb[:, :, ref_speaker], reg=reg)
    elif method == "das":
        fp = ts.points[bright[ref_point]] if focus is None else focus
        c = ts.room.c if ts.room is not None else 343.0
        q = delay_and_sum(ts.speakers, f, focus=fp, c=c)
    elif method == "tr":
        q = time_reversal(Hb[:, ref_point, :])
    else:
        raise ValueError(f"unknown method {method!r}")
    q = normalise_to_reference(q, Hb, ref_speaker, ref_point)
    taps = fir_from_weights(q, bins, n_taps, ts.dt, delay=delay, window=window, n_fft=n_fft)
    R = fir_response(taps, f, ts.dt)
    return BroadbandDesign(
        method=method,
        taps=taps,
        freqs=f,
        q=q,
        ideal_db=contrast_spectrum_db(Hb, Hd, q),
        fir_db=contrast_spectrum_db(Hb, Hd, R),
        effort_db=array_effort_db(q, Hb, ref_speaker),
    )


@dataclass
class FirCheck:
    """Time-domain verification of a broadband design."""

    predicted_db: float  # broadband contrast predicted from H and the FIR response
    measured_db: float  # broadband contrast from the engine run
    bright_rec: np.ndarray  # (M_B, T)
    dark_rec: np.ndarray  # (M_D, T)
    energy: "np.ndarray | None"  # loudness map (sum p^2) over the run


def predicted_band_contrast(
    ts: TransferSet,
    taps: np.ndarray,
    programme: np.ndarray,
    bright: np.ndarray,
    dark: np.ndarray,
    band: tuple[float, float],
) -> float:
    """Contrast predicted in the frequency domain for ``programme`` through ``taps``.

    :math:`\\sum_f |X|^2 E_B(f) / \\sum_f |X|^2 E_D(f)` on the transfer
    set's own FFT grid, with :math:`E` from :math:`H(f)\\,C(f)`.
    """
    f_all = np.fft.rfftfreq(ts.steps, ts.dt)
    f = f_all[(f_all >= band[0]) & (f_all <= band[1])]
    H = ts.at(f)
    R = fir_response(taps, f, ts.dt)  # (F, S)
    X = dtft(programme, f, ts.dt)
    w = np.abs(X) ** 2
    eb = zone_energy(H[:, bright], R)
    ed = zone_energy(H[:, dark], R)
    return float(10 * np.log10(np.sum(w * eb) / np.sum(w * ed)))


def verify_fir(
    room: Room,
    ts: TransferSet,
    taps: np.ndarray,
    programme: np.ndarray,
    bright: np.ndarray,
    dark: np.ndarray,
    band: tuple[float, float],
    *,
    tail: float = 0.15,
    energy_map: bool = False,
) -> FirCheck:
    """Play ``programme`` through ``taps`` in the engine and score the zones.

    The run lasts the drive length plus ``tail`` seconds of decay; the
    measured contrast is the ratio of in-band recorded energies.
    """
    drives = apply_fir(taps, programme)
    steps = drives.shape[1] + int(np.ceil(tail / ts.dt))
    probes = np.concatenate([ts.points[bright], ts.points[dark]])
    res = simulate_drives(
        room, ts.speakers, drives, probes, steps, energy_window=(0, steps) if energy_map else None
    )
    yb, yd = res.rec[: len(bright)], res.rec[len(bright) :]
    return FirCheck(
        predicted_db=predicted_band_contrast(ts, taps, programme, bright, dark, band),
        measured_db=energy_contrast_db(yb, yd, ts.dt, band),
        bright_rec=yb,
        dark_rec=yd,
        energy=res.energy,
    )


__all__ = [
    "BroadbandDesign",
    "FirCheck",
    "acoustic_contrast_control",
    "apply_fir",
    "delay_and_sum",
    "design_broadband",
    "design_grid",
    "fir_from_weights",
    "fir_response",
    "normalise_to_reference",
    "predicted_band_contrast",
    "pressure_matching",
    "time_reversal",
    "time_reversal_fir",
    "verify_fir",
]
