"""Adaptive (FxLMS) noise cancellation at a quiet zone (plan 7.4).

A primary source plays noise :math:`x[n]`. A secondary speaker, driven by
an adaptive FIR filter :math:`w` of the reference, cancels it at an error
microphone. Feedforward with a perfect reference: the controller hears
:math:`x` itself, like a reference mic next to the noise source.

.. math::
    y[n] = \\sum_{k=0}^{K-1} w_k\\, x[n - kD], \\qquad
    e[n] = d[n] + (s * y)[n],

where :math:`d` is the primary noise at the error mic and :math:`s` the
secondary path (speaker to mic). The filter taps are spaced by a stride
of :math:`D` samples. The engine runs at 27.4 kHz, far above the noise
band, and adjacent samples are almost equal, so sparse taps span a useful
time window with few weights. Filtered-x NLMS (Widrow; Burgess 1981)
updates

.. math::
    w_k \\leftarrow w_k - \\mu\\, \\frac{e[n]\\, x'[n - kD]}
        {\\epsilon + \\sum_j x'[n - jD]^2},
    \\qquad x' = \\hat s * x,

with :math:`\\hat s` the secondary-path estimate. Here that is the
band-limited impulse response from the transfer-function measurement
(:meth:`.transfer.TransferSet.impulse_responses`). It converges if the
phase error of :math:`\\hat s` stays below 90 degrees in the band.

Everything runs sample by sample in the time-domain engine: at step
:math:`n` the controller output :math:`y[n]` is injected with the primary
sample, and the error :math:`e[n]` is read from the field, so the loop
has the true acoustic delay. A second, uncontrolled run (primary only)
gives :math:`d[n]` and the reference loudness map.

Quiet-zone size (7.4.2): after convergence the weights are frozen, and
both runs accumulate :math:`\\sum p^2` over a window at every cell. The
attenuation map is :math:`10\\log_{10}(E_{\\text{off}}/E_{\\text{on}})`,
and the quiet zone is the connected region around the error mic with at
least 10 dB attenuation. Its equivalent diameter
:math:`2\\sqrt{A/\\pi}` is compared with the classic diffuse-field
estimate of about :math:`\\lambda/10` (Elliott, Joseph, Bullmore &
Nelson 1988).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.signal import lfilter

from .transfer import Point, Room


@dataclass
class FxlmsResult:
    """Outcome of one FxLMS run."""

    error: np.ndarray  # e[n] at the error mic, control on
    disturbance: np.ndarray  # d[n] at the error mic, control off
    drive: np.ndarray  # y[n], the secondary speaker's drive
    weights: np.ndarray  # final w (K,)
    attenuation_db: float  # 10 log10(sum d^2 / sum e^2) over the measurement window
    learning_db: np.ndarray  # attenuation per block of ``block`` steps (dB)
    block: int
    att_map: "np.ndarray | None"  # (nx, ny) attenuation over the window (dB, nan in walls)
    dt: float
    adapt_steps: int


def fxlms(
    room: Room,
    primary: Point,
    secondary: Point,
    mic: Point,
    noise: np.ndarray,
    s_hat: np.ndarray,
    *,
    n_taps: int = 16,
    stride: int = 8,
    mu: float = 0.02,
    update_every: int | None = None,
    eps: float = 1e-8,
    mic_highpass: float = 30.0,
    adapt_steps: int | None = None,
    measure_steps: int = 2000,
    field: bool = True,
    block: int = 500,
) -> FxlmsResult:
    """Run filtered-x NLMS against the engine.

    Parameters
    ----------
    room
        The scene.
    primary, secondary, mic
        World positions of the noise source, the control speaker and the
        error microphone.
    noise
        Primary drive samples; also the controller's reference. Its length
        sets the total run (``adapt_steps + measure_steps`` at most).
    s_hat
        Secondary-path impulse-response estimate (engine drive convention).
    n_taps, stride
        Adaptive filter length K and tap spacing D (samples).
    mu, eps
        NLMS step size and regulariser.
    update_every
        Adapt once every this many samples (default: ``stride``, i.e. the
        controller updates at its own decimated rate fs/D). Per-sample
        updates at 27 kHz multiply the effective step size by D; near a
        room mode, where the plant's response to a weight change builds up
        over the reverberation time, that makes the loop unstable.
    mic_highpass
        Corner [Hz] of a first-order DC blocker on the error mic (and, for
        a consistent phase, on the filtered reference), like an AC-coupled
        microphone. A soft source injects net pressure whenever its drive
        has a DC component (any tone onset), and a closed 2D room turns
        that into a slowly decaying quasi-static wake. It is inaudible, but
        without the blocker it dominates the error signal at low
        frequencies. The scored mic signals and field maps are blocked the
        same way. 0 disables it.
    adapt_steps
        Steps with adaptation on (default: all but ``measure_steps``); the
        weights are frozen for the last ``measure_steps``.
    field
        Also return the attenuation map over the measurement window.
    """
    x = np.asarray(noise, dtype=np.float64)
    T = len(x)
    adapt = T - measure_steps if adapt_steps is None else int(adapt_steps)
    if adapt < 0 or adapt + measure_steps > T:
        raise ValueError("noise is shorter than adapt_steps + measure_steps")
    K, D = int(n_taps), int(stride)
    U = D if update_every is None else max(1, int(update_every))
    dt = room.timestep
    # DC blocker y[n] = a (y[n-1] + x[n] - x[n-1]).
    a = float(np.exp(-2 * np.pi * mic_highpass * dt)) if mic_highpass > 0 else 1.0
    xf = np.convolve(x, np.asarray(s_hat, dtype=np.float64))[:T]
    if mic_highpass > 0:
        xf = lfilter([a, -a], [1.0, -a], xf)
    pad = K * D
    xp = np.concatenate([np.zeros(pad), x])
    xfp = np.concatenate([np.zeros(pad), xf])
    lags = D * np.arange(K)

    cells = room.cells([primary, secondary, mic])
    (pi, pj), (si, sj), (mi, mj) = cells
    on, off = room.make_sim(), room.make_sim()
    w = np.zeros(K)
    e = np.zeros(T)
    d = np.zeros(T)
    y = np.zeros(T)
    raw_e = raw_d = 0.0
    # Field maps: blocked the same way, starting ~5 time constants before
    # the measurement window so the blocker has settled.
    warm = int(5.0 / (2 * np.pi * mic_highpass * dt)) if mic_highpass > 0 else 0
    f0 = max(0, adapt - warm)
    e_on = np.zeros(room.shape) if field else None
    e_off = np.zeros(room.shape) if field else None
    hp_on = np.zeros(room.shape) if field else None
    hp_off = np.zeros(room.shape) if field else None
    prev_on = np.zeros(room.shape, dtype=np.float32) if field else None
    prev_off = np.zeros(room.shape, dtype=np.float32) if field else None
    for n in range(T):
        xv = xp[n + pad - lags]
        yn = float(w @ xv)
        y[n] = yn
        on.step()
        off.step()
        on.p[pi, pj] += x[n]
        on.p[si, sj] += yn
        off.p[pi, pj] += x[n]
        ro, rf = float(on.p[mi, mj]), float(off.p[mi, mj])
        en = a * ((e[n - 1] if n else 0.0) + ro - raw_e) if a < 1.0 else ro
        dn = a * ((d[n - 1] if n else 0.0) + rf - raw_d) if a < 1.0 else rf
        raw_e, raw_d = ro, rf
        e[n] = en
        d[n] = dn
        if n < adapt and n % U == 0:
            xfv = xfp[n + pad - lags]
            w -= (mu * en / (eps + float(xfv @ xfv))) * xfv
        if hp_on is not None and hp_off is not None and n >= f0:
            assert prev_on is not None and prev_off is not None
            assert e_on is not None and e_off is not None
            if a < 1.0:
                hp_on = a * (hp_on + on.p - prev_on)
                hp_off = a * (hp_off + off.p - prev_off)
                prev_on[...] = on.p
                prev_off[...] = off.p
            else:
                hp_on, hp_off = on.p.astype(np.float64), off.p.astype(np.float64)
            if n >= adapt:
                e_on += hp_on**2
                e_off += hp_off**2
    win = slice(adapt, adapt + measure_steps)
    att = 10 * np.log10(np.sum(d[win] ** 2) / max(np.sum(e[win] ** 2), 1e-300))
    nb = T // block
    eb = np.square(e[: nb * block]).reshape(nb, block).sum(1)
    db = np.square(d[: nb * block]).reshape(nb, block).sum(1)
    learning = 10 * np.log10(np.maximum(db, 1e-300) / np.maximum(eb, 1e-300))
    att_map = None
    if e_on is not None and e_off is not None:
        ok = (e_off > 1e-30) & (e_on > 1e-30)
        att_map = np.full(room.shape, np.nan)
        att_map[ok] = 10 * np.log10(e_off[ok] / e_on[ok])
    return FxlmsResult(e, d, y, w, float(att), learning, block, att_map, dt, adapt)


@dataclass
class QuietZone:
    """The >= threshold attenuation region around the error mic."""

    mask: np.ndarray  # (nx, ny) bool
    area_m2: float
    diameter_m: float  # equivalent diameter 2 sqrt(A / pi)
    extent_m: tuple[float, float]  # bounding-box size along x and y


def quiet_zone(
    att_map: np.ndarray, room: Room, mic: Point, threshold_db: float = 10.0
) -> QuietZone:
    """Connected region of ``att_map >= threshold_db`` that contains the mic cell."""
    ij = room.cells([mic])[0]
    good = np.nan_to_num(att_map, nan=-np.inf) >= threshold_db
    lab, _ = ndimage.label(good)
    k = lab[ij[0], ij[1]]
    mask = lab == k if k > 0 else np.zeros_like(good)
    area = float(mask.sum()) * room.dx**2
    if mask.any():
        ii, jj = np.nonzero(mask)
        ext = ((ii.max() - ii.min() + 1) * room.dx, (jj.max() - jj.min() + 1) * room.dx)
    else:
        ext = (0.0, 0.0)
    return QuietZone(mask, area, 2.0 * np.sqrt(area / np.pi), ext)


def tone(f: float, dt: float, steps: int, ramp: float = 0.01) -> np.ndarray:
    """A sine at ``f`` Hz with a raised-cosine onset of ``ramp`` seconds."""
    t = np.arange(steps) * dt
    x = np.sin(2 * np.pi * f * t)
    r = min(steps, max(1, int(ramp / dt)))
    x[:r] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
    return x


def band_noise(
    band: tuple[float, float], dt: float, steps: int, seed: int = 0, ramp: float = 0.01
) -> np.ndarray:
    """Unit-RMS Gaussian noise band-limited to ``band`` (FFT mask), with an onset ramp."""
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.standard_normal(steps))
    f = np.fft.rfftfreq(steps, dt)
    X[(f < band[0]) | (f > band[1])] = 0.0
    x = np.fft.irfft(X, n=steps)
    x /= np.sqrt(np.mean(x**2))
    r = min(steps, max(1, int(ramp / dt)))
    x[:r] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
    return x


def stride_for(f_max: float, dt: float) -> int:
    """Tap spacing that samples ``f_max`` about four times per period."""
    return max(1, int(round(1.0 / (4.0 * f_max * dt))))


__all__ = [
    "FxlmsResult",
    "QuietZone",
    "band_noise",
    "fxlms",
    "quiet_zone",
    "stride_for",
    "tone",
]
