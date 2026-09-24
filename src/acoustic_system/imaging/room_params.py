"""Room acoustic parameters from recordings (plan Task 6.5.1).

Given a chirp recording and the known excitation, recover the impulse
response by spectral division (``ir.wiener_deconvolve``; the recording
must outlast the chirp plus the decay so the division is exact) and read
off three parameters.

Reverberation time (Schroeder)
------------------------------
The backward-integrated energy decay curve

.. math::
    \\mathrm{EDC}(t) = 10\\log_{10}\\frac{\\int_t^\\infty h^2(t')\\,dt'}
                                       {\\int_0^\\infty h^2(t')\\,dt'}

is the ensemble average of the decay of interrupted noise (Schroeder,
1965). T60 is extrapolated from a least-squares line fitted between
:math:`-5` and :math:`-25` dB (T20) or :math:`-35` dB (T30).

Direct-to-reverberant ratio
---------------------------
.. math::
    \\mathrm{DRR} = 10\\log_{10}\\frac{\\sum_{|n - n_d| \\le w} h^2[n]}
                                     {\\sum_{n > n_d + w} h^2[n]},

with :math:`n_d` the direct-sound peak and :math:`w` a short window.

Mean absorption (Eyring inversion)
----------------------------------
In a diffuse 2D room of area :math:`S` and perimeter :math:`L`, the
energy decays as :math:`\\exp(-c L a t/(\\pi S))` (the 2D mean free path
is :math:`\\pi S/L`), so

.. math::
    T_{60} = \\frac{6 \\ln 10\\, \\pi S}{c L a}, \\qquad
    a_\\text{Sabine} = \\bar\\alpha, \\quad a_\\text{Eyring} = -\\ln(1-\\bar\\alpha),

and the inversions are :math:`\\bar\\alpha_S = 6\\ln 10\\,\\pi S/(c L T_{60})`
and :math:`\\bar\\alpha_E = 1 - \\exp(-6\\ln 10\\,\\pi S/(c L T_{60}))`
(``scripts/verify_physics.py`` section 6 uses the same relations; the 3D
shoebox form is ``utils/room_ir.alpha_from_t60``). For an impedance wall
with normalised admittance :math:`\\beta` the plane-wave reflection is
:math:`R(\\theta) = (\\cos\\theta - \\beta)/(\\cos\\theta + \\beta)` and the
2D random-incidence absorption is
:math:`\\alpha_d = \\tfrac12\\int_{-\\pi/2}^{\\pi/2} (1 - R^2)\\cos\\theta\\,d\\theta`,
the reference the estimate is validated against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .ir import wiener_deconvolve

LN10 = float(np.log(10.0))


def schroeder_edc(h: NDArray) -> NDArray[np.float64]:
    """Energy decay curve in dB (0 dB at the start)."""
    e = np.cumsum((np.asarray(h, dtype=np.float64) ** 2)[::-1])[::-1]
    if e.size == 0 or e[0] <= 0:
        raise ValueError("silent impulse response")
    return 10.0 * np.log10(e / e[0] + 1e-300)


def t60_schroeder(
    h: NDArray, dt: float, fit_range: tuple[float, float] = (-5.0, -25.0), start: int = 0
) -> float:
    """T60 from a line fitted to the EDC between ``fit_range`` dB.

    ``start`` drops the samples before the direct sound. Returns ``nan``
    when the EDC never reaches the lower fit bound.
    """
    edc = schroeder_edc(np.asarray(h)[start:])
    hi, lo = fit_range
    above = np.nonzero(edc <= hi)[0]
    below = np.nonzero(edc <= lo)[0]
    if not above.size or not below.size or below[0] - above[0] < 4:
        return float("nan")
    i0, i1 = int(above[0]), int(below[0])
    t = np.arange(i0, i1 + 1) * dt
    slope = np.polyfit(t, edc[i0 : i1 + 1], 1)[0]
    return float(-60.0 / slope) if slope < 0 else float("nan")


def direct_to_reverberant(h: NDArray, window: int = 10, direct_index: int | None = None) -> float:
    """DRR in dB around the direct peak (``direct_index`` or ``argmax |h|``)."""
    x = np.asarray(h, dtype=np.float64)
    nd = int(np.argmax(np.abs(x))) if direct_index is None else int(direct_index)
    e_dir = float((x[max(nd - window, 0) : nd + window + 1] ** 2).sum())
    e_rev = float((x[nd + window + 1 :] ** 2).sum())
    return float(10.0 * np.log10(e_dir / e_rev)) if e_rev > 0 else float("inf")


def alpha_sabine_2d(t60: float, area: float, perimeter: float, c: float = 1.0) -> float:
    """Mean absorption from T60 by the 2D Sabine relation."""
    return float(6.0 * LN10 * np.pi * area / (c * perimeter * t60))


def alpha_eyring_2d(t60: float, area: float, perimeter: float, c: float = 1.0) -> float:
    """Mean absorption from T60 by the 2D Eyring relation."""
    return float(1.0 - np.exp(-6.0 * LN10 * np.pi * area / (c * perimeter * t60)))


def t60_sabine_2d(alpha: float, area: float, perimeter: float, c: float = 1.0) -> float:
    return float(6.0 * LN10 * np.pi * area / (c * perimeter * alpha))


def t60_eyring_2d(alpha: float, area: float, perimeter: float, c: float = 1.0) -> float:
    return float(6.0 * LN10 * np.pi * area / (c * perimeter * -np.log(1.0 - alpha)))


def alpha_diffuse_2d(beta: float) -> float:
    """Random-incidence absorption of a locally reacting wall in 2D."""
    th = np.linspace(-np.pi / 2, np.pi / 2, 4001)
    r = (np.cos(th) - beta) / (np.cos(th) + beta)
    return float(0.5 * np.trapezoid((1 - r**2) * np.cos(th), th))


@dataclass
class RoomParams:
    t60: float
    t30: float
    drr_db: float
    alpha_eyring: float
    alpha_sabine: float


def bandpass(
    h: NDArray, dt: float, band: tuple[float, float], order: int = 4
) -> NDArray[np.float64]:
    """Zero-phase Butterworth band-pass (``band`` in cycles per unit time)."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(order, band, btype="band", fs=1.0 / dt, output="sos")
    return sosfiltfilt(sos, np.asarray(h, dtype=np.float64))


def estimate_room_params(
    recording: NDArray,
    drive: NDArray,
    dt: float,
    area: float,
    perimeter: float,
    c: float = 1.0,
    band: tuple[float, float] | None = None,
    lam: float = 1e-4,
    drr_window: int = 10,
) -> RoomParams:
    """T60 (T20 fit), T30, DRR and mean absorption from one chirp recording.

    ``recording`` and ``drive`` are per-step arrays; the recording must
    contain the whole decay (it is zero-padded and divided spectrally).
    ``band`` restricts the impulse response to the excited band before the
    Schroeder integral. Outside it the regularised division returns
    near-DC residue that decays far more slowly than the room and bends the
    EDC (in the validation below it doubles the broadband T60), so a band
    inside the chirp's sweep should always be given.
    """
    y = np.asarray(recording, dtype=np.float64)
    h = wiener_deconvolve(y, drive, lam=lam, n_fft=1 << int(np.ceil(np.log2(2 * y.size))))
    if band is not None:
        h = bandpass(h, dt, band)
    nd = int(np.argmax(np.abs(h)))
    t60 = t60_schroeder(h, dt, (-5.0, -25.0), start=nd)
    t30 = t60_schroeder(h, dt, (-5.0, -35.0), start=nd)
    drr = direct_to_reverberant(h, drr_window, nd)
    return RoomParams(
        t60=t60,
        t30=t30,
        drr_db=drr,
        alpha_eyring=alpha_eyring_2d(t60, area, perimeter, c),
        alpha_sabine=alpha_sabine_2d(t60, area, perimeter, c),
    )


def dc_free_chirp(
    n_steps: int, dt: float, f_start: float, f_end: float, amplitude: float = 5.0
) -> NDArray[np.float64]:
    """A tapered linear chirp with zero mean and zero first moment.

    A soft source adds :math:`s_n` to :math:`p^{n+1}`, so the spatial sum of
    :math:`p` obeys :math:`\\Sigma^{n+1} = 2\\Sigma^n - \\Sigma^{n-1} + s_n` plus
    wall losses: any :math:`\\sum_n s_n \\ne 0` or :math:`\\sum_n n s_n \\ne 0`
    pumps a quasi-static pressure that a rigid or weakly absorbing room
    never sheds, and the EDC then flattens. Taking the second difference of
    a Tukey-tapered chirp removes both moments exactly.
    """
    from scipy.signal.windows import tukey

    t = np.arange(n_steps) * dt
    T = n_steps * dt
    k = (f_end - f_start) / T
    x = np.sin(2 * np.pi * (f_start * t + 0.5 * k * t * t)) * tukey(n_steps, 0.2)
    d2 = np.diff(x, 2, prepend=0.0, append=0.0)
    return amplitude * d2 / np.abs(d2).max()


def simulate_absorbing_room(
    shape: tuple[int, int],
    beta: float,
    source: tuple[int, int],
    mics: list[tuple[int, int]],
    drive: NDArray,
    n_steps: int,
    n_scatterers: int = 14,
    seed: int = 0,
) -> tuple[NDArray[np.float64], float]:
    """Record a 2D room with impedance walls of admittance ``beta``.

    Rigid 5x5 scatterers (material 2) make the field closer to diffuse, as in
    ``scripts/verify_physics.py``; cells around the devices are kept clear.
    Returns ``(recordings (n_mics, n_steps), dt)``.
    """
    from ..simulation.setup import Driver
    from ..simulation.simulate import Simulate
    from .ir import SampledWaveform

    nx, ny = shape
    rng = np.random.default_rng(seed)
    sim = Simulate(shape, boundary="absorb", boundary_beta=beta)
    m = np.zeros(shape, np.uint8)
    for _ in range(n_scatterers):
        ci = int(rng.integers(15, nx - 15))
        cj = int(rng.integers(15, ny - 15))
        m[ci - 2 : ci + 3, cj - 2 : cj + 3] = 2
    for q in list(mics) + [source]:
        m[q[0] - 3 : q[0] + 4, q[1] - 3 : q[1] + 4] = 0
    sim.set_material_map(m)
    sim.set_drivers(
        [Driver(tuple(source), SampledWaveform(np.asarray(drive, float), sim.timestep))]
    )
    rec = np.zeros((len(mics), n_steps))
    for n in range(n_steps):
        sim.step()
        for i, q in enumerate(mics):
            rec[i, n] = sim.p[q]
    return rec, float(sim.timestep)


__all__ = [
    "schroeder_edc",
    "t60_schroeder",
    "direct_to_reverberant",
    "alpha_sabine_2d",
    "alpha_eyring_2d",
    "t60_sabine_2d",
    "t60_eyring_2d",
    "alpha_diffuse_2d",
    "RoomParams",
    "bandpass",
    "estimate_room_params",
    "dc_free_chirp",
    "simulate_absorbing_room",
]
