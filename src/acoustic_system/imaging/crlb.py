"""Cramér-Rao bounds for echo ranging and bearing (plan Task 6.4).

What accuracy can *any* unbiased estimator reach from a laptop's echoes?
The bounds below assume a known transmitted waveform :math:`s(t)` in
additive white Gaussian noise (the active-sensing setting), a single
specular reflector, and independent noise across mics and poses.

Time of arrival
---------------
For :math:`y(t) = a\\,s(t - \\tau) + n(t)` with noise spectral density
:math:`N_0/2`, the Fisher information on :math:`\\tau` gives

.. math::
    \\operatorname{var}(\\hat\\tau) \\ge \\frac{1}{8\\pi^2 \\beta^2\\,
        \\mathrm{SNR}}, \\qquad \\mathrm{SNR} = \\frac{2E}{N_0},

with :math:`E` the echo energy (so SNR is the matched-filter output
SNR, which already includes the chirp's time-bandwidth gain) and
:math:`\\beta^2` the mean-square frequency of :math:`|S(f)|^2`:

* **coherent** (carrier phase usable): :math:`\\beta^2 = \\int f^2 |S|^2 /
  \\int |S|^2`, which for a flat band :math:`[f_c - B/2, f_c + B/2]` is
  :math:`f_c^2 + B^2/12`;
* **envelope** (phase unknown or ambiguous, the robust case):
  :math:`\\beta^2 = B_\\text{rms}^2 = \\int (f - f_c)^2 |S|^2 / \\int |S|^2
  = B^2/12` for a flat band.

The envelope bound therefore scales as :math:`1/(B\\sqrt{\\mathrm{SNR}})`.
The coherent bound is far lower but only holds above the SNR threshold
where the estimator stops jumping between carrier cycles (ambiguity
region), so the chart reports both.

Range to a planar reflector
---------------------------
A co-located source and mic see the reflector at :math:`R = c\\tau/2`, so
:math:`\\sigma_R = \\tfrac{c}{2}\\sigma_\\tau / \\sqrt{N}` for :math:`N`
independent (mic, pose) observations. The two-reflector resolution is
the Rayleigh limit :math:`\\delta R = c/(2B)`.

Bearing from TDOA
-----------------
A far-field echo from bearing :math:`\\theta` (from broadside) reaches mic
:math:`m` at position :math:`x_m` on the array axis at
:math:`\\tau_m = \\tau_0 + x_m \\sin\\theta / c`. With the common delay
:math:`\\tau_0` unknown (range is a nuisance parameter), the Fisher
information on :math:`\\theta` is :math:`\\cos^2\\theta \\sum_m (x_m -
\\bar x)^2 / (c^2 \\sigma_\\tau^2)`, so

.. math::
    \\sigma_\\theta \\ge \\frac{c\\,\\sigma_\\tau}{\\cos\\theta\\,
        \\sqrt{\\sum_m (x_m - \\bar x)^2}}\\frac{1}{\\sqrt{K}},

which for a pair with spacing :math:`d` is :math:`\\sqrt2\\,c\\sigma_\\tau /
(d\\cos\\theta)` and for a uniform line array of :math:`M` mics is
:math:`\\sum (x_m - \\bar x)^2 = d^2 M (M^2 - 1)/12`. :math:`K` poses
viewing the same reflector add information (the bound falls as
:math:`1/\\sqrt K`). A spacing above :math:`\\lambda_\\min/2` makes the
coherent bearing ambiguous (grating lobes), which is why the envelope
bound is the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..simulation.units import LAPTOP_ROOM, SPEED_OF_SOUND_AIR


def db_to_linear(snr_db: float | NDArray) -> NDArray[np.float64] | float:
    return 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 10.0)


def band_beta(f_lo: float, f_hi: float, coherent: bool = False) -> float:
    """RMS frequency :math:`\\beta` of a flat band (Hz or grid units)."""
    b = f_hi - f_lo
    fc = 0.5 * (f_lo + f_hi)
    beta2 = b * b / 12.0 + (fc * fc if coherent else 0.0)
    return float(np.sqrt(beta2))


def signal_beta(signal: NDArray, fs: float, coherent: bool = False) -> float:
    """RMS frequency :math:`\\beta` of an arbitrary real signal sampled at ``fs``."""
    x = np.asarray(signal, dtype=np.float64)
    S2 = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, d=1.0 / fs)
    w = S2 / S2.sum()
    fc = float((w * f).sum())
    beta2 = float((w * (f - fc) ** 2).sum()) + (fc * fc if coherent else 0.0)
    return float(np.sqrt(beta2))


def toa_std(beta: float, snr: float | NDArray) -> NDArray[np.float64] | float:
    """:math:`\\sigma_\\tau \\ge 1/(2\\pi\\beta\\sqrt{\\mathrm{SNR}})` (SNR linear)."""
    return 1.0 / (2.0 * np.pi * beta * np.sqrt(np.asarray(snr, dtype=np.float64)))


def range_std(
    bandwidth: float | NDArray,
    snr: float | NDArray,
    c: float = SPEED_OF_SOUND_AIR,
    centre: float = 0.0,
    coherent: bool = False,
    n_obs: int = 1,
) -> NDArray[np.float64] | float:
    """CRLB on the range to one planar reflector (co-located source and mic).

    ``bandwidth`` and ``centre`` in Hz (or grid units, with ``c`` to match);
    ``snr`` linear matched-filter SNR; ``n_obs`` independent observations.
    """
    b = np.asarray(bandwidth, dtype=np.float64)
    beta = np.sqrt(b * b / 12.0 + (centre * centre if coherent else 0.0))
    return 0.5 * c * toa_std(beta, snr) / np.sqrt(n_obs)


def array_moment(n_mics: int, spacing: float) -> float:
    """:math:`\\sum_m (x_m - \\bar x)^2` of a uniform line array."""
    m = int(n_mics)
    return spacing * spacing * m * (m * m - 1) / 12.0


def bearing_std(
    bandwidth: float | NDArray,
    snr: float | NDArray,
    spacing: float | NDArray,
    n_mics: int = 2,
    theta: float = 0.0,
    c: float = SPEED_OF_SOUND_AIR,
    centre: float = 0.0,
    coherent: bool = False,
    n_poses: int = 1,
) -> NDArray[np.float64] | float:
    """CRLB on the bearing (radians) of one echo from TDOA across a line array."""
    b = np.asarray(bandwidth, dtype=np.float64)
    beta = np.sqrt(b * b / 12.0 + (centre * centre if coherent else 0.0))
    s_tau = toa_std(beta, snr)
    d = np.asarray(spacing, dtype=np.float64)
    moment = d * d * n_mics * (n_mics * n_mics - 1) / 12.0
    return c * s_tau / (np.cos(theta) * np.sqrt(moment) * np.sqrt(n_poses))


def range_resolution(bandwidth: float, c: float = SPEED_OF_SOUND_AIR) -> float:
    """Rayleigh two-reflector range resolution :math:`c/(2B)`."""
    return c / (2.0 * bandwidth)


@dataclass(frozen=True)
class Setup:
    """A sensing hardware configuration in SI units."""

    name: str
    f_lo: float  # Hz
    f_hi: float  # Hz
    n_mics: int
    spacing: float  # m, between adjacent mics
    n_poses: int = 1

    @property
    def bandwidth(self) -> float:
        return self.f_hi - self.f_lo

    @property
    def centre(self) -> float:
        return 0.5 * (self.f_lo + self.f_hi)


def laptop_setups() -> list[Setup]:
    """Hardware setups for the design chart (plan 6.4.2).

    The first one is the project's reference scale ``LAPTOP_ROOM``
    (``simulation/units.py``: 20 cm baseline, the grid-resolved 300-1700 Hz
    band). The others span what laptops and phones can play and record.
    """
    lr = LAPTOP_ROOM
    return [
        Setup("LAPTOP_ROOM sim band, 2 mics", lr.band_hz[0], lr.band_hz[1], 2, lr.mic_baseline_m),
        Setup("laptop audible 0.3-8 kHz, 2 mics", 300.0, 8000.0, 2, lr.mic_baseline_m),
        Setup("laptop 0.3-8 kHz, 2 mics, 8 poses", 300.0, 8000.0, 2, lr.mic_baseline_m, 8),
        Setup("laptop 4-mic line array (7 cm)", 300.0, 8000.0, 4, 0.07),
        Setup("laptop near-ultrasound 17-21 kHz", 17000.0, 21000.0, 2, lr.mic_baseline_m),
        Setup("phone 0.1-20 kHz, 2 mics 15 cm", 100.0, 20000.0, 2, 0.15),
    ]


def design_table(
    setups: list[Setup] | None = None,
    snr_db: float = 20.0,
    distance_m: float = 3.0,
    c: float = SPEED_OF_SOUND_AIR,
) -> list[dict]:
    """Per-setup bounds and resolution limits (SI units) at one SNR.

    Columns: envelope and coherent range CRLB (mm), envelope bearing CRLB
    (degrees, broadside), the implied lateral position error at
    ``distance_m`` (cm), the Rayleigh range resolution :math:`c/2B` (cm), the
    smallest feature the band resolves :math:`\\lambda_\\min/2` (cm), and
    whether the adjacent-mic spacing exceeds :math:`\\lambda_\\min/2`
    (coherent bearing ambiguous).
    """
    snr = float(db_to_linear(snr_db))
    rows = []
    for s in setups or laptop_setups():
        n_obs = s.n_mics * s.n_poses
        sr_env = range_std(s.bandwidth, snr, c, n_obs=n_obs)
        sr_coh = range_std(s.bandwidth, snr, c, centre=s.centre, coherent=True, n_obs=n_obs)
        sb = bearing_std(s.bandwidth, snr, s.spacing, s.n_mics, c=c, n_poses=s.n_poses)
        lam_min = c / s.f_hi
        rows.append(
            {
                "setup": s.name,
                "band_hz": (s.f_lo, s.f_hi),
                "range_env_mm": 1e3 * float(sr_env),
                "range_coh_mm": 1e3 * float(sr_coh),
                "bearing_deg": float(np.degrees(sb)),
                "lateral_cm": 1e2 * distance_m * float(sb),
                "range_resolution_cm": 1e2 * range_resolution(s.bandwidth, c),
                "feature_cm": 1e2 * lam_min / 2.0,
                "grating_lobes": bool(s.spacing > lam_min / 2.0),
            }
        )
    return rows


def design_chart(path: str, snr_db: float = 20.0, c: float = SPEED_OF_SOUND_AIR) -> None:
    """Two-panel design chart: range CRLB vs bandwidth and bearing CRLB map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setups = laptop_setups()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bw = np.logspace(2, np.log10(25000), 200)
    for db, ls in ((0, ":"), (20, "-"), (40, "--")):
        ax[0].loglog(bw, 1e3 * range_std(bw, db_to_linear(db), c), "k" + ls, label=f"SNR {db} dB")
    for s in setups:
        ax[0].plot(
            s.bandwidth,
            1e3 * range_std(s.bandwidth, db_to_linear(snr_db), c, n_obs=s.n_mics * s.n_poses),
            "o",
            label=s.name,
        )
    ax[0].plot(bw, 1e3 * c / (2 * bw), color="0.6", lw=3, alpha=0.5, label="Rayleigh c/2B")
    ax[0].set_xlabel("bandwidth B [Hz]")
    ax[0].set_ylabel("range std [mm] (envelope CRLB)")
    ax[0].set_title("Range to a planar reflector")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=6.5)

    d = np.linspace(0.03, 0.4, 120)
    B = np.logspace(np.log10(500), np.log10(21000), 120)
    DD, BB = np.meshgrid(d, B)
    sig = np.degrees(bearing_std(BB, db_to_linear(snr_db), DD, 2, c=c))
    cs = ax[1].contourf(
        DD * 100, BB, np.log10(sig), levels=np.linspace(-2, 1.5, 15), cmap="viridis"
    )
    cl = ax[1].contour(DD * 100, BB, sig, levels=[0.1, 0.3, 1, 3], colors="w", linewidths=0.8)
    ax[1].clabel(cl, fmt="%g°", fontsize=7)
    ax[1].set_yscale("log")
    for s in setups:
        if s.n_mics == 2 and s.n_poses == 1:
            ax[1].plot(s.spacing * 100, s.bandwidth, "r*")
            ax[1].annotate(
                s.name.split(",")[0], (s.spacing * 100, s.bandwidth), fontsize=6, color="w"
            )
    fig.colorbar(cs, ax=ax[1], label="log10 bearing std [deg]")
    ax[1].set_xlabel("mic spacing [cm]")
    ax[1].set_ylabel("bandwidth B [Hz]")
    ax[1].set_title(f"Bearing (2 mics, broadside, SNR {snr_db:g} dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


__all__ = [
    "db_to_linear",
    "band_beta",
    "signal_beta",
    "toa_std",
    "range_std",
    "array_moment",
    "bearing_std",
    "range_resolution",
    "Setup",
    "laptop_setups",
    "design_table",
    "design_chart",
]
