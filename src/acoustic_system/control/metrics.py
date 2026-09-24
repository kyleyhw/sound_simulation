"""Sound-zone metrics (plan 7.1.2).

Notation: :math:`H_B(f)` (:math:`M_B \\times S`) and :math:`H_D(f)`
(:math:`M_D \\times S`) are the transfer functions from the S speakers to
the bright-zone and dark-zone points, and :math:`q(f) \\in \\mathbb{C}^S`
are the speaker weights (drive spectra). The zone pressures are
:math:`p_B = H_B q` and :math:`p_D = H_D q`.

* **Acoustic contrast** (Choi & Kim 2002), the ratio of mean squared
  pressure in the two zones:

  .. math::
      C = 10 \\log_{10} \\frac{\\|p_B\\|^2 / M_B}{\\|p_D\\|^2 / M_D} \\;\\text{dB}.

  The broadband (band) contrast sums both energies over frequency first
  (the contrast a flat-spectrum programme would get), so it is not an
  average of per-frequency decibels.
* **Normalised reproduction error** against a target field :math:`p_T`:

  .. math::
      \\epsilon = \\frac{\\|p - p_T\\|^2}{\\|p_T\\|^2}.

* **Array effort**: the drive energy needed for a given loudness, relative
  to a single reference speaker producing the same mean squared pressure in
  the bright zone:

  .. math::
      E = 10 \\log_{10} \\frac{\\|q\\|^2}{|q_r|^2}, \\qquad
      |q_r|^2 = \\frac{\\|H_B q\\|^2}{\\|H_B e_r\\|^2}.

  0 dB means the array works as hard as one speaker; large positive values
  flag weights that cancel each other (ill-conditioned, fragile designs).

Time-domain versions (:func:`energy_contrast_db`, :func:`band_energy`)
score recordings from the engine.
"""

from __future__ import annotations

import numpy as np


def _db(x: "np.ndarray | float") -> "np.ndarray | float":
    return 10.0 * np.log10(np.maximum(x, 1e-300))


def zone_energy(H: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Mean squared pressure :math:`\\|H q\\|^2 / M` per frequency.

    ``H`` is (F, M, S) and ``q`` (F, S); returns (F,).
    """
    p = np.einsum("fms,fs->fm", H, q)
    return np.mean(np.abs(p) ** 2, axis=-1)


def acoustic_contrast_db(p_bright: np.ndarray, p_dark: np.ndarray, axis: int = -1) -> np.ndarray:
    """Contrast in dB from zone pressures (complex or real), averaged along ``axis``."""
    eb = np.mean(np.abs(p_bright) ** 2, axis=axis)
    ed = np.mean(np.abs(p_dark) ** 2, axis=axis)
    return np.asarray(_db(eb / np.maximum(ed, 1e-300)))


def contrast_spectrum_db(Hb: np.ndarray, Hd: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-frequency contrast (F,) of weights ``q`` (F, S)."""
    return np.asarray(_db(zone_energy(Hb, q) / np.maximum(zone_energy(Hd, q), 1e-300)))


def band_contrast_db(
    Hb: np.ndarray, Hd: np.ndarray, q: np.ndarray, weights: "np.ndarray | None" = None
) -> float:
    """Broadband contrast: :math:`\\sum_f w_f E_B(f) / \\sum_f w_f E_D(f)` in dB."""
    w = np.ones(len(q)) if weights is None else np.asarray(weights, dtype=np.float64)
    return float(_db(np.sum(w * zone_energy(Hb, q)) / np.sum(w * zone_energy(Hd, q))))


def reproduction_error(p: np.ndarray, target: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalised reproduction error :math:`\\|p - p_T\\|^2 / \\|p_T\\|^2`."""
    num = np.sum(np.abs(p - target) ** 2, axis=axis)
    den = np.sum(np.abs(target) ** 2, axis=axis)
    return num / np.maximum(den, 1e-300)


def reproduction_error_db(p: np.ndarray, target: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.asarray(_db(reproduction_error(p, target, axis)))


def array_effort_db(q: np.ndarray, Hb: np.ndarray, ref: int = 0) -> np.ndarray:
    """Array effort (F,) in dB relative to speaker ``ref`` at equal bright-zone level."""
    eb = zone_energy(Hb, q)
    e_ref = np.mean(np.abs(Hb[:, :, ref]) ** 2, axis=-1)
    q_ref2 = eb / np.maximum(e_ref, 1e-300)
    return np.asarray(_db(np.sum(np.abs(q) ** 2, axis=-1) / np.maximum(q_ref2, 1e-300)))


def band_energy(
    y: np.ndarray, dt: float, band: "tuple[float, float] | None" = None, n_fft: int | None = None
) -> np.ndarray:
    """Energy of recordings ``y`` (..., T) inside ``band`` [Hz] (Parseval, per channel)."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[-1] if n_fft is None else int(n_fft)
    if band is None:
        return np.sum(y**2, axis=-1)
    Y = np.fft.rfft(y, n=n, axis=-1)
    f = np.fft.rfftfreq(n, dt)
    sel = (f >= band[0]) & (f <= band[1])
    w = np.where((f == 0) | (f == f[-1]), 1.0, 2.0)  # one-sided spectrum weights
    return np.sum(w[sel] * np.abs(Y[..., sel]) ** 2, axis=-1) / n


def energy_contrast_db(
    y_bright: np.ndarray,
    y_dark: np.ndarray,
    dt: float | None = None,
    band: "tuple[float, float] | None" = None,
) -> float:
    """Time-domain contrast of recordings (M_B, T) and (M_D, T), optionally in a band."""
    if band is not None and dt is None:
        raise ValueError("band needs dt")
    eb = band_energy(y_bright, dt or 1.0, band).mean()
    ed = band_energy(y_dark, dt or 1.0, band).mean()
    return float(_db(eb / max(float(ed), 1e-300)))


__all__ = [
    "acoustic_contrast_db",
    "array_effort_db",
    "band_contrast_db",
    "band_energy",
    "contrast_spectrum_db",
    "energy_contrast_db",
    "reproduction_error",
    "reproduction_error_db",
    "zone_energy",
]
