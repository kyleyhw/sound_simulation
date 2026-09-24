"""Source and receiver models (plan Task 5.6).

All sources here are expressed as lists of ordinary ``Driver`` objects, so
they need no engine changes and run on every path (fast, general, GPU).

* **Scaled/delayed waveforms**: ``ScaledWaveform(w, gain, delay)`` evaluates
  ``gain * w(t - delay)``.
* **Sub-cell positions**: a source at a fractional position is split over the
  surrounding 2^d cells with bilinear/trilinear weights (they sum to 1).
  To first order this is a point source at the exact position. The
  receiver twin, ``sample_subcell``, interpolates the field the same way.
* **Band-limited (soft) injection**: ``spread`` > 0 distributes the source
  over a Gaussian of that width in cells. Its spatial spectrum
  :math:`e^{-k^2 s^2/2}` suppresses the poorly resolved high-wavenumber
  content that otherwise shows up as dispersive ringing.
* **Directivity**:
  - *Dipole*: two opposite monopoles a spacing :math:`d` apart. The far
    field is :math:`\\propto kd\\cos\\theta`, a figure-of-eight with a null
    broadside.
  - *Cardioid*: delay-and-subtract end-fire pair. The front monopole
    emits :math:`w(t)`; the rear one emits :math:`-w(t - d/c)`. On axis the
    rear contribution reaches the front aligned in time and doubles the
    output. Towards the rear it arrives with delay :math:`2d/c`. For
    :math:`kd \\ll 1` the pattern is :math:`\\propto (1+\\cos\\theta)`: a
    cardioid with its null at the back, the classic model of a baffled
    loudspeaker or a directional microphone.
* **Microphone model**: additive self-noise at a given SNR, fixed gain error,
  and ADC quantisation to ``bits`` over a full scale.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .setup import Driver
from .waveforms import Waveform


@dataclass
class ScaledWaveform(Waveform):
    """``gain * base(t - delay)``."""

    base: Waveform = field(default_factory=Waveform)
    gain: float = 1.0
    delay: float = 0.0

    def __call__(self, t: float) -> float:
        return float(self.gain * self.base(t - self.delay))


def subcell_weights(pos: Sequence[float]) -> list[tuple[tuple[int, ...], float]]:
    """Multilinear weights of a fractional position over its 2^d corner cells."""
    base = [int(np.floor(v)) for v in pos]
    frac = [float(v) - b for v, b in zip(pos, base)]
    out = []
    for corner in itertools.product((0, 1), repeat=len(pos)):
        w = 1.0
        for f, c in zip(frac, corner):
            w *= f if c else (1.0 - f)
        if w > 1e-12:
            out.append((tuple(b + c for b, c in zip(base, corner)), w))
    return out


def point_source(
    pos: Sequence[float],
    waveform: Waveform,
    *,
    spread: float = 0.0,
    gain: float = 1.0,
    delay: float = 0.0,
) -> list[Driver]:
    """Drivers for a (sub-cell, optionally band-limited) point source."""
    weights: dict[tuple[int, ...], float] = {}
    if spread <= 0:
        for cell, w in subcell_weights(pos):
            weights[cell] = weights.get(cell, 0.0) + w
    else:
        r = int(np.ceil(3 * spread))
        base = [int(round(v)) for v in pos]
        for off in itertools.product(range(-r, r + 1), repeat=len(pos)):
            cell = tuple(b + o for b, o in zip(base, off))
            d2 = sum((c - v) ** 2 for c, v in zip(cell, pos))
            weights[cell] = np.exp(-d2 / (2 * spread**2))
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    return [
        Driver(position=cell, waveform=ScaledWaveform(waveform, gain * w, delay))
        for cell, w in weights.items()
    ]


def dipole(
    pos: Sequence[float], direction: Sequence[float], waveform: Waveform, spacing: float = 2.0
) -> list[Driver]:
    """Figure-of-eight source along ``direction`` (unit vector)."""
    u = np.asarray(direction, float) / np.linalg.norm(direction)
    p = np.asarray(pos, float)
    return point_source(p + 0.5 * spacing * u, waveform, gain=0.5) + point_source(
        p - 0.5 * spacing * u, waveform, gain=-0.5
    )


def cardioid(
    pos: Sequence[float],
    direction: Sequence[float],
    waveform: Waveform,
    spacing: float = 2.0,
    wavespeed: float = 1.0,
) -> list[Driver]:
    """Delay-and-subtract end-fire pair radiating towards ``direction``."""
    u = np.asarray(direction, float) / np.linalg.norm(direction)
    p = np.asarray(pos, float)
    front = point_source(p + 0.5 * spacing * u, waveform, gain=0.5)
    rear = point_source(p - 0.5 * spacing * u, waveform, gain=-0.5, delay=spacing / wavespeed)
    return front + rear


def sample_subcell(field_arr: np.ndarray, pos: Sequence[float]) -> float:
    """Multilinear interpolation of a field at a fractional position."""
    return float(sum(w * field_arr[cell] for cell, w in subcell_weights(pos)))


@dataclass
class MicModel:
    """Receiver non-idealities applied to a recorded timeseries."""

    snr_db: Optional[float] = None  # self-noise relative to the signal RMS
    bits: Optional[int] = None  # ADC resolution over +-full_scale
    full_scale: Optional[float] = None  # defaults to 1.25 x the signal peak
    gain_error_db: float = 0.0

    def apply(self, x: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        y = np.asarray(x, dtype=np.float64) * 10 ** (self.gain_error_db / 20)
        if self.snr_db is not None:
            rms = float(np.sqrt(np.mean(y**2))) or 1.0
            y = y + rng.normal(0.0, rms * 10 ** (-self.snr_db / 20), size=y.shape)
        if self.bits is not None:
            fs = self.full_scale or 1.25 * float(np.max(np.abs(y)) or 1.0)
            q = 2 * fs / (2**self.bits)
            y = np.clip(np.round(y / q) * q, -fs, fs - q)
        return y.astype(np.float32)


__all__ = [
    "MicModel",
    "ScaledWaveform",
    "cardioid",
    "dipole",
    "point_source",
    "sample_subcell",
    "subcell_weights",
]
