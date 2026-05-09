from dataclasses import dataclass

import numpy as np


class Waveform:
    """Base class for callable driver waveforms p_src(t)."""

    def __call__(self, t: float) -> float:
        raise NotImplementedError


@dataclass
class Cosine(Waveform):
    """Continuous monochromatic source: p_src(t) = A cos(2 pi f t).

    For a leap-frog FDTD with timestep dt, the source must be sampled
    well below Nyquist: f * dt < 0.5; ten or more samples per period
    (f * dt < 0.1) is comfortable.
    """

    frequency: float = 0.05
    amplitude: float = 1.0

    def __call__(self, t: float) -> float:
        return self.amplitude * np.cos(2.0 * np.pi * self.frequency * t)


@dataclass
class GaussianPulse(Waveform):
    """Single-lobe Gaussian: p_src(t) = A exp(-(t - t0)^2 / (2 sigma^2)).

    Generates a low-pass DC-rich impulse. Spectrally bright but contains a
    persistent DC offset, which can leave a quasi-static residual in
    closed (hard-walled) domains.
    """

    amplitude: float = 1.0
    center_time: float = 0.5
    width: float = 0.1

    def __call__(self, t: float) -> float:
        return self.amplitude * np.exp(-((t - self.center_time) ** 2) / (2.0 * self.width**2))


@dataclass
class RickerWavelet(Waveform):
    """Ricker (Mexican-hat) wavelet — second derivative of a Gaussian.

    p_src(t) = A (1 - 2 (pi f (t - t0))^2) exp(-(pi f (t - t0))^2)

    Standard FDTD excitation: zero mean (no DC pump-up), single dominant
    frequency f, finite duration ~ 2/f around t0. Choose t0 >= 1/f so the
    pulse starts near zero and ends near zero, leaving the field clean.
    """

    amplitude: float = 1.0
    frequency: float = 0.1
    delay: float = 20.0

    def __call__(self, t: float) -> float:
        arg = np.pi * self.frequency * (t - self.delay)
        arg2 = arg * arg
        return self.amplitude * (1.0 - 2.0 * arg2) * np.exp(-arg2)


waveform_registry = {
    "Cosine": Cosine,
    "GaussianPulse": GaussianPulse,
    "RickerWavelet": RickerWavelet,
}
