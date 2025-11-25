import numpy as np
from dataclasses import dataclass


# --- Waveform Definitions ---

class Waveform:
    """Base class for all driver waveforms."""

    def __call__(self, t):
        # The __call__ method makes instances of the class callable like functions
        raise NotImplementedError


@dataclass
class Cosine(Waveform):
    frequency: float = 3.0
    amplitude: float = 1.0

    def __call__(self, t):
        return self.amplitude * np.cos(2 * np.pi * self.frequency * t)


@dataclass
class GaussianPulse(Waveform):
    amplitude: float = 1.0
    center_time: float = 0.5
    width: float = 0.1

    def __call__(self, t):
        return self.amplitude * np.exp(-((t - self.center_time) ** 2) / (2 * self.width ** 2))


# This registry maps string names to the actual Waveform classes
waveform_registry = {
    'Cosine': Cosine,
    'GaussianPulse': GaussianPulse,
}