from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .waveforms import Cosine, Waveform


@dataclass(frozen=True)
class Driver:
    """A point source: a grid position and a callable waveform p_src(t).

    Frozen so a driver's position cannot change behind ``Simulate``'s
    single-driver fast-path cache; build a new ``Driver`` and use
    ``Simulate.set_drivers`` / ``add_driver`` instead. Positions are
    normalised to a tuple of Python ints (a float coordinate would
    otherwise fail as a NumPy index on the first step).
    """

    position: Tuple[int, ...]
    waveform: Waveform = field(default_factory=Cosine)

    def __post_init__(self) -> None:
        coords = tuple(int(round(float(c))) for c in np.atleast_1d(self.position))
        object.__setattr__(self, "position", coords)

    def get_value(self, time: float) -> float:
        return float(self.waveform(time))


@dataclass
class Sensor:
    """A recording point; ``timeseries`` is filled by the caller."""

    position: Tuple[int, ...]
    timeseries: Optional[np.ndarray] = None
    sample_rate: Optional[float] = None
