from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .utils import LocationGenerator
from .waveforms import Cosine, Waveform


@dataclass
class Driver:
    """A point source: a grid position and a callable waveform p_src(t)."""

    position: Tuple[int, ...]
    waveform: Waveform = field(default_factory=Cosine)

    def get_value(self, time: float) -> float:
        return float(self.waveform(time))


@dataclass
class Sensor:
    position: Tuple[int, ...]
    timeseries: Optional[np.ndarray] = None
    sample_rate: Optional[float] = None


class GenerateSensor:
    def __init__(self, gridsize: Tuple[int, ...]):
        self.gridsize = gridsize

    def get_random_basic(self, position=None, detailed: bool = False):
        if position is None:
            position = LocationGenerator(gridsize=self.gridsize).get_new_location()
        sensor = Sensor(position=position, timeseries=None, sample_rate=None)
        if detailed:
            return sensor, (position,)
        return sensor


class GenerateDriver:
    def __init__(self, gridsize: Tuple[int, ...]):
        self.gridsize = gridsize

    def get_random_cosine(self, position=None, frequency=None, amplitude=None, detailed: bool = False):
        if position is None:
            position = LocationGenerator(gridsize=self.gridsize).get_new_location()
        if frequency is None:
            frequency = float(np.random.uniform(low=0.02, high=0.1))
        if amplitude is None:
            amplitude = float(np.random.uniform(low=0.5, high=0.9))
        waveform = Cosine(frequency=frequency, amplitude=amplitude)
        driver = Driver(position=position, waveform=waveform)
        if detailed:
            return driver, (position, frequency, amplitude)
        return driver
