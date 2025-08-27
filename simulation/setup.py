from dataclasses import dataclass
from typing import Tuple
import numpy as np

from waveforms import Waveform, Cosine
from utils import LocationGenerator


@dataclass
class Driver:
    """A class to hold a location and a Waveform object."""
    location: Tuple[int, ...]
    waveform: Waveform = Cosine

    def get_value(self, time):
        return self.waveform(time)  # Call the waveform object directly

@dataclass
class Sensor:
    location: Tuple[int, ...]
    timeseries: np.ndarray
    sample_rate: float

class GenerateSensor:
    def __init__(self, gridsize: tuple):
        self.gridsize = gridsize

    def get_random_basic(self, location=None, detailed=False):
        if not location:
            location = LocationGenerator(gridsize=self.gridsize).get_new_location()

        sensor = Sensor(location=location, timeseries=None, sample_rate=None)

        if detailed:
            print(sensor, (location,))
            return sensor, (location,)
        return sensor


class GenerateDriver:
    def __init__(self, gridsize: tuple):
        self.gridsize = gridsize

    def get_random_cosine(self, location=None, frequency=None, amplitude=None, detailed=False):
        if not location:
            location = LocationGenerator(gridsize=self.gridsize).get_new_location()
        if not frequency:
            frequency = np.random.randint(low=1, high=10, size=1)
        if not amplitude:
            amplitude = np.random.uniform(low=0.5, high=0.9, size=1)
        waveform = Cosine(frequency=frequency, amplitude=amplitude)

        driver = Driver(location=location, waveform=waveform)

        if detailed:
            print(driver, (location, frequency, amplitude))
            return driver, (location, frequency, amplitude)
        return driver