from dataclasses import dataclass
from typing import Tuple
from waveforms import Waveform, Cosine


@dataclass
class Driver:
    """A class to hold a location and a Waveform object."""
    location: Tuple[int, ...]
    waveform: Waveform = Cosine

    def get_value(self, time):
        return self.waveform(time)  # Call the waveform object directly