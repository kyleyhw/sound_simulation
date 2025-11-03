# Documentation for `setup.py`

## 1. Purpose

`setup.py` is responsible for defining and generating the core components of the simulation environment other than the grid itself: **Drivers** (sound sources) and **Sensors** (virtual microphones).

It uses dataclasses for simple and clear data structures and provides generator classes to create randomized instances of these components, which is particularly useful for generating varied datasets for machine learning.

## 2. Implementation Details

### Data Structures

-   **`@dataclass class Driver`**: A simple container to associate a physical `location` on the grid with a specific `waveform` object. The `get_value(self, time)` method retrieves the waveform's amplitude at a given time.

-   **`@dataclass class Sensor`**: A container that holds the `location` of a sensor. The `timeseries` and `sample_rate` fields are populated *after* the simulation has run by the `assign_sensors` method in the `Simulate` class.

### Generator Classes

-   **`class GenerateSensor`**: A factory for creating `Sensor` objects.
    -   **Rationale**: Encapsulating the sensor generation logic in a class makes it easy to create sensors with consistent properties tied to a specific grid size. The `get_random_basic` method uses the `LocationGenerator` from `utils.py` to place sensors at random, non-edge locations, ensuring they capture meaningful data from within the simulation domain.

-   **`class GenerateDriver`**: A factory for creating `Driver` objects.
    -   **Rationale**: Similar to the sensor generator, this class simplifies the creation of drivers. The `get_random_cosine` method demonstrates how to create a driver with a specific waveform (`Cosine`) and randomized parameters (frequency, amplitude). This allows for the programmatic creation of diverse simulation scenarios.
