# Documentation for `waveforms.py`

## 1. Purpose

This script defines the mathematical functions for the various sound sources (drivers) that can be used in the simulation. It establishes a clear, extensible system for adding new types of waveforms.

## 2. Scientific Principles

The script implements standard mathematical functions that are commonly used to model physical wave sources.

-   **Cosine Wave**: A simple, continuous sinusoidal wave defined by its frequency and amplitude. It represents a pure tone.
    $$ f(t) = A \cos(2\pi f t) $$
    Where `A` is amplitude and `f` is frequency.

-   **Gaussian Pulse**: A pulse with a Gaussian-shaped envelope. It is localized in time and is useful for simulating transient events like a click or a tap. Its shape in the frequency domain is also a Gaussian, meaning it is localized in frequency as well.
    $$ f(t) = A \exp\left(-\frac{(t - t_0)^2}{2\sigma^2}\right) $$
    Where `A` is amplitude, `t_0` is the center time of the pulse, and `σ` is the width.

## 3. Implementation Details

### Base Class

-   **`class Waveform`**: This is an abstract base class that defines the interface for all waveform types.
    -   **`__call__(self, t)`**: The key feature of this class is the `__call__` method. This is a special Python method that allows an *instance* of a class to be called as if it were a function. This provides a clean and intuitive way for the `Driver` object to get a value from its waveform at a specific time `t` (e.g., `driver.waveform(time)`).

### Waveform Implementations

-   **`@dataclass class Cosine(Waveform)`**: Implements a cosine wave. It inherits from `Waveform` and its `__call__` method returns the value of the cosine function for a given time `t`.

-   **`@dataclass class GaussianPulse(Waveform)`**: Implements a Gaussian pulse. It also inherits from `Waveform` and implements the corresponding mathematical function.

### Design Rationale

-   **Object-Oriented Approach**: Using a base class and inheriting from it makes the system highly extensible. To add a new waveform (e.g., a sine wave or a square wave), one only needs to create a new class that inherits from `Waveform` and implements the `__call__` method with the desired mathematical function.
-   **Dataclasses**: Using the `@dataclass` decorator automatically generates methods like `__init__` and `__repr__`, reducing boilerplate code and making the waveform definitions clean and readable.
-   **`waveform_registry`**: This dictionary acts as a registry that maps string names to the actual waveform classes. This is a powerful pattern that allows for selecting and instantiating waveforms based on a string name, which can be useful for setting up simulations from configuration files or user input.
