# Documentation for `main.py`

## 1. Purpose

`main.py` serves as the primary entry point for running a simulation with a predefined configuration. It is a script designed to be executed directly (`if __name__ == '__main__':`).

Its main role is to demonstrate how to instantiate the `Simulate` class, add drivers and sensors, run the simulation, and generate visualizations from the results.

## 2. Implementation Details

This script is not designed for general-purpose use but rather as a specific example. Here is a breakdown of its workflow:

1.  **Configuration**: It starts by setting hard-coded parameters for the simulation:
    -   `dims`: The number of spatial dimensions (2 or 3).
    -   `gridsize`: The size of the grid, determined by `dims`.
    -   `duration`, `timestep`, `wavespeed`: Core physical and simulation parameters.

2.  **Initialization**: It creates an instance of the `Simulate` class with the specified parameters.

3.  **Adding Components**:
    -   It creates and adds two `Driver` objects with different `Cosine` waveforms, frequencies, and amplitudes at fixed locations.
    -   It creates and adds three `Sensor` objects at different locations.

4.  **Stability Check**: It calls `simulation.check_stability()` to verify that the chosen parameters satisfy the CFL condition, issuing a warning if they do not.

5.  **Execution**: It calls `simulation.run()` to execute the main FDTD loop and then `simulation.assign_sensors()` to populate the sensor objects with the recorded time-series data.

6.  **Visualization**: Based on the number of dimensions, it calls the appropriate methods from the `Visualize` class:
    -   For 2D simulations, it plots the time-series data and the Fast Fourier Transform (FFT) for each sensor.
    -   For 3D simulations, it is configured to generate a 3D animation of the wave field (note: the `plot3D` method using `mayavi` is currently commented out in the source code).

## 3. Design Rationale

-   **Example Script**: The primary value of this script is as a clear, working example of how to use the simulation library's components together. It shows the end-to-end process from setup to visualization.
-   **Configuration Hub**: It acts as a central place to set all the parameters for a specific simulation run. For more advanced use, these hard-coded values could be replaced by command-line arguments or a configuration file.
