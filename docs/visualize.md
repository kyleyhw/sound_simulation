# Documentation for `visualize.py`

## 1. Purpose

`visualize.py` is dedicated to creating visual representations of the simulation results. This includes animating the wave propagation over time and plotting the data recorded by the sensors.

## 2. Scientific Principles

-   **Time-Domain Visualization**: The 2D and 3D animations directly plot the pressure field `p(x, y, t)` at each frame. This provides an intuitive understanding of how the waves propagate, reflect, and interfere throughout the grid.

-   **Frequency-Domain Analysis (FFT)**: The `plot_sensor_fft` method applies the Fast Fourier Transform (FFT) to the time-series data recorded by a sensor. The FFT is an algorithm that computes the Discrete Fourier Transform (DFT), converting a signal from the time domain to the frequency domain.
    -   **Rationale**: This is a critical tool for analysis. By examining the signal in the frequency domain, one can identify the dominant frequencies present at the sensor's location. This is useful for verifying that the simulated drivers are producing the correct frequencies and for analyzing how the environment (e.g., room resonances) affects the frequency content of the sound.

## 3. Implementation Details

### Class: `Visualize`

-   **`__init__(self, history, params=None)`**: The constructor takes the full `history` of the simulation (a large array containing the pressure field for every time step) and the simulation parameters.

-   **`plot2D(...)`**: This method generates a 2D animation of the wave field.
    -   **How it Works**: It uses `matplotlib.animation.FuncAnimation`, a standard tool for creating animations in Matplotlib. It initializes a plot with the first frame of the history and provides an update function (`_update2D`) that `FuncAnimation` calls for each subsequent frame to set the new data. The result can be shown on screen or saved as an `.mp4` file using `ffmpeg`.

-   **`plot3D(...)`**: This method (currently commented out) is intended to generate a 3D volumetric animation using the `Mayavi` library. Mayavi is a powerful tool for 3D scientific data visualization. The logic involves creating a 3D scalar field and updating its data for each time step in an animation loop.

-   **`plot_sensor_timeseries(...)`**: This method creates a simple line plot of the pressure amplitude recorded by each sensor over time. This shows the raw signal captured by the virtual microphones.

-   **`plot_sensor_fft(...)`**: This method calculates and plots the frequency spectrum for each sensor.
    -   **How it Works**: For each sensor's timeseries, it calls `np.fft.fft` to compute the FFT. It also calculates the corresponding frequencies using `np.fft.fftfreq`. The `np.fft.fftshift` function is used to re-center the frequency spectrum so that the zero-frequency component is in the middle, which is a standard convention for visualization.
