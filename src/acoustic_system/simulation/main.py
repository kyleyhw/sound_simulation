import numpy as np

from .setup import Driver, Sensor
from .simulate import Simulate
from .visualize import Visualize
from .waveforms import Cosine


def main():
    """
    A standalone script for running and visualizing a simulation.
    This is not used by the web UI but serves as a useful testbed.
    """
    dims = 2
    grid_shape = (256, 256)
    duration = 500  # Number of steps

    # Initialize Simulation
    simulation = Simulate(grid_shape=grid_shape, wavespeed=1.0, timestep=0.1, gridstep=1.0)

    # Add Drivers
    waveform1 = Cosine(frequency=2, amplitude=1)
    driver1 = Driver(position=(grid_shape[0] // 4,) * dims, waveform=waveform1)
    simulation.drivers.append(driver1)

    waveform2 = Cosine(frequency=5, amplitude=0.5)
    driver2 = Driver(position=(grid_shape[0] * 3 // 4,) * dims, waveform=waveform2)
    simulation.drivers.append(driver2)

    # Add Sensors
    for i in range(3):
        sensor_pos = (grid_shape[0] // 4 * (i + 1),) * dims
        simulation.sensors.append(Sensor(position=sensor_pos, timeseries=None, sample_rate=None))

    # --- Run Simulation and Collect History ---
    history = np.zeros((duration,) + grid_shape, dtype=np.float32)
    for i in range(duration):
        simulation.step()
        history[i] = simulation.p
        # Simple progress indicator
        if (i % 100) == 0:
            print(f"Step {i}/{duration}")

    print("Simulation finished.")

    # --- Assign Sensor Data ---
    # Manually extract timeseries for each sensor from the history
    for sensor in simulation.sensors:
        timeseries_list = [history[t][sensor.position] for t in range(duration)]
        sensor.timeseries = np.array(timeseries_list)
        sensor.sample_rate = 1 / simulation.timestep

    # --- Visualize ---
    if dims == 2:
        params = {
            "grid_shape": grid_shape,
            "wavespeed": simulation.wavespeed,
            "timestep": simulation.timestep,
        }
        visualize = Visualize(history=history, params=params)
        visualize.plot2D(show=True, save=False)
        visualize.plot_sensor_timeseries(sensors=simulation.sensors, show=True, save=False)
        visualize.plot_sensor_fft(sensors=simulation.sensors, show=True, save=False)


if __name__ == "__main__":
    main()
