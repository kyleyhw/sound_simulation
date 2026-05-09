import os
from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm

from .data_io import SaveSimulationResults
from .setup import GenerateDriver, GenerateSensor
from .simulate import Simulate


def main():
    grid_shape = (256, 256)
    duration = 500  # Number of steps
    wavespeed = 1.0
    timestep = 0.1
    gridstep = 1.0

    save_type = "sensor_results"
    number_of_runs = 3

    directory_path = "training_data"
    os.makedirs(directory_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"simulation_archive_{timestamp}_{number_of_runs}x_{save_type}.hdf5"

    params = {
        "grid_shape": grid_shape,
        "wavespeed": wavespeed,
        "timestep": timestep,
        "gridstep": gridstep,
    }

    with h5py.File(f"./{directory_path}/{filename}", "w") as hdf5_file:
        simsaver = SaveSimulationResults(hdf5_file=hdf5_file)
        hdf5_file.attrs["save_type"] = save_type

        for run_number in tqdm(range(number_of_runs), desc="generating data"):
            # --- Setup Simulation ---
            simulation = Simulate(
                grid_shape=grid_shape, wavespeed=wavespeed, timestep=timestep, gridstep=gridstep
            )
            driver_gen = GenerateDriver(gridsize=grid_shape)
            sensor_gen = GenerateSensor(gridsize=grid_shape)

            # Add random drivers
            number_of_speakers = np.random.randint(low=1, high=6)
            for _ in range(number_of_speakers):
                driver = driver_gen.get_random_cosine()
                simulation.drivers.append(driver)

            # Add a random sensor
            sensor = sensor_gen.get_random_basic()
            simulation.sensors.append(sensor)

            # --- Run Simulation and Collect History ---
            history = np.zeros((duration,) + grid_shape, dtype=np.float32)
            for i in range(duration):
                simulation.step()
                history[i] = simulation.p

            # --- Assign Sensor Data ---
            for s in simulation.sensors:
                timeseries_list = [history[t][s.position] for t in range(duration)]
                s.timeseries = np.array(timeseries_list)
                s.sample_rate = 1 / simulation.timestep

            # --- Save Results ---
            if save_type == "sensor_results":
                simsaver.save_sensor_results(
                    simulation_id=run_number,
                    sensors=simulation.sensors,
                    params=params,
                    drivers=simulation.drivers,
                )
            elif save_type == "full_history":
                simsaver.save_full_history(
                    simulation_id=run_number,
                    history=history,
                    params=params,
                    drivers=simulation.drivers,
                )


if __name__ == "__main__":
    main()
