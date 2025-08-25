import numpy as np
import h5py
from datetime import datetime

from simulate import Simulate
from visualize import Visualize
from setup import GenerateDriver
from data_io import save_full_simulation_results



if __name__ == '__main__':
    dims = 2

    if dims == 2:
        gridsize = (256, 256)

    gridstep = 8

    duration = 5
    timestep = 0.01

    wavespeed = 330



    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    number_of_runs = 1
    filename = f"simulation_archive_{timestamp}_{number_of_runs}_runs.hdf5"

    with h5py.File('./training_data/'+filename, 'w') as hdf5_file:
        for i in range(number_of_runs):
            simulation = Simulate(gridsize=gridsize, gridstep=gridstep, duration=duration, timestep=timestep,
                                  wavespeed=wavespeed)
            driver_gen = GenerateDriver(gridsize=gridsize)

            number_of_speakers = np.random.randint(low=1, high=6)
            for i in range(number_of_speakers):
                driver, driver_params = driver_gen.get_random_cosine(verbose=True)
                simulation.add_driver(driver=driver)
                print(driver_params)

            simulation.check_stability()

            history = simulation.run()

            save_full_simulation_results(simulation_object=simulation, hdf5_file=hdf5_file, simulation_id=1)


    visualize = False
    if visualize:
        params = None

        visualize = Visualize(history=history, params=params).plot2D(show=True)
