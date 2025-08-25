import numpy as np
import h5py

from simulate import Simulate
from visualize import Visualize
from setup import GenerateDriver
from io import save_full_simulation_results



if __name__ == '__main__':
    dims = 2

    if dims == 2:
        gridsize = (256, 256)

    gridstep = 8

    duration = 5
    timestep = 0.01

    wavespeed = 330

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

    filename = 'single_simulation.hdf5'
    with h5py.File('/training_data/', 'w') as hdf5_file:
        save_full_simulation_results(simulation_object=simulation, hdf5_file=hdf5_file, simulation_id=1)


    visualize = True
    if visualize:
        params = None

        visualize = Visualize(history=history, params=params).plot2D(show=True)
