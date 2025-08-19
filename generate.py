import numpy as np

from simulate import Simulate
from visualize import Visualize
import waveforms
from setup import GenerateDriver
from utils import LocationGenerator



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

    history = simulation.run()

    simulation.check_stability()


    visualize = True
    if visualize:
        params = None

        visualize = Visualize(history=history, params=params).plot2D(show=True)
