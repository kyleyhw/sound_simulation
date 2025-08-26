import numpy as np

from simulate import Simulate
from visualize import Visualize
from setup import Driver
import waveforms

if __name__ == '__main__':
    dims = 2

    if dims == 2:
        gridsize = (256, 256)
    if dims == 3:
        gridsize = (32, 32, 32)

    gridstep = 8

    duration = 5
    timestep = 0.01

    wavespeed = 330

    simulation = Simulate(gridsize=gridsize, gridstep=gridstep, duration=duration, timestep=timestep,
                          wavespeed=wavespeed)
    waveform = waveforms.Cosine(frequency=2, amplitude=1)
    driver = Driver(location=(int(gridsize[0]/4),) * dims, waveform=waveform)
    simulation.add_driver(driver=driver)
    waveform2 = waveforms.Cosine(frequency=5, amplitude=0.5)
    driver2 = Driver(location=(int(gridsize[0]*3/4),) * dims, waveform=waveform2)
    simulation.add_driver(driver=driver2)
    history = simulation.run()

    simulation.check_stability()

    # print(history)


    if dims == 2:
        visualize = Visualize(history=history, params=simulation.get_params())
        visualize.plot2D(show=True, save=False)
    if dims == 3:
        # visualize = Visualize(history=history[:, 4, :, :], params=simulation.get_params())
        # visualize.plot2D(show=True, save=False)
        visualize = Visualize(history=history, params=simulation.get_params())
        visualize.plot3D(show=False, save=True)
        # print(history)