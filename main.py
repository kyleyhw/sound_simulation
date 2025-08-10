from simulate import Simulate
from visualize import Visualize

if __name__ == '__main__':
    dims = 2

    if dims == 2:
        gridsize = (64, 64)
    if dims == 3:
        gridsize = (64, 64, 64)

    gridstep = 10

    duration = 40
    timestep = 0.01

    wavespeed = 330

    simulation = Simulate(gridsize=gridsize, gridstep=gridstep, duration=duration, timestep=timestep,
                          wavespeed=wavespeed)
    history = simulation.run()

    isStable = simulation.check_stability()
    if isStable:
        print('this is stable')
    else:
        print('this is unstable')

    # print(history)

    visualize = Visualize(history=history, params=simulation.get_params())
    if dims == 2:
        visualize.plot2D(show=True, save=True)
    if dims == 3:
        visualize.plot3D(show=True, save=True)
