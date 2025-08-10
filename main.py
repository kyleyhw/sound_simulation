from simulate import Simulate
from visualize import Visualize

if __name__ == '__main__':
    dims = 3

    if dims == 2:
        gridsize = (64, 128)
    if dims == 3:
        gridsize = (32, 32, 32)

    gridstep = 10

    duration = 2
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


    if dims == 2:
        visualize = Visualize(history=history, params=simulation.get_params())
        visualize.plot2D(show=True, save=False)
    if dims == 3:
        # visualize = Visualize(history=history[:, 4, :, :], params=simulation.get_params())
        # visualize.plot2D(show=True, save=False)
        visualize = Visualize(history=history, params=simulation.get_params())
        visualize.plot3D(show=False, save=True)
        # print(history)