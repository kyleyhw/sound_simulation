from simulate import Simulate
from visualize import Visualize

gridsize = (128, 128)
gridstep = 10

duration = 100
timestep = 0.01

wavespeed = 330

simulation = Simulate(gridsize=gridsize, gridstep=gridstep, duration=duration, timestep=timestep, wavespeed=wavespeed)
history = simulation.run()

if __name__ == '__main__':
    print(history)

    visualize = Visualize(history=history)
    visualize.plot2D(show=True, save=True)

