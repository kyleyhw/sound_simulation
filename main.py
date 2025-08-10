from simulate import Simulate
from visualize import Visualize

gridsize = (16, 16)
gridstep = 100

duration = 100
timestep = 0.1

wavespeed = 330

simulation = Simulate(gridsize=gridsize, gridstep=gridstep, duration=duration, timestep=timestep, wavespeed=wavespeed)
history = simulation.run()

if __name__ == '__main__':
    print(history)

    visualize = Visualize(history=history)
    visualize.plot2D()
