import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Visualize:
    def __init__(self, history):
        self.history = history
        self.number_of_frames = history.shape[0]

    def _update2D(self, frame, plot_object):
        plot_object.set_data(self.history[frame])
        return (plot_object,)

    def plot2D(self):
        fig, ax = plt.subplots()

        plot_object = ax.imshow(self.history[0], cmap='viridis', vmin=-0.5, vmax=1.0)

        fig.colorbar(plot_object, ax=ax)

        animation = FuncAnimation(fig,
                                  self._update2D,
                                  frames=self.number_of_frames,
                                  fargs=(plot_object,),
                                  interval=20,
                                  blit=False)

        plt.show()




if __name__ == '__main__':
    print('hello visualize')