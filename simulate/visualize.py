from tqdm import tqdm
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
# from mayavi import mlab
# from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction

class Visualize:
    def __init__(self, history, params):
        self.history = history
        self.number_of_frames = history.shape[0]
        self.params = params

    def _update2D(self, frame, plot_object):
        plot_object.set_data(self.history[frame])
        return (plot_object,)

    def plot2D(self, show=False, save=False):
        fig, ax = plt.subplots()

        plot_object = ax.imshow(self.history[0], cmap='viridis', vmin=-1.0, vmax=1.0)

        fig.colorbar(plot_object, ax=ax)

        animation = FuncAnimation(fig,
                                  self._update2D,
                                  frames=self.number_of_frames,
                                  fargs=(plot_object,),
                                  interval=10,
                                  blit=False)
        if self.params:
            params_string = '|'.join([str(key) + ' ' + str(self.params[key]) for key in self.params.keys()])
            fig.suptitle(params_string)

        if show:
            plt.show()
        if save:
            # 1. Create a tqdm progress bar instance
            with tqdm(total=self.number_of_frames, desc="saving") as pbar:
                # 2. Define the callback function that updates the bar
                def progress_update(current_frame, total_frames):
                    pbar.update(1)

                # 3. Pass the function to the save method
                animation.save('simulation_2d.mp4',
                               writer='ffmpeg',
                               dpi=150,
                               progress_callback=progress_update)

    def plot3D(self, show=False, save=False):
        # Transpose the entire history array once at the beginning
        history_transposed = np.transpose(self.history, (0, 3, 2, 1))

        # --- 1. Setup the Scene ---
        fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

        source = mlab.pipeline.scalar_field(history_transposed[0])
        volume = mlab.pipeline.volume(source)

        # Apply Color and Opacity Transfer Functions
        ctf = ColorTransferFunction()
        ctf.add_rgb_point(-0.1, 0.0, 0.0, 1.0)  # Blue
        ctf.add_rgb_point(0.0, 0.0, 1.0, 0.0)  # Green
        ctf.add_rgb_point(0.1, 1.0, 0.0, 0.0)  # Red
        otf = PiecewiseFunction()
        otf.add_point(-0.1, 0.8)
        otf.add_point(0.0, 0.0)
        otf.add_point(0.1, 0.8)
        volume._volume_property.set_color(ctf)
        volume._otf = otf
        volume.update_ctf = True

        mlab.outline()
        mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')

        # --- 2. Create the Animation ---
        @mlab.animate(delay=50)
        def anim():
            pbar = tqdm(total=self.number_of_frames, desc="Animating Frames")
            for i in range(self.number_of_frames):
                # Calculate the current time
                current_time = i * self.params['timestep']

                # THE FIX: Update the scene title with the current time
                mlab.title(f'Time: {current_time:.2f} s', color=(0, 0, 0))

                # Update the 3D data source
                source.mlab_source.scalars = history_transposed[i]
                volume.update_ctf = True

                pbar.update(1)
                yield

        # Run the animation and show the window
        anim()
        mlab.show()


if __name__ == '__main__':
    print('hello visualize')