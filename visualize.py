from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

        plot_object = ax.imshow(self.history[0], cmap='viridis', vmin=-0.5, vmax=1.0)

        fig.colorbar(plot_object, ax=ax)

        animation = FuncAnimation(fig,
                                  self._update2D,
                                  frames=self.number_of_frames,
                                  fargs=(plot_object,),
                                  interval=10,
                                  blit=False)
        params_string = '|'.join([str(key) + ' ' + str(self.params[key]) for key in self.params.keys()])
        fig.suptitle(params_string)

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

        # This block handles saving the animation to a file NON-interactively
        if save:
            # Create the grid object
            grid = pv.ImageData()
            grid.dimensions = self.history[0].shape
            grid["pressure"] = self.history[0].flatten(order="F")

            # Use off_screen=True for saving without a popup window
            plotter = pv.Plotter(off_screen=True)
            plotter.add_volume(grid, scalars="pressure", cmap="viridis", clim=[-0.5, 1.0])

            print("Opening movie file for saving...")
            plotter.open_movie('simulation_3d.mp4')

            for frame_data in tqdm(self.history[1:], desc="Rendering 3D Animation"):
                grid["pressure"] = frame_data.flatten(order="F")
                plotter.write_frame()

            plotter.close()
            print("3D animation saved to simulation_3d.mp4")

        # This block handles showing the animation in a LIVE, interactive window
        if show:
            # Use BackgroundPlotter for live rendering
            plotter = BackgroundPlotter()

            grid = pv.ImageData()
            grid.dimensions = self.history[0].shape
            grid["pressure"] = self.history[0].flatten(order="F")

            plotter.add_volume(grid, scalars="pressure", cmap="viridis", clim=[-0.5, 1.0])

            print("Starting live interactive animation...")
            # Loop through the frames to update the plot
            for frame_data in self.history[1:]:
                grid["pressure"] = frame_data.flatten(order="F")
                plotter.render()  # Update the render
                time.sleep(0.02)  # Control the frame rate

            print("Animation finished. You can continue to interact with the window.")
            # The window will remain open until you manually close it




if __name__ == '__main__':
    print('hello visualize')