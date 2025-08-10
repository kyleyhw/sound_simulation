from tqdm import tqdm
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import rescale # For resizing data if needed
from tqdm import tqdm

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
        if not save:
            return

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('viridis')

        # --- The Update Function ---
        def update(frame_number, history_data):
            ax.clear()
            frame = history_data[frame_number]

            # 1. Normalize the data to be between 0 and 1 for the colormap
            # We use a fixed range like [-0.1, 0.1] to keep the colors consistent
            norm_data = np.clip((frame + 0.1) / 0.2, 0, 1)

            # 2. Map the normalized data to RGBA colors
            colors = cmap(norm_data)

            # 3. Set the alpha (transparency) based on pressure
            # Make low-pressure areas more transparent
            colors[..., 3] = norm_data * 0.3

            # 4. For performance, only draw voxels above a small threshold
            filled = norm_data > 0.1

            # 5. Plot the voxels
            ax.voxels(filled, facecolors=colors, edgecolor='none', shade=False)

            ax.set_title(f'Frame {frame_number}')
            ax.set_xlim(0, frame.shape[0])
            ax.set_ylim(0, frame.shape[1])
            ax.set_zlim(0, frame.shape[2])
            # Optional: Set a fixed camera angle
            ax.view_init(elev=30, azim=45)
            return fig,

        # --- Create and Save the Animation ---
        animation = FuncAnimation(fig, update, frames=self.number_of_frames,
                                  fargs=(self.history,), blit=False, interval=100)

        with tqdm(total=self.number_of_frames, desc="Saving Voxel Animation") as pbar:
            def progress_update(current_frame, total_frames):
                pbar.update(1)

            animation.save('simulation_3d_voxels.mp4', writer='ffmpeg', dpi=100,
                           progress_callback=progress_update)

        plt.close(fig)
        print("Voxel animation saved to simulation_3d_voxels.mp4")




if __name__ == '__main__':
    print('hello visualize')