import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.backend_bases import MouseEvent#, ScrollEvent

class InteractiveSetup:
    """
    An interactive UI for drawing obstacles and placing drivers on a 2D grid.
    """
    def __init__(self, gridsize, brush_size=1):
        self.gridsize = gridsize
        self.brush_size = brush_size

        self.obstacle_grid = np.zeros(gridsize, dtype=int)
        self.driver_locations = []

        # --- Setup the plot ---
        self.fig, self.ax = plt.subplots(figsize=(8, 8))  # Fixed size for better interaction

        # Define custom colormap: 0=Empty (light gray), 1=Obstacle (black), 2=Driver (blue)
        colors = ['#DDDDDD', 'black', '#4682B4']  # Light gray, Black, SteelBlue
        cmap = ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5]  # Bounds for 0, 1, 2 values
        norm = plt.Normalize(vmin=-0.5, vmax=2.5)  # Normalize to cover all values

        # Initialize the plot with the combined display grid
        self.display_grid = self._get_combined_display_grid()
        self.plot_object = self.ax.imshow(self.display_grid.T, cmap=cmap, norm=norm, origin='lower')

        self.ax.set_title(self._get_title())

        # Grid lines and ticks for clarity
        self.ax.set_xticks(np.arange(-.5, self.gridsize[0], 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.gridsize[1], 1), minor=True)
        self.ax.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        self.ax.set_xticks(np.arange(0, self.gridsize[0], 10))  # Major ticks every 10 units
        self.ax.set_yticks(np.arange(0, self.gridsize[1], 10))
        self.ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


        self.is_drawing_obstacles = False

        plt.show()

    def _get_title(self):
        """Helper to generate the title string."""
        return f"Brush Size: {self.brush_size} | Left-Click: Draw Boundary | Right-Click: Add Driver"

    def _connect_events(self):
        """Connects the Matplotlib event callbacks to the handler methods."""
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    # Add the 'MouseEvent' type hint to all event handlers
    def on_press(self, event: MouseEvent):
        """Callback for mouse button press. Handles left and right clicks."""
        if event.inaxes != self.ax: return

        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        location = (ix, iy)

        if event.button == 1: # Left mouse button
            self.is_drawing_obstacles = True
            self._draw_obstacle_at_point(ix, iy)
        elif event.button == 3: # Right mouse button
            if location not in self.driver_locations:
                self.driver_locations.append(location)
            self._update_display()

        print('press detected')

    def on_release(self, event: MouseEvent):
        """Callback for mouse button release."""
        if event.button == 1:
            self.is_drawing_obstacles = False

        print('release detected')

    def on_motion(self, event: MouseEvent):
        """Callback for mouse motion (for drawing obstacles)."""
        if self.is_drawing_obstacles and event.inaxes == self.ax:
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            self._draw_obstacle_at_point(ix, iy)

        print('motion detected')

    def on_scroll(self, event): # ScrollEvent type cannot be imported from matplotlib.backend_bases
        """Callback for mouse wheel scroll."""
        if event.inaxes != self.ax: return
        # Increase brush size when scrolling up, decrease when scrolling down
        if event.step > 0:
            self.brush_size += 1
        else:
            # Ensure brush size doesn't go below 0
            self.brush_size = max(0, self.brush_size - 1)

        # Update the title to give the user visual feedback
        self.ax.set_title(self._get_title())
        self.fig.canvas.draw_idle()
        print(f"Brush size set to: {self.brush_size}")

        print('scroll detected')

    def _draw_obstacle_at_point(self, ix, iy):
        """Helper to update the obstacle grid and then the display."""
        # Ensure coordinates are within grid boundaries
        if not (0 <= ix < self.gridsize[0] and 0 <= iy < self.gridsize[1]):
            return

        y, x = np.ogrid[-iy:self.gridsize[1] - iy, -ix:self.gridsize[0] - ix]
        mask = x * x + y * y <= self.brush_size * self.brush_size
        self.obstacle_grid[mask] = 1  # Mark as obstacle

        self._update_display()  # Refresh the combined display

    def _get_combined_display_grid(self):
        """Creates a grid combining obstacles and drivers for display."""
        combined_grid = self.obstacle_grid.copy()
        for x, y in self.driver_locations:
            if 0 <= x < self.gridsize[0] and 0 <= y < self.gridsize[1]:
                combined_grid[x, y] = 2  # Mark drivers
        return combined_grid

    def _update_display(self):
        """Updates the plot object with the current combined grid."""
        self.display_grid = self._get_combined_display_grid()
        self.plot_object.set_data(self.display_grid.T)
        self.fig.canvas.draw_idle()

    def setup(self) -> (np.ndarray, list):
        self._connect_events()
        self._update_display()  # Initial display update
        plt.show()
        return self.obstacle_grid, self.driver_locations


if __name__ == '__main__':
    drawer = InteractiveSetup(gridsize=(16,16), brush_size=1)