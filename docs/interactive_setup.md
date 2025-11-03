# Documentation for `interactive_setup.py`

## 1. Purpose

`interactive_setup.py` provides a graphical user interface (GUI) for interactively designing a 2D simulation environment. It allows the user to visually "paint" obstacles and place sound sources (drivers) on the grid before running a simulation.

## 2. Key Technologies

-   **Matplotlib (Interactive Backend)**: The UI is built entirely using Matplotlib. By using an interactive backend (`matplotlib.use('WebAgg')` or a native GUI backend like `TkAgg`, `Qt5Agg`), Matplotlib can capture mouse clicks, motion, and scroll events, turning a static plot into an interactive canvas.

## 3. Implementation Details

### Class: `InteractiveSetup`

-   **`__init__(...)`**: The constructor sets up the Matplotlib figure and axes. It initializes the `obstacle_grid` and `driver_locations` arrays, which will be modified by user interaction. It also defines a custom colormap to visually distinguish between empty space, obstacles, and drivers.

-   **`_connect_events()`**: This crucial method connects the different Matplotlib event signals (e.g., `'button_press_event'`) to the corresponding handler methods in the class (`on_press`).

### Event Handlers

-   **`on_press(self, event)`**: This method is called when a mouse button is pressed.
    -   **Left-Click**: It sets a flag `is_drawing_obstacles = True` and calls the drawing helper function. This begins the "painting" action.
    -   **Right-Click**: It adds the current mouse coordinates to the `driver_locations` list.

-   **`on_release(self, event)`**: When the left mouse button is released, it sets `is_drawing_obstacles = False` to stop the drawing action.

-   **`on_motion(self, event)`**: If the user is moving the mouse while the left button is held down (`is_drawing_obstacles` is `True`), this method repeatedly calls the drawing helper function, allowing the user to draw continuous lines of obstacles.

-   **`on_scroll(self, event)`**: This method adjusts the `brush_size` attribute based on the scroll direction, allowing the user to change the size of the circular brush used for painting obstacles.

### Helper Methods

-   **`_draw_obstacle_at_point(...)`**: This is the core drawing logic. It creates a circular mask around the current mouse position with a radius equal to `brush_size` and sets the corresponding elements in the `obstacle_grid` to `1`.

-   **`_get_combined_display_grid()`**: This method creates a temporary grid for visualization purposes by taking the `obstacle_grid` and marking the driver locations on top of it.

-   **`_update_display()`**: This method is called after any user interaction that changes the grid. It updates the data in the `imshow` plot object and redraws the canvas to provide immediate visual feedback.

-   **`setup()`**: This is the main public method that starts the interactive session by connecting the events and showing the plot. It returns the final `obstacle_grid` and `driver_locations` after the user is done and closes the plot window.
