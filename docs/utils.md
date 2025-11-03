# Documentation for `utils.py`

## 1. Purpose

`utils.py` contains miscellaneous helper functions and classes that provide common, reusable functionality across the simulation project. This helps to keep the other scripts clean and focused on their primary purpose.

## 2. Implementation Details

### Edge-Handling Functions

These functions are designed to work on N-dimensional NumPy arrays, making them highly flexible.

-   **`get_edge_indices(arr: np.ndarray) -> np.ndarray`**
    -   **Purpose**: To find the coordinates of all elements that lie on any edge of an N-dimensional array.
    -   **How it Works**: It creates a boolean mask of the same shape as the input array. It then iterates through each dimension (`axis`) and sets the elements at the start (`index 0`) and end (`index -1`) of that dimension to `True`. Finally, it uses `np.where` to convert this boolean mask into a list of coordinates.

-   **`get_edge_values(arr: np.ndarray) -> np.ndarray`**
    -   **Purpose**: To retrieve the values of all elements on the edges of an array.
    -   **How it Works**: It first calls `get_edge_indices()` to get the locations of the edge elements. It then uses NumPy's advanced indexing to extract all the values at these locations in a single, efficient operation.

-   **`set_edge_values(arr: np.ndarray, value) -> np.ndarray`**
    -   **Purpose**: To set all elements on the edges of an array to a specific value.
    -   **How it Works**: This function is used in `simulate.py` to apply a simple Dirichlet boundary condition where pressure at the boundary is zero. Like `get_edge_values`, it uses `get_edge_indices()` to find the target elements and then assigns the given `value` to all of them at once.

### Class: `LocationGenerator`

-   **Purpose**: To generate random, valid coordinates within the simulation grid.
-   **Rationale**: This class is used by the `GenerateSensor` and `GenerateDriver` classes in `setup.py`. By centralizing the logic for random location generation, we ensure that all components are created within the valid bounds of the grid. The generator is initialized with the grid dimensions and specifically generates locations that are not on the very edge (from `1` to `dim - 1`), which is a sensible default for placing sources and sensors.
-   **`get_new_location(self) -> tuple`**: This method uses `np.random.randint` to generate a new random integer coordinate for each dimension, ensuring it falls within the `low` and `high` bounds defined in the constructor.
