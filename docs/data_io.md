# Documentation for `data_io.py`

## 1. Purpose

`data_io.py` provides a robust mechanism for saving and loading simulation results to and from disk. This is essential for creating persistent datasets for machine learning, for archiving results, and for separating the computationally expensive simulation process from the analysis and visualization stages.

## 2. Key Technologies

-   **HDF5 (Hierarchical Data Format)**: The script uses the `h5py` library to interact with HDF5 files.
    -   **Rationale**: HDF5 is a high-performance data storage format designed specifically for scientific and numerical data. It was chosen for several key reasons:
        1.  **Hierarchical Structure**: It allows data to be organized in a group/dataset structure, much like folders and files in a filesystem. This is used to neatly store different simulations (`simulation_0001`, `simulation_0002`) and their components (drivers, sensors, history) within a single archive file.
        2.  **Metadata Support**: HDF5 allows attributes (small key-value metadata) to be attached to groups and datasets. This is used to store the simulation parameters (grid size, timestep, etc.) directly alongside the data itself.
        3.  **Performance & Compression**: It is highly efficient for I/O operations on large numerical arrays and supports on-the-fly compression (like `gzip`) to reduce file sizes, which is critical when storing the full simulation history.

## 3. Implementation Details

### Class: `SaveSimulationResults`

-   **Purpose**: To handle the saving of simulation data into an HDF5 file.
-   **`_saver_registry`**: This dictionary maps a `save_type` string (e.g., `'full_history'`) to the specific method responsible for that save operation. This is a flexible design that allows new save types to be added easily.
-   **`_save_params(...)`**: Saves the simulation parameters as attributes of the main simulation group.
-   **`_save_drivers(...)`**: Creates a `drivers` group and saves the parameters of each `Driver`'s waveform as attributes.
-   **`_save_sensors(...)`**: Saves the sensor locations and their full time-series data.
-   **`_save_full_history(...)`**: Saves the parameters, drivers, and the entire simulation `history` array.
-   **`_save_sensor_results(...)`**: A more lightweight save option that saves the parameters, drivers, and only the final sensor time-series data, omitting the full (and very large) history array.

### Class: `LoadSimulationResults`

-   **Purpose**: To read the raw data from an HDF5 file without immediately reconstructing it into the original Python objects (`Simulate`, `Driver`, etc.).
-   **`_loader_registry`**: Similar to the saver, this maps the `save_type` (read from the file's attributes) to the appropriate loading method.
-   **`load_raw_results(...)`**: This is the main public method. It reads the data for a given `simulation_id` and returns it as a nested dictionary. This raw data can then be used for analysis, visualization, or as input to a machine learning pipeline.
