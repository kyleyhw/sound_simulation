# Documentation Index

This index provides an overview and links to the detailed documentation for each script in the project.

## Simulation Scripts

- [**`main.py`**](./main.md): The main entry point for running a predefined simulation.
- [**`simulate.py`**](./simulate.md): The main simulation driver, containing the time-stepping loop.
- [**`calculate.py`**](./calculate.md): Contains the core Laplacian calculation kernel.
- [**`setup.py`**](./setup.md): Defines and generates `Driver` and `Sensor` components.
- [**`waveforms.py`**](./waveforms.md): Defines the mathematical functions for sound sources.
- [**`boundary.py`**](./boundary.md): Defines a generic structure for boundary conditions.
- [**`interactive_setup.py`**](./interactive_setup.md): Provides a GUI for interactively designing a 2D simulation environment.
- [**`visualize.py`**](./visualize.md): Handles the creation of animations and plots from simulation results.
- [**`data_io.py`**](./data_io.md): Manages saving and loading simulation data to/from HDF5 files.
- [**`reconstruct.py`**](./reconstruct.md): An example script for loading and inspecting saved data.
- [**`utils.py`**](./utils.md): Contains miscellaneous helper functions.

## Inference Scripts

- [**`cnn1d.py`**](./cnn1d.md): Defines the 1D Convolutional Neural Network model.
- [**`datasetformat.py`**](./datasetformat.md): Defines the PyTorch `Dataset` for loading simulation data.
- [**`trainingtest.py`**](./trainingtest.md): The main script for training the CNN model.
- [**`quicktest.py`**](./quicktest.md): A utility script to generate dummy data for testing the training pipeline.
