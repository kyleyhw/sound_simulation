# Documentation Index

This index provides an overview and links to the detailed documentation for each component of the project.

## Web UI

- [**`web_ui.md`**](./web_ui.md): Architecture, wire protocol, and operation of the interactive browser-based simulation UI (the FastAPI/Socket.IO backend in `src/acoustic_system/app/` and the React frontend in `frontend/`).

## Machine learning (Phase 2)

- [**`learning.md`**](./learning.md): The active-sensing pipeline — dataset generation (`scripts/generate_active_sensing.py`), the `DualInputCNN` obstacle-mask model, BCE+Dice loss, training/eval loops, and the results to date.

## Simulation engine

- [**`main.py`**](./main.md): The standalone batch-mode entry point.
- [**`simulate.py`**](./simulate.md): The step-at-a-time FDTD engine (`Simulate` class).
- [**`calculate.py`**](./calculate.md): The discrete Laplacian kernel.
- [**`calculate_gpu.py`**](./gpu.md): The CUDA (CuPy) backend — GPU twins of the fused kernels, transfer strategy, gates and benchmarks.
- [**`setup.py`**](./setup.md): `Driver` and `Sensor` definitions.
- [**`waveforms.py`**](./waveforms.md): Source waveforms (Cosine, GaussianPulse, RickerWavelet).
- [**`boundary.py`**](./boundary.md): Boundary-condition scaffold.
- [**`interactive_setup.py`**](./interactive_setup.md): Matplotlib-based 2D scene editor (legacy; superseded for live use by the web UI).
- [**`visualize.py`**](./visualize.md): Plotting and animation of saved runs.
- [**`data_io.md`**](./data_io.md): HDF5 read/write of simulation results.
- [**`reconstruct.py`**](./reconstruct.md): Loading saved data for inspection.
- [**`utils.py`**](./utils.md): Edge-index helpers and random location generation.
