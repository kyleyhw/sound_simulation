# Documentation Index

This index provides an overview and links to the detailed documentation for each component of the project.

## Project plan

- [**`plan_audit.md`**](./plan_audit.md): The 2026-09-24 audit of `PROJECT_PLAN.md`. It covers the no-audio IoU baseline (`scripts/eval_no_audio_baseline.py`), the boundary-physics and units findings, plan-structure gaps, and hygiene.

## Web UI

- [**`web_app.md`**](./web_app.md): The browser-native Acoustic Sandbox (`web/`), hosted on GitHub Pages. It covers the UX spec, visual design, architecture, engine paths and tests.
- [**`lab.md`**](./lab.md): Phase 9 real-hardware tooling. It covers the browser Lab (sweep IRs, device calibration, echoes, T60, room twin, crosstalk cancellation, head tracking), `scripts/eval_real_captures.py`, and sim-to-real randomisation.


## Machine learning (Phase 2)

- [**`learning.md`**](./learning.md): The active-sensing pipeline — dataset generation (`scripts/generate_active_sensing.py`), the `DualInputCNN` obstacle-mask model, BCE+Dice loss, training/eval loops, and the results to date.
- [**`demos.md`**](./demos.md): Phase 1+2 demonstrations — the standalone room-mapping figure (`scripts/demo_room_mapping.py`) built on `learning/sensing.py`.

## Sound-field control (Phase 7)

- [**`control.md`**](./control.md): Transfer functions from the engine, sound zones (delay-and-sum, pressure matching, ACC, time reversal, broadband FIR), crosstalk cancellation, FxLMS noise cancellation, sensing requirements, and differentiable control. Results are in `tests/reports/control_2026_09_24.md`.

## Physics imaging (Phase 6)

- [**`benchmark.md`**](./benchmark.md): The room-sensing benchmark. It covers the data (regenerated from the manifest), the rules, scoring against the no-audio baseline (`scripts/benchmark.py`), and the leaderboard.

- [**`imaging.md`**](./imaging.md): The no-ML imagers: deconvolution, echo ellipses and free-space carving, delay-and-sum back-projection, time reversal, and FWI. It also covers the observable targets, the Cramér–Rao design chart, and T60/DRR/absorption estimation. Results are in `tests/reports/imaging_2026_09_24.md`.

## Simulation engine

- [**`main.py`**](./main.md): The standalone batch-mode example (`--save` for headless runs).
- [**`physics.md`**](./physics.md): Physics model (walls, materials, absorbing boundaries including the CPML, c(x), units, sources), and the verification report: modes, convergence, energy, dispersion, Green's functions, edge reflection, T60.
- [**`simulate.py`**](./simulate.md): The step-at-a-time FDTD engine (`Simulate` class).
- [**`calculate.py`**](./calculate.md): The discrete Laplacian kernel.
- [**`calculate_gpu.py`**](./gpu.md): The CUDA (CuPy) backend — GPU twins of the fused kernels, transfer strategy, gates and benchmarks.
- [**`setup.py`**](./setup.md): `Driver` and `Sensor` definitions.
- [**`waveforms.py`**](./waveforms.md): Source waveforms (Cosine, GaussianPulse, RickerWavelet).
- [**`visualize.py`**](./visualize.md): Plotting and animation of saved runs.
- [**`data_io.md`**](./data_io.md): HDF5 read/write of simulation results.
- [**`utils.py`**](./utils.md): Edge-index helpers.
