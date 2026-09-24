# `main.py` — standalone batch simulation

## Purpose

`src/acoustic_system/simulation/main.py` is a small end-to-end example:
it builds a 2D `Simulate`, adds two sources and three sensors, runs a
fixed number of steps, and plots the sensor timeseries, their spectra,
and an animation of the field.

```bash
cd src
uv run python -m acoustic_system.simulation.main            # interactive windows
uv run python -m acoustic_system.simulation.main --save     # headless: ./plots/*.png
uv run python -m acoustic_system.simulation.main --grid 128 --steps 300
```

## Walkthrough

1. `Simulate(grid_shape=(n, n), courant=0.5)`. The timestep comes from
   the Courant number: $\Delta t = 0.5\,\Delta x / c$.
2. Two drivers are added through `add_driver` (the `drivers` attribute is
   read-only; see `simulate.md`):
   - a Ricker pulse, dominant frequency $f = 0.05$, i.e. a 20-cell
     wavelength
   - a continuous tone at $f = 0.03$
3. Three `Sensor`s are placed on the diagonal. The script records the
   full field history and then slices each sensor's timeseries out of
   it.
4. `Visualize` draws the sensor timeseries and FFT, then the 2D
   animation. `--save` selects matplotlib's non-interactive `Agg`
   backend and writes PNGs instead.

## Why these source parameters

Both sources stay well inside the grid's limits. The tone has
$f\,\Delta t = 0.015$ cycles per step, far below Nyquist (0.5) and the
10-samples-per-period guideline (0.1). Its wavelength of 33 cells is
far above the ~10-cell minimum for low numerical dispersion.

An earlier version of this script used `Cosine(frequency=5)` at
$\Delta t = 0.1$. That gives exactly 0.5 cycles per step: a $\pm 1$
alternating checkerboard rather than a wave (see `waveforms.md`).
