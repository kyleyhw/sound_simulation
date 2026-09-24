"""Standalone batch simulation: two sources, three sensors, three figures.

Run from ``src/`` with ``python -m acoustic_system.simulation.main``.
Pass ``--save`` to write the figures to ./plots instead of opening
windows (headless machines have no interactive matplotlib backend).

Source choice. Both sources are sampled well inside the FDTD limits: the
Ricker pulse has its dominant frequency at f = 0.05 (wavelength 20 cells)
and the tone at f = 0.03 gives f * dt = 0.015 samples per step, far below
the Nyquist bound 0.5 and the 10-samples-per-period rule of thumb. (An
earlier version drove Cosine(frequency=5) at dt = 0.1, i.e. f * dt = 0.5
exactly: an alternating +-1 checkerboard, not a wave.)
"""

from __future__ import annotations

import argparse

import numpy as np

from .setup import Driver, Sensor
from .simulate import Simulate
from .waveforms import Cosine, RickerWavelet


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--save", action="store_true", help="write figures to ./plots (headless)")
    args = ap.parse_args(argv)

    n = args.grid
    simulation = Simulate(grid_shape=(n, n), wavespeed=1.0, gridstep=1.0, courant=0.5)
    simulation.add_driver(
        Driver(position=(n // 4, n // 4), waveform=RickerWavelet(amplitude=5.0, frequency=0.05))
    )
    simulation.add_driver(
        Driver(position=(3 * n // 4, 3 * n // 4), waveform=Cosine(frequency=0.03, amplitude=0.5))
    )
    sensors = [Sensor(position=(n // 4 * (i + 1), n // 4 * (i + 1))) for i in range(3)]
    simulation.sensors.extend(sensors)

    history = np.zeros((args.steps, n, n), dtype=np.float32)
    for i in range(args.steps):
        simulation.step()
        history[i] = simulation.p
        if i % 100 == 0:
            print(f"Step {i}/{args.steps}")
    print("Simulation finished.")

    for sensor in sensors:
        sensor.timeseries = history[(slice(None),) + tuple(sensor.position)].copy()
        sensor.sample_rate = 1.0 / simulation.timestep

    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    from .visualize import Visualize

    params = {
        "grid_shape": (n, n),
        "wavespeed": simulation.wavespeed,
        "timestep": simulation.timestep,
    }
    vis = Visualize(history=history, params=params)
    show = not args.save
    vis.plot_sensor_timeseries(sensors=sensors, show=show, save=args.save)
    vis.plot_sensor_fft(sensors=sensors, show=show, save=args.save)
    if show:
        vis.plot2D(show=True, save=False)


if __name__ == "__main__":
    main()
