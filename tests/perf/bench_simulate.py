"""Wall-clock benchmark for evolve candidates.

Measures the median of ``--trials`` independent runs of ``--steps`` time
steps on a ``--grid``x``--grid`` 2D grid with a centred Ricker driver.
Prints a single grep-friendly line:

    BENCH median_ms=<float>  trials_ms=[t1, t2, ...]  steps=<int>  grid=<int>

The first trial absorbs any one-time JIT compile / page-fault cost;
``--trials >= 5`` makes the median robust against that and against OS
scheduler noise.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.simulation.setup import Driver  # noqa: E402
from acoustic_system.simulation.simulate import Simulate  # noqa: E402
from acoustic_system.simulation.waveforms import RickerWavelet  # noqa: E402


def build_sim(grid: int) -> Simulate:
    driver = Driver(
        position=(grid // 2, grid // 2),
        waveform=RickerWavelet(amplitude=5.0, frequency=0.1, delay=20.0),
    )
    return Simulate(
        grid_shape=(grid, grid),
        drivers=[driver],
        wavespeed=1.0,
        gridstep=1.0,
        courant=0.5,
    )


def time_one_run(grid: int, steps: int) -> float:
    sim = build_sim(grid)
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    times_s = [time_one_run(args.grid, args.steps) for _ in range(args.trials)]
    times_ms = [round(t * 1000, 3) for t in times_s]
    median_ms = round(statistics.median(times_ms), 3)
    print(
        f"BENCH median_ms={median_ms}  trials_ms={times_ms}  "
        f"steps={args.steps}  grid={args.grid}"
    )


if __name__ == "__main__":
    main()
