"""Wall-clock benchmark for the 3D fused leap-frog kernel.

Sweeps a list of cubic grid sizes and prints one line per size:

    BENCH_3D grid=N median_ms=<float>  trials_ms=[t1, t2, ...]  steps=<int>

Run with `--grids 32 64 100 128 200` (or any subset). Defaults to a sweep
that maps the visualisation candidates discussed in the project plan:
$32^3$, $64^3$, $100^3$, $128^3$, plus an optional $200^3$ that the user
can opt into with `--include-large` (it is excluded by default because it
takes ~1 s/step on consumer CPUs and would dominate the benchmark wall
clock).

The first trial of every grid size absorbs any one-time JIT compile cost
(the @njit kernel is cached across runs by ``cache=True`` in calculate.py
so subsequent invocations of this script are fast). ``--trials`` >= 5
makes the median robust against that warm-up and against scheduler noise.
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

DEFAULT_GRIDS = [32, 64, 100, 128]
LARGE_GRIDS = [200]


def build_sim(grid: int) -> Simulate:
    return Simulate(
        grid_shape=(grid, grid, grid),
        drivers=[
            Driver(
                position=(grid // 2, grid // 2, grid // 2),
                waveform=RickerWavelet(amplitude=5.0, frequency=0.1, delay=20.0),
            )
        ],
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
    parser.add_argument(
        "--grids",
        nargs="+",
        type=int,
        default=None,
        help="Cubic side lengths to benchmark. Default: 32 64 100 128.",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Also include the slow 200^3 case (~1 s/step on consumer CPUs).",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    grids = list(args.grids) if args.grids is not None else list(DEFAULT_GRIDS)
    if args.include_large and 200 not in grids:
        grids.extend(LARGE_GRIDS)

    for grid in grids:
        times_s = [time_one_run(grid, args.steps) for _ in range(args.trials)]
        times_ms = [round(t * 1000.0, 3) for t in times_s]
        median_ms = round(statistics.median(times_ms), 3)
        per_step_ms = round(median_ms / args.steps, 3)
        print(
            f"BENCH_3D grid={grid} median_ms={median_ms} per_step_ms={per_step_ms} "
            f"trials_ms={times_ms}  steps={args.steps}"
        )


if __name__ == "__main__":
    main()
