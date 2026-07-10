"""CPU-vs-GPU wall-clock benchmark for the FDTD backends (Task 1.5).

For each requested grid, times ``--steps`` steps of the same
centred-Ricker scenario on ``backend="cpu"`` and ``backend="gpu"`` and
reports the median of ``--trials`` runs plus the speedup. 2D grids are
given as a single side length (``--grids-2d 512 1024``), 3D likewise
(``--grids-3d 64 128``).

GPU timing methodology: ``cupy.cuda.Device().synchronize()`` is called
before starting and before stopping each trial's clock — kernel
launches are asynchronous, so without the sync the loop would only
measure enqueue time, not execution. The first trial absorbs NVRTC
compilation (~100 ms) and the numba JIT/page-fault cost on the CPU
side; the median over >= 5 trials is robust to both.

Prints one grep-able line per (dims, grid):

    BENCH_GPU dims=<2|3> grid=<int> cpu_ms=<float> gpu_ms=<float> speedup=<float> steps=<int>
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.simulation import calculate_gpu  # noqa: E402
from acoustic_system.simulation.setup import Driver  # noqa: E402
from acoustic_system.simulation.simulate import Simulate  # noqa: E402
from acoustic_system.simulation.waveforms import RickerWavelet  # noqa: E402


def build_sim(grid_shape: Tuple[int, ...], backend: str) -> Simulate:
    driver = Driver(
        position=tuple(s // 2 for s in grid_shape),
        waveform=RickerWavelet(amplitude=5.0, frequency=0.1, delay=20.0),
    )
    return Simulate(
        grid_shape=grid_shape,
        drivers=[driver],
        wavespeed=1.0,
        gridstep=1.0,
        courant=0.5,
        backend=backend,
    )


def time_one_run(grid_shape: Tuple[int, ...], steps: int, backend: str) -> float:
    sim = build_sim(grid_shape, backend)
    sync = None
    if backend == "gpu":
        sync = calculate_gpu.cp.cuda.Device().synchronize
        sim.step()  # absorb NVRTC compile outside the timed region
        sim.reset()
        sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    if sync is not None:
        sync()
    return time.perf_counter() - t0


def bench_shape(grid_shape: Tuple[int, ...], steps: int, trials: int) -> None:
    cpu_ms = statistics.median(
        [time_one_run(grid_shape, steps, "cpu") * 1000 for _ in range(trials)]
    )
    gpu_ms = statistics.median(
        [time_one_run(grid_shape, steps, "gpu") * 1000 for _ in range(trials)]
    )
    print(
        f"BENCH_GPU dims={len(grid_shape)} grid={grid_shape[0]} "
        f"cpu_ms={cpu_ms:.1f} gpu_ms={gpu_ms:.1f} "
        f"speedup={cpu_ms / gpu_ms:.2f} steps={steps}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids-2d", type=int, nargs="*", default=[256, 512, 1024, 2048])
    parser.add_argument("--grids-3d", type=int, nargs="*", default=[64, 128, 200])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    if not calculate_gpu.gpu_available():
        print("BENCH_GPU error=no-cuda-device")
        sys.exit(1)

    for g in args.grids_2d:
        bench_shape((g, g), args.steps, args.trials)
    for g in args.grids_3d:
        bench_shape((g, g, g), args.steps, args.trials)


if __name__ == "__main__":
    main()
