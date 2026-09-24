"""Batched multi-room throughput (plan 5.7.3).

Dataset generation simulates many small rooms. This compares three ways
of running B rooms of the same size for T steps:

* ``numba-seq``: one ``Simulate`` per room, one after another (the
  current ``dataset.py`` path);
* ``torch-batch``: one ``TorchFDTD`` launch with a leading batch axis;
* ``torch-batch`` on CUDA when available.

Rooms get random p = 0 obstacle masks, one source and four mics.
Reports room-steps per second (B * T / wall time), median of ``--trials``.

    uv run python tests/perf/bench_batched.py --grid 128 --steps 400 --batch 1 8 32
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

from acoustic_system.simulation.setup import Driver
from acoustic_system.simulation.simulate import Simulate
from acoustic_system.simulation.torch_engine import TorchFDTD, TorchGrid, sample_waveform
from acoustic_system.simulation.waveforms import RickerWavelet


def rooms(b: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    occ = np.zeros((b, n, n), dtype=np.float32)
    for k in range(b):
        for _ in range(6):
            i, j = rng.integers(8, n - 16, 2)
            occ[k, i : i + 8, j : j + 8] = 1.0
        occ[k, n // 2 - 4 : n // 2 + 4, n // 4 - 4 : n // 4 + 4] = 0.0  # keep the source free
    return occ


def numba_seq(occ: np.ndarray, steps: int) -> float:
    n = occ.shape[1]
    wf = RickerWavelet(1.0, 0.1, 15.0)
    t0 = time.perf_counter()
    for k in range(occ.shape[0]):
        sim = Simulate((n, n), drivers=[Driver((n // 2, n // 4), wf)])
        sim.set_obstacle([tuple(c) for c in np.argwhere(occ[k] > 0)])
        for _ in range(steps):
            sim.step()
    return time.perf_counter() - t0


def torch_batch(occ: np.ndarray, steps: int, device: str) -> float:
    b, n, _ = occ.shape
    grid = TorchGrid((n, n))
    src_pos = torch.tensor([[[n // 2, n // 4]]] * b)
    mic_pos = torch.tensor([[[n // 2 + d, 3 * n // 4] for d in (-6, -2, 2, 6)]] * b)
    eng = TorchFDTD(grid, src_pos, mic_pos, device=device)
    wf = sample_waveform(RickerWavelet(1.0, 0.1, 15.0), steps, grid.timestep)
    src = torch.tensor(np.broadcast_to(wf, (b, 1, steps)).copy(), device=device)
    o = torch.tensor(occ, device=device)
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.run(src, occ=o)
        if device == "cuda":
            torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="numba and torch thread count (0 = library defaults). Use 1 on a loaded "
        "machine: parallel barriers collapse when cores are oversubscribed.",
    )
    args = ap.parse_args()
    if args.threads > 0:
        import numba

        numba.set_num_threads(args.threads)
        torch.set_num_threads(args.threads)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    numba_seq(rooms(1, args.grid), 20)  # JIT warm-up
    torch_batch(rooms(1, args.grid), 20, "cpu")
    print(f"grid {args.grid}^2, {args.steps} steps, torch threads {torch.get_num_threads()}")
    print(
        f"{'B':>4} {'numba-seq':>14} "
        + " ".join(f"{'torch-' + d:>14}" for d in devices)
        + "   (room-steps/s)"
    )
    for b in args.batch:
        occ = rooms(b, args.grid)
        rs = b * args.steps
        row = [rs / statistics.median(numba_seq(occ, args.steps) for _ in range(args.trials))]
        for d in devices:
            row.append(
                rs / statistics.median(torch_batch(occ, args.steps, d) for _ in range(args.trials))
            )
        print(f"{b:>4} " + " ".join(f"{v:>14.0f}" for v in row))


if __name__ == "__main__":
    main()
