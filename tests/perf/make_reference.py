"""Produce reference.npz: the baseline final pressure field after 200 steps
on a 200x200 grid with one centred Ricker wavelet driver. Candidate
implementations are compared against this snapshot in check_simulate.py.

Run this exactly once on the unmodified baseline (i.e. the protected
branch). It is intentionally not committed by the candidate explorers:
the reference is the truth they must reproduce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.simulation.setup import Driver  # noqa: E402
from acoustic_system.simulation.simulate import Simulate  # noqa: E402
from acoustic_system.simulation.waveforms import RickerWavelet  # noqa: E402


def build_correctness_sim() -> Simulate:
    driver = Driver(
        position=(100, 100),
        waveform=RickerWavelet(amplitude=5.0, frequency=0.1, delay=20.0),
    )
    return Simulate(
        grid_shape=(200, 200),
        drivers=[driver],
        wavespeed=1.0,
        gridstep=1.0,
        courant=0.5,
    )


def main() -> None:
    sim = build_correctness_sim()
    for _ in range(200):
        sim.step()

    out = Path(__file__).parent / "reference.npz"
    np.savez_compressed(
        out,
        p=sim.p.astype(np.float32, copy=False),
        p_prev=sim.p_prev.astype(np.float32, copy=False),
        time=np.float64(sim.time),
        step_count=np.int64(sim.step_count),
        timestep=np.float64(sim.timestep),
    )
    print(f"reference written: {out}")
    print(f"  max(|p|) = {float(np.max(np.abs(sim.p))):.6e}")
    print(f"  norm(p)  = {float(np.linalg.norm(sim.p)):.6e}")
    print(f"  step={sim.step_count} t={sim.time:.4f}")


if __name__ == "__main__":
    main()
