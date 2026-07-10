"""Correctness gate for the CuPy GPU backend (Task 1.5).

Runs identical scenarios on ``Simulate(backend="cpu")`` and
``Simulate(backend="gpu")`` and requires the final pressure fields to
agree within float32 round-off tolerance. The CPU backend is itself
gated against ``reference.npz`` (2D) and the scipy reference (3D), so
agreement here chains the GPU backend to the same ground truth.

Scenarios are chosen to exercise every ordering contract the engine
documents:

* an interior Ricker driver (the standard excitation),
* a driver ON the boundary wall (must overwrite the Dirichlet zero),
* an interior rectangular obstacle (masked scrub between stencil and
  driver injection), uploaded via the bulk ``set_obstacle_mask`` path
  on GPU and per-cell ``set_obstacle`` on CPU — so the two upload APIs
  are checked for equivalence at the same time,
* ``reset()`` mid-life, after which a re-run must reproduce the run
  from a fresh object.

Tolerances match the 3D inline gate (atol=1e-4, rtol=1e-3): the only
expected difference is FMA/rounding order between numba-fastmath x86
and NVRTC-default CUDA, ~1e-6 relative L2 in practice.

Prints one grep-able line per scenario:

    CHECK_GPU_<name> pass=true|false  max_abs=<float>  l2_rel=<float>  failure=<str|->
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.simulation import calculate_gpu  # noqa: E402
from acoustic_system.simulation.setup import Driver  # noqa: E402
from acoustic_system.simulation.simulate import Simulate  # noqa: E402
from acoustic_system.simulation.waveforms import RickerWavelet  # noqa: E402

ATOL = 1e-4
RTOL = 1e-3


def fail(name: str, msg: str, max_abs: float = float("nan"), l2_rel: float = float("nan")) -> None:
    print(f"CHECK_GPU_{name} pass=false  max_abs={max_abs:.3e}  l2_rel={l2_rel:.3e}  failure={msg}")
    sys.exit(1)


def ok(name: str, max_abs: float = 0.0, l2_rel: float = 0.0) -> None:
    print(f"CHECK_GPU_{name} pass=true   max_abs={max_abs:.3e}  l2_rel={l2_rel:.3e}  failure=-")


def compare(name: str, a: np.ndarray, b: np.ndarray) -> None:
    """Assert two float32 fields agree within (ATOL, RTOL); print one line."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    base = float(np.linalg.norm(b.astype(np.float64))) or 1.0
    l2_rel = float(np.linalg.norm(diff) / base)
    if not (max_abs < ATOL and l2_rel < RTOL):
        fail(name, f"divergence (atol={ATOL:.0e}, rtol={RTOL:.0e})", max_abs, l2_rel)
    ok(name, max_abs, l2_rel)


def make_sim(backend: str, grid_shape: Tuple[int, ...]) -> Simulate:
    """Identical scenario on either backend: interior + wall drivers."""
    centre = tuple(s // 2 for s in grid_shape)
    wall = (0,) + tuple(s // 3 for s in grid_shape[1:])  # on the i=0 face
    return Simulate(
        grid_shape=grid_shape,
        drivers=[
            Driver(
                position=centre, waveform=RickerWavelet(amplitude=5.0, frequency=0.1, delay=20.0)
            ),
            Driver(
                position=wall, waveform=RickerWavelet(amplitude=3.0, frequency=0.08, delay=30.0)
            ),
        ],
        wavespeed=1.0,
        gridstep=1.0,
        courant=0.5,
        backend=backend,
    )


def obstacle_mask(grid_shape: Tuple[int, ...]) -> np.ndarray:
    """A rectangular block in the upper-left interior quadrant."""
    mask = np.zeros(grid_shape, dtype=bool)
    slices = tuple(slice(s // 8, s // 4) for s in grid_shape)
    mask[slices] = True
    return mask


def run_pair(name: str, grid_shape: Tuple[int, ...], steps: int) -> None:
    """CPU vs GPU on the same scenario, including obstacles and reset()."""
    cpu = make_sim("cpu", grid_shape)
    gpu = make_sim("gpu", grid_shape)

    mask = obstacle_mask(grid_shape)
    # CPU takes the per-cell API, GPU the bulk-upload API: equivalence of
    # the two obstacle paths is part of what this gate certifies.
    cpu.set_obstacle(map(tuple, np.argwhere(mask)))
    gpu.set_obstacle_mask(mask)

    for _ in range(steps):
        cpu.step()
        gpu.step()
    compare(name, gpu.p_host(), cpu.p)

    # reset() must zero device state exactly like host state: a re-run
    # after reset must reproduce the original run.
    gpu.reset()
    if float(abs(gpu.p_host()).max()) != 0.0 or gpu.step_count != 0:
        fail(f"{name}_RESET", "reset() left nonzero field or step_count")
    for _ in range(steps):
        gpu.step()
    compare(f"{name}_RESET", gpu.p_host(), cpu.p)


def main() -> None:
    if not calculate_gpu.gpu_available():
        fail("ENV", "no usable CUDA device (install --extra gpu, check nvidia-smi)")

    # p_host() type contract.
    probe = Simulate(grid_shape=(32, 32), backend="gpu")
    if not isinstance(probe.p_host(), np.ndarray):
        fail("HOST", f"p_host() returned {type(probe.p_host())}, expected numpy.ndarray")
    ok("HOST")

    # 1D must be rejected.
    try:
        Simulate(grid_shape=(64,), backend="gpu")
        fail("GUARD", "1D + backend='gpu' did not raise")
    except ValueError:
        ok("GUARD")

    run_pair("2D", (128, 128), steps=300)
    run_pair("3D", (40, 40, 40), steps=100)


if __name__ == "__main__":
    main()
