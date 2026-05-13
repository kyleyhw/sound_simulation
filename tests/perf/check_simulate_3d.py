"""3D correctness gate for the fused leap-frog kernel.

Runs the same N steps with two implementations and checks the final
pressure fields agree within a tight tolerance:

  * **Candidate**: ``Simulate(grid_shape=(N, N, N))`` going through the
    fused ``fused_leapfrog_step_3d`` numba kernel.
  * **Reference**: explicit Python loop applying ``scipy.ndimage.laplace``
    + the leap-frog combine + ``set_edge_values``, the same path the
    engine used before the 3D fused kernel landed.

The reference is computed inline rather than stored as an .npz: a 100³
float32 snapshot is 4 MB, which is excessive to commit. The two
implementations agree to within float32 round-off because they use the
same 7-point stencil; the only meaningful difference is fastmath FMA
fusion in the njit kernel, which contributes ~1e-7 max-abs error.

Prints exactly one line so the supervisor can grep:

    CHECK_3D pass=true|false  max_abs=<float>  l2_rel=<float>  failure=<str|->
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.simulation.setup import Driver  # noqa: E402
from acoustic_system.simulation.simulate import Simulate  # noqa: E402
from acoustic_system.simulation.utils import set_edge_values  # noqa: E402
from acoustic_system.simulation.waveforms import RickerWavelet  # noqa: E402

# fastmath in the 3D njit kernel introduces ~1e-7 abs / ~1e-6 rel error
# vs the scipy combine. Generous-but-tight tolerance.
ATOL = 1e-4
RTOL = 1e-3

GRID = 40
STEPS = 100


def fail(msg: str, max_abs: float = float("nan"), l2_rel: float = float("nan")) -> None:
    print(f"CHECK_3D pass=false  max_abs={max_abs:.3e}  l2_rel={l2_rel:.3e}  failure={msg}")
    sys.exit(1)


def run_candidate() -> np.ndarray:
    """Fused 3D kernel via Simulate.step()."""
    sim = Simulate(
        grid_shape=(GRID, GRID, GRID),
        drivers=[
            Driver(
                position=(GRID // 2, GRID // 2, GRID // 2),
                waveform=RickerWavelet(amplitude=5.0, frequency=0.1, delay=20.0),
            )
        ],
        wavespeed=1.0,
        gridstep=1.0,
        courant=0.5,
    )
    if sim._kernel is None or sim._kernel.__name__ != "fused_leapfrog_step_3d":
        fail(f"3D fused kernel not bound: kernel={sim._kernel}")
    for _ in range(STEPS):
        sim.step()
    return sim.p.astype(np.float32, copy=False)


def run_reference() -> np.ndarray:
    """Plain scipy.ndimage.laplace + leap-frog, same parameters."""
    grid_shape = (GRID, GRID, GRID)
    timestep = (
        0.5 * 1.0 / 1.0 / np.sqrt(3) * np.sqrt(3) * 0.5
    )  # match Simulate's chosen_courant=0.5 -> dt=0.5*dx/c=0.5
    # Reproduce Simulate's auto-chosen timestep exactly: chosen_courant = min(0.5, 0.95/sqrt(3)) = 0.5
    timestep = 0.5 * 1.0 / 1.0
    coeff = (1.0 * timestep / 1.0) ** 2  # (c dt / dx) ** 2
    p_prev = np.zeros(grid_shape, dtype=np.float32)
    p = np.zeros(grid_shape, dtype=np.float32)
    driver_pos = (GRID // 2, GRID // 2, GRID // 2)
    waveform = RickerWavelet(amplitude=5.0, frequency=0.1, delay=20.0)
    time = 0.0
    for _ in range(STEPS):
        # 7-point Laplacian via scipy. ndimage.laplace returns the
        # discrete Laplacian; dividing by gridstep**2 (=1) is a no-op
        # so omit it for speed.
        laplacian = sp.ndimage.laplace(p)
        p_next = (2.0 * p - p_prev + coeff * laplacian).astype(np.float32)
        # Hard-wall (Dirichlet, p=0) on all six faces.
        set_edge_values(arr=p_next, value=0)
        # Driver injection AFTER boundary zeroing — matches the engine's
        # ordering (a driver on a boundary overwrites the wall).
        p_next[driver_pos] += float(waveform(time))
        # Buffer rotation.
        p_prev = p
        p = p_next
        time = time + timestep
    return p


def main() -> None:
    p_cand = run_candidate()
    p_ref = run_reference()

    if p_cand.shape != p_ref.shape:
        fail(f"shape mismatch: cand={p_cand.shape} ref={p_ref.shape}")

    diff = p_cand.astype(np.float64) - p_ref.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    base_norm = float(np.linalg.norm(p_ref.astype(np.float64))) or 1.0
    l2_rel = float(np.linalg.norm(diff) / base_norm)

    if not (max_abs < ATOL and l2_rel < RTOL):
        fail(
            f"divergence beyond tolerance (atol={ATOL:.0e}, rtol={RTOL:.0e})",
            max_abs=max_abs,
            l2_rel=l2_rel,
        )

    print(f"CHECK_3D pass=true   max_abs={max_abs:.3e}  l2_rel={l2_rel:.3e}  failure=-")


if __name__ == "__main__":
    main()
