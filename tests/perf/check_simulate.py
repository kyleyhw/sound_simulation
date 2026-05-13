"""Correctness gate for evolve candidates.

Runs the same 200x200 / 200-step sanity simulation as make_reference.py,
loads the baseline snapshot, and asserts:

  * max |p_cand - p_base| < ATOL                    (1e-5)
  * ||p_cand - p_base||_2 / ||p_base||_2 < RTOL     (1e-4)
  * Simulate.reset() actually zeros the field, time, and step counter
  * Constructing Simulate with an explicit timestep that violates the
    CFL bound 1/sqrt(d) emits a RuntimeWarning containing 'CFL'
  * The public attribute set on Simulate is preserved

Prints exactly one line:
    CHECK pass=true|false  max_abs=<float>  l2_rel=<float>  failure=<str|->
so the supervisor can grep the result.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.simulation.setup import Driver  # noqa: E402
from acoustic_system.simulation.simulate import Simulate  # noqa: E402
from acoustic_system.simulation.waveforms import RickerWavelet  # noqa: E402

ATOL = 1e-5
RTOL = 1e-4

REQUIRED_ATTRS = (
    "p",
    "p_prev",
    "time",
    "step_count",
    "grid_shape",
    "timestep",
    "wavespeed",
    "gridstep",
)


def build_sim() -> Simulate:
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


def fail(msg: str, max_abs: float = float("nan"), l2_rel: float = float("nan")) -> None:
    print(
        f"CHECK pass=false  max_abs={max_abs:.3e}  l2_rel={l2_rel:.3e}  failure={msg}"
    )
    sys.exit(1)


def main() -> None:
    ref_path = Path(__file__).parent / "reference.npz"
    if not ref_path.exists():
        fail(f"reference snapshot missing: {ref_path}")
    ref = np.load(ref_path)
    p_base = ref["p"].astype(np.float32, copy=False)

    # 1. API surface
    sim = build_sim()
    for attr in REQUIRED_ATTRS:
        if not hasattr(sim, attr):
            fail(f"missing public attribute Simulate.{attr}")

    # 2. CFL warning behaviour
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad = Simulate(
            grid_shape=(50, 50),
            drivers=[],
            wavespeed=1.0,
            timestep=10.0,
            gridstep=1.0,
        )
        cfl_warned = any("CFL" in str(w.message) for w in caught)
    if not cfl_warned:
        fail("CFL violation did not raise a RuntimeWarning containing 'CFL'")
    del bad

    # 3. reset() correctness
    sim_reset = build_sim()
    for _ in range(20):
        sim_reset.step()
    sim_reset.reset()
    if (
        float(np.max(np.abs(sim_reset.p))) != 0.0
        or float(np.max(np.abs(sim_reset.p_prev))) != 0.0
        or sim_reset.time != 0.0
        or sim_reset.step_count != 0
    ):
        fail("reset() did not zero p/p_prev/time/step_count")

    # 4. Element-wise field equivalence
    for _ in range(200):
        sim.step()
    p_cand = sim.p.astype(np.float32, copy=False)

    if p_cand.shape != p_base.shape:
        fail(f"shape mismatch: cand={p_cand.shape} base={p_base.shape}")

    diff = p_cand.astype(np.float64) - p_base.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    base_norm = float(np.linalg.norm(p_base.astype(np.float64))) or 1.0
    l2_rel = float(np.linalg.norm(diff) / base_norm)

    if not (max_abs < ATOL and l2_rel < RTOL):
        fail(
            f"divergence beyond tolerance (atol={ATOL:.0e}, rtol={RTOL:.0e})",
            max_abs=max_abs,
            l2_rel=l2_rel,
        )

    print(f"CHECK pass=true   max_abs={max_abs:.3e}  l2_rel={l2_rel:.3e}  failure=-")


if __name__ == "__main__":
    main()
