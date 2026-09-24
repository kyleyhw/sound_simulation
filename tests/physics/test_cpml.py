"""Convolutional PML (plan 5.4.2)."""

from __future__ import annotations

import numpy as np

from acoustic_system.simulation.setup import Driver
from acoustic_system.simulation.simulate import Simulate
from acoustic_system.simulation.waveforms import RickerWavelet


def _pulse() -> RickerWavelet:
    return RickerWavelet(1.0, 0.1, 15.0)


def _trace(n: int, boundary: str, dims: int = 2, steps: int = 420, r: int = 50) -> np.ndarray:
    c = n // 2
    src = (c,) * dims
    probe = (c,) * (dims - 1) + (c + r,)
    sim = Simulate((n,) * dims, boundary=boundary, drivers=[Driver(src, _pulse())], cpml_cells=16)
    out = []
    for _ in range(steps):
        sim.step()
        out.append(float(sim.p[probe]))
    return np.array(out)


def test_cpml_beats_sponge_and_mur_in_2d() -> None:
    ref = _trace(600, "soft")
    peak = float(np.abs(ref).max())
    errs = {b: float(np.abs(_trace(200, b) - ref).max()) / peak for b in ("mur", "sponge", "cpml")}
    assert errs["cpml"] < 3e-3, errs  # < -50 dB of the direct peak
    assert errs["cpml"] < errs["sponge"] / 5
    assert errs["cpml"] < errs["mur"] / 20


def test_cpml_3d_and_long_run_stability() -> None:
    ref = _trace(160, "soft", dims=3, steps=160, r=20)
    peak = float(np.abs(ref).max())
    err = float(np.abs(_trace(80, "cpml", dims=3, steps=160, r=20) - ref).max()) / peak
    assert err < 0.01, err
    # No late-time growth (the plain PML's known weakness; alpha suppresses it).
    sim = Simulate((64, 64), boundary="cpml", cpml_cells=12, drivers=[Driver((32, 32), _pulse())])
    peak = 0.0
    for _ in range(100):
        sim.step()
        peak = max(peak, float(np.abs(sim.p).max()))
    late = []
    for k in range(8000):
        sim.step()
        if k >= 1000 and k % 500 == 0:
            late.append(float(np.abs(sim.p).max()))
    # Only the slowly decaying 2D wake remains: tiny, and not growing.
    assert max(late) < 1e-3 * peak
    assert max(late[len(late) // 2 :]) <= 1.5 * max(late[: len(late) // 2])


def test_cpml_per_face_and_reset() -> None:
    sim = Simulate((80, 80), boundary=("cpml", "rigid", "rigid", "rigid"), cpml_cells=12)
    sim.set_drivers([Driver((40, 40), _pulse())])
    for _ in range(300):
        sim.step()
    assert sim._cpml is not None and float(np.abs(sim._cpml.axes[0]["psi"]).max()) > 0
    assert sim._cpml.axes[1] == {}  # axis 1 has no CPML face
    assert float(np.abs(sim.p[0, :]).max()) == 0.0  # held outer cell
    sim.reset()
    assert float(np.abs(sim._cpml.axes[0]["psi"]).max()) == 0.0
    # Painting geometry keeps the layer state (no reset mid-run).
    for _ in range(50):
        sim.step()
    layer = sim._cpml
    sim.set_material([(60, 60)], 2)
    sim.step()
    assert sim._cpml is layer


def test_walls_inside_the_layer_stay_stable() -> None:
    """Rigid, impedance and soft cells painted into the layer (the case that
    diverged before the no-flux masking) must not blow up."""
    sim = Simulate((60, 50), boundary=("cpml", "cpml", "rigid", "cpml"), cpml_cells=10)
    sim.set_material([(i, 10) for i in range(10, 60)], 2)
    sim.set_material([(55, j) for j in range(5, 25)], 5)
    sim.set_material([(i, 45) for i in range(20, 40)], 3)
    sim.set_obstacle([(20, j) for j in range(35, 50)])
    sim.set_drivers([Driver((30, 22), _pulse())])
    peak = 0.0
    for k in range(6000):
        sim.step()
        if k < 200:
            peak = max(peak, float(np.abs(sim.p).max()))
    assert np.isfinite(sim.p).all()
    assert float(np.abs(sim.p).max()) < 0.05 * peak
