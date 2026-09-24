"""General-path physics (plan Phase 5): walls, impedance, absorbing edges, c(x).

Reflection is measured with a plane pulse in a waveguide: rigid side walls
(so a line source launches an exact plane wave), travelling along axis 0 to
the wall under test at the far end. The incident and reflected pulses are
separated in time at a probe row.
"""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.simulation.physics import MATERIALS, absorption_from_beta
from acoustic_system.simulation.setup import Driver
from acoustic_system.simulation.simulate import Simulate
from acoustic_system.simulation.waveforms import RickerWavelet


def _energy(sim: Simulate) -> float:
    p = sim.p.astype(np.float64)
    pp = sim.p_prev.astype(np.float64)
    kin = (((p - pp) / sim.timestep) ** 2).sum()
    pot = sum((np.diff(p, axis=a) * np.diff(pp, axis=a)).sum() for a in range(p.ndim))
    return 0.5 * kin + 0.5 * pot


def _pulse(delay: float = 15.0) -> RickerWavelet:
    return RickerWavelet(amplitude=1.0, frequency=0.08, delay=delay)


@pytest.mark.parametrize("boundary", ["soft", "rigid"])
def test_lossless_boxes_conserve_energy(boundary: str) -> None:
    sim = Simulate((80, 80), boundary=boundary, drivers=[Driver((40, 40), _pulse())])
    for _ in range(120):
        sim.step()
    e0 = _energy(sim)
    for _ in range(500):
        sim.step()
    assert abs(_energy(sim) / e0 - 1.0) < 1e-3


def _plane_wave_reflection(
    wall_setup, length: int = 300, width: int = 8, freq: float = 0.04
) -> float:
    """Signed reflection coefficient of a wall for a plane pulse.

    Magnitude: sqrt of reflected/incident window energy at the probe
    (robust to the numerical-dispersion spreading that lowers peaks over
    a long path). Sign: of the correlation between the two windows.
    """
    sim = Simulate((length, width), boundary="rigid", courant=0.5)
    wall_setup(sim, length, width)
    src_row, probe_row = 40, 120
    delay = 3.0 / freq
    sim.set_drivers([Driver((src_row, j), RickerWavelet(1.0, freq, delay)) for j in range(width)])
    trace = []
    wall_row = length - 1
    t_end = delay + (wall_row - src_row) + (wall_row - probe_row) + 3.0 / freq
    for _ in range(int(t_end / sim.timestep)):
        sim.step()
        trace.append(float(sim.p[probe_row, width // 2]))
    tr = np.array(trace)
    t = np.arange(len(tr)) * sim.timestep
    t_inc = delay + (probe_row - src_row)
    t_ref = delay + (wall_row - src_row) + (wall_row - probe_row)
    win = 2.5 / freq
    inc = tr[np.abs(t - t_inc) < win]
    ref = tr[np.abs(t - t_ref) < win]
    n = min(len(inc), len(ref))
    inc, ref = inc[:n], ref[:n]
    mag = float(np.sqrt((ref**2).sum() / (inc**2).sum()))
    sign = 1.0 if float(np.dot(inc, ref)) >= 0 else -1.0
    return sign * mag


def test_rigid_wall_reflects_with_plus_one() -> None:
    r = _plane_wave_reflection(lambda sim, n, w: None)  # outer rigid at the far end
    assert r == pytest.approx(1.0, abs=0.05)


def test_soft_wall_reflects_with_minus_one() -> None:
    def setup(sim, n, w):
        sim.set_material([(n - 1, j) for j in range(w)], 1)

    assert _plane_wave_reflection(setup) == pytest.approx(-1.0, abs=0.05)


@pytest.mark.parametrize("mid", [3, 4, 5])
def test_impedance_wall_reflection_matches_theory(mid: int) -> None:
    beta = MATERIALS[mid].beta

    def setup(sim, n, w):
        sim.set_material([(n - 1, j) for j in range(w)], mid)

    r = _plane_wave_reflection(setup)
    theory = (1 - beta) / (1 + beta)
    assert r == pytest.approx(theory, abs=0.06), (mid, r, theory)
    assert absorption_from_beta(beta) == pytest.approx(1 - theory**2)


def test_frequency_dependent_curtain_absorbs_highs_more_than_lows() -> None:
    def setup(sim, n, w):
        sim.set_material([(n - 1, j) for j in range(w)], 6)

    def refl(freq: float) -> float:
        sim = Simulate((300, 8), boundary="rigid")
        setup(sim, 300, 8)
        sim.set_drivers([Driver((40, j), RickerWavelet(1.0, freq, 3 / freq)) for j in range(8)])
        trace = []
        for _ in range(1300):
            sim.step()
            trace.append(float(sim.p[120, 4]))
        tr = np.abs(np.array(trace))
        half = len(tr) // 2
        return float(tr[half:].max() / tr[:half].max())

    assert refl(0.12) < refl(0.02) - 0.2


def test_mur_and_sponge_absorb_outgoing_waves() -> None:
    """Edge reflection = deviation from a free-field reference domain.

    A 2D point source leaves a slowly decaying wake, so 'late signal' is not
    reflection. Instead compare the probe trace with the same source in a
    domain 3x larger (its walls are too far to reflect within the window).
    """

    def trace(n: int, boundary: str) -> np.ndarray:
        c = n // 2
        sim = Simulate((n, n), boundary=boundary, drivers=[Driver((c, c), _pulse())])
        out = []
        for _ in range(420):
            sim.step()
            out.append(float(sim.p[c, c + 50]))
        return np.array(out)

    ref = trace(600, "soft")
    peak = float(np.abs(ref).max())
    errs = {b: float(np.abs(trace(200, b) - ref).max()) / peak for b in ("soft", "mur", "sponge")}
    assert errs["soft"] > 0.5  # a hard edge reflects strongly (sanity check of the test)
    assert errs["mur"] < 0.15
    assert errs["sponge"] < 0.05


def test_speed_map_changes_travel_time() -> None:
    def arrival(speed: float) -> float:
        sim = Simulate((600, 8), boundary="rigid")
        sim.set_speed_map(np.full((600, 8), speed, dtype=np.float32))
        sim.set_drivers([Driver((300, j), _pulse()) for j in range(8)])
        for step in range(1, 1200):
            sim.step()
            if abs(float(sim.p[400, 4])) > 0.5:  # first arrival
                return step * sim.timestep
        raise AssertionError("no arrival")

    t1, t2 = arrival(1.0), arrival(0.5)
    # Travel over 100 cells: 100 vs 200 time units.
    assert (t2 - t1) == pytest.approx(100.0, rel=0.05)


def test_default_is_fast_path_and_general_rejects_gpu_style_misuse() -> None:
    sim = Simulate((32, 32))
    assert sim._general is False
    with pytest.raises(ValueError):
        Simulate((32, 32), boundary="nonsense")
    sim.set_material([(5, 5)], 1)  # soft = obstacle-equivalent, still general-free
    assert sim._general is False
    sim.set_material([(6, 6)], 2)
    assert sim._general is True


def test_per_face_boundaries() -> None:
    """Open (sponge) on one face, rigid elsewhere: energy leaves only
    through the open face, and the p = 0 held face stays exactly zero."""
    sim = Simulate((120, 120), boundary=("rigid", "sponge", "rigid", "soft"), sponge_cells=16)
    sim.set_drivers([Driver((60, 60), _pulse())])
    for _ in range(100):
        sim.step()
    e0 = _energy(sim)
    for _ in range(800):
        sim.step()
    assert _energy(sim) < 0.6 * e0  # losing energy through the open face
    assert float(np.abs(sim.p[:, -1]).max()) == 0.0  # soft face held at 0
    assert float(np.abs(sim.p[0, 1:-1]).max()) > 0.0  # rigid face is active
    with pytest.raises(ValueError):
        Simulate((20, 20), boundary=("rigid", "soft"))
