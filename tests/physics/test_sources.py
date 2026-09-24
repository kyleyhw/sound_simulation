"""Source/receiver models (plan 5.6): sub-cell, band-limited, directivity, mic."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.simulation.simulate import Simulate
from acoustic_system.simulation.sources import (
    MicModel,
    cardioid,
    dipole,
    point_source,
    sample_subcell,
    subcell_weights,
)
from acoustic_system.simulation.waveforms import RickerWavelet


def test_subcell_weights_sum_to_one_and_interpolate_linearly() -> None:
    w = subcell_weights((3.25, 7.5))
    assert sum(v for _, v in w) == pytest.approx(1.0)
    f = np.add.outer(np.arange(10.0), 2 * np.arange(10.0))  # f = i + 2j (linear)
    assert sample_subcell(f, (3.25, 7.5)) == pytest.approx(3.25 + 15.0)


def _arrival(pos: tuple[float, float]) -> float:
    sim = Simulate((200, 40), boundary="sponge", sponge_cells=12)
    sim.set_drivers(point_source(pos, RickerWavelet(1.0, 0.05, 30.0)))
    tr = []
    for _ in range(500):
        sim.step()
        tr.append(float(sim.p[140, 20]))
    tr = np.array(tr)
    k = int(np.argmax(tr))
    a, b, c = tr[k - 1 : k + 2]
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sim.timestep


def test_half_cell_shift_moves_arrival_by_half_a_time_unit() -> None:
    assert _arrival((60.5, 20.0)) - _arrival((60.0, 20.0)) == pytest.approx(-0.5, abs=0.08)


def _pattern(drivers, radius: int = 45, n: int = 240) -> dict[int, float]:
    sim = Simulate((n, n), boundary="sponge", sponge_cells=20)
    sim.set_drivers(drivers)
    c = n // 2
    angles = {0: (c, c + radius), 90: (c - radius, c), 180: (c, c - radius)}
    peaks = {a: 0.0 for a in angles}
    for _ in range(420):
        sim.step()
        for a, q in angles.items():
            peaks[a] = max(peaks[a], abs(float(sim.p[q])))
    return peaks


def test_dipole_has_a_broadside_null() -> None:
    c = 120
    pk = _pattern(dipole((c, c), (0, 1), RickerWavelet(1.0, 0.05, 30.0), spacing=2.0))
    assert pk[90] < 0.1 * pk[0]
    assert pk[180] == pytest.approx(pk[0], rel=0.15)


def test_cardioid_front_to_back_ratio() -> None:
    c = 120
    pk = _pattern(cardioid((c, c), (0, 1), RickerWavelet(1.0, 0.05, 30.0), spacing=2.0))
    assert 20 * np.log10(pk[0] / pk[180]) > 10.0


def test_band_limited_source_has_less_high_frequency_ringing() -> None:
    """A Gaussian-spread source (width 1 cell) filters wavenumbers near the
    grid Nyquist, which a broadband single-cell source excites strongly."""

    def hf(spread: float) -> float:
        sim = Simulate((200, 200), boundary="sponge", sponge_cells=16)
        sim.set_drivers(point_source((100.0, 100.0), RickerWavelet(1.0, 0.25, 8.0), spread=spread))
        tr = []
        for _ in range(300):
            sim.step()
            tr.append(float(sim.p[100, 150]))
        spec = np.abs(np.fft.rfft(tr))
        f = np.fft.rfftfreq(len(tr), sim.timestep)
        return float(spec[f > 0.3].sum() / spec.sum())

    assert hf(1.0) < 0.5 * hf(0.0)


def test_mic_model_snr_and_quantisation() -> None:
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 200, 20000))
    y = MicModel(snr_db=20).apply(x, rng)
    noise = y - x
    snr = 10 * np.log10(np.mean(x**2) / np.mean(noise**2))
    assert snr == pytest.approx(20, abs=0.5)
    q = MicModel(bits=4, full_scale=1.0).apply(x, rng)
    assert len(np.unique(q)) <= 16
