"""Compact 9-point scheme (plan 5.7.1): stability at Courant 1, lower dispersion."""

from __future__ import annotations

import numpy as np

from acoustic_system.simulation.setup import Driver
from acoustic_system.simulation.simulate import Simulate
from acoustic_system.simulation.waveforms import RickerWavelet


def _mode_error(scheme: str, courant: float) -> float:
    """Worst relative frequency error over two high modes of a p = 0 box.

    The field is initialised with an exact discrete sine mode (an eigenvector
    of both stencils with Dirichlet walls) at rest, so it oscillates at a
    single frequency: the scheme's eigenfrequency for that wavevector.
    Modes: (6, 6), about 7 cells per wavelength along the diagonal, and
    (10, 0), 6 cells per wavelength along an axis.
    """
    n = 31
    L = n - 1
    i = np.arange(n)
    worst = 0.0
    for m, k in ((6, 6), (10, 0)):
        sim = Simulate((n, n), courant=courant, scheme=scheme)
        ky = k if k else 1  # (10, 0) is not a Dirichlet mode; use (10, 1)
        mode = np.outer(np.sin(np.pi * m * i / L), np.sin(np.pi * ky * i / L)).astype(np.float32)
        sim.p[...] = mode
        sim.p_prev[...] = mode
        tr = []
        for _ in range(int(4000 / sim.timestep)):
            sim.step()
            tr.append(float(sim.p[7, 9]))
        x = np.array(tr) * np.hanning(len(tr))
        nfft = 1 << 20
        spec = np.abs(np.fft.rfft(x, nfft))
        f = np.fft.rfftfreq(nfft, sim.timestep)
        f_meas = f[np.argmax(spec)]
        f_true = 0.5 * np.hypot(m / L, ky / L)
        worst = max(worst, abs(f_meas / f_true - 1))
    return worst


def test_compact_is_stable_at_courant_one() -> None:
    sim = Simulate((64, 64), courant=1.0, scheme="compact")
    assert sim.timestep == 1.0
    sim.set_drivers([Driver((20, 20), RickerWavelet(1.0, 0.2, 8.0))])
    for _ in range(5000):
        sim.step()
    assert np.isfinite(sim.p).all() and np.abs(sim.p).max() < 10


def test_compact_scheme_has_lower_dispersion() -> None:
    std = _mode_error("standard", 0.5)
    cmp = _mode_error("compact", 1.0)
    assert cmp < 0.5 * std, (std, cmp)
