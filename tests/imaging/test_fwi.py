"""Full-waveform inversion proof of concept (plan 6.2.5)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.imaging.ir import SampledWaveform

torch = pytest.importorskip("torch")


def test_fwi_reduces_misfit_and_finds_the_obstacle(v2_drive):
    from acoustic_system.imaging.fwi import invert_room, total_variation
    from acoustic_system.simulation.setup import Driver
    from acoustic_system.simulation.simulate import Simulate

    torch.set_num_threads(1)
    n, steps = 32, 160
    drive = v2_drive[:steps]
    mask = np.zeros((n, n), dtype=bool)
    mask[14:18, 17:21] = True
    sources = np.array([[8, 8], [24, 8]])
    mics = np.array([[[6, 12], [10, 4]], [[26, 12], [22, 4]]])
    rec = np.zeros((2, steps, 2))
    for k in range(2):
        sim = Simulate((n, n))
        sim.set_obstacle_mask(mask)
        sim.set_drivers([Driver(tuple(sources[k]), SampledWaveform(drive, sim.timestep))])
        for t in range(steps):
            sim.step()
            rec[k, t] = [sim.p[tuple(m)] for m in mics[k]]
    res = invert_room((n, n), sources, mics, drive, rec, iterations=(8, 8), cutoffs=(0.2, 0.47))
    assert res.history[-1] < 0.5 * res.history[0]
    assert res.occupancy.shape == (n, n) and 0 <= res.occupancy.min() <= res.occupancy.max() <= 1
    # The recovered occupancy is highest on the illuminated (left) face region.
    near = np.zeros((n, n), dtype=bool)
    near[12:20, 15:23] = True
    assert res.occupancy[near].max() > 5 * np.median(res.occupancy[~near])
    # TV of a constant map is only the epsilon floor.
    tv = float(total_variation(torch.zeros(4, 4), eps=1e-2))
    assert tv == pytest.approx(16 * 1e-2)
