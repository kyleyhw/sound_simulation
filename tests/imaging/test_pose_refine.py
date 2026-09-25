"""Pose refinement from the empty-box model (plan 6.6.2)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.imaging.ir import empty_room_response
from acoustic_system.imaging.pose_refine import (
    quiet_cost,
    refine_pose,
    refine_pose_joint,
    window_cells,
)

N = 32
T = 160


@pytest.fixture(scope="module")
def drive(v2_drive):
    return v2_drive[:T]


def test_engine_is_reciprocal(drive):
    a, _ = empty_room_response((N, N), (8, 20), drive, [(22, 5)])
    b, _ = empty_room_response((N, N), (22, 5), drive, [(8, 20)])
    assert np.abs(a - b).max() < 1e-5 * np.abs(a).max()


def test_window_cells_stay_inside():
    c = window_cells(np.array([1, 30]), 2, (N, N))
    assert c[:, 0].min() == 1 and c[:, 1].max() == N - 2
    assert len(c) == 3 * 3


def test_quiet_cost_prefers_the_exact_trace():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(20), rng.standard_normal(80)])
    exact = y.copy()
    late = y.copy()
    late[60:] += 0.5  # the same error, starting late
    early = y.copy()
    early[25:] += 0.5  # starting early
    c = quiet_cost(y, np.stack([exact, late, early], 1))
    assert c[0] < c[1] < c[2]


@pytest.mark.parametrize("refiner", [refine_pose, refine_pose_joint])
def test_refinement_recovers_a_pose_in_the_empty_box(drive, refiner):
    """With no obstacles the empty-box model is exact, so the true cells are found."""
    src, mics = np.array([10, 12]), np.array([[20, 8], [22, 20]])
    y, _ = empty_room_response((N, N), src, drive, mics)
    res = refiner((N, N), y, drive, src + [1, -1], mics + [[-1, 0], [1, 1]], radius=1)
    np.testing.assert_array_equal(res.source, src)
    np.testing.assert_array_equal(res.mics, mics)
    assert res.cost < res.cost_assumed
    np.testing.assert_allclose(res.incident[0], y, atol=1e-6 * np.abs(y).max())


def test_joint_refinement_with_the_true_map_is_exact(drive):
    """With obstacles in the room and the true map in the model, the fit is exact."""
    from acoustic_system.imaging.ir import SampledWaveform
    from acoustic_system.simulation.setup import Driver
    from acoustic_system.simulation.simulate import Simulate

    mask = np.zeros((N, N), dtype=bool)
    mask[14:18, 14:20] = True
    src, mics = np.array([8, 8]), np.array([[8, 22], [24, 10]])
    sim = Simulate((N, N))
    sim.set_obstacle_mask(mask)
    sim.set_drivers([Driver(tuple(src), SampledWaveform(drive, sim.timestep))])
    y = np.zeros((2, T))
    for n in range(T):
        sim.step()
        y[:, n] = sim.p[mics[:, 0], mics[:, 1]]
    res = refine_pose_joint(
        (N, N), y, drive, src + [1, 0], mics + [[0, 1], [-1, -1]], radius=1, mask=mask
    )
    np.testing.assert_array_equal(res.source, src)
    np.testing.assert_array_equal(res.mics, mics)
    assert res.cost < 1e-8 * float((y**2).sum())
