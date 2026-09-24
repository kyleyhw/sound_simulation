"""Shared scenes for the Phase 7 control tests."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.control.transfer import Box, Room, TransferSet, disk_points, measure_transfer


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    """Run these tests with one numba (and torch) thread.

    The control tests step small grids tens of thousands of times; numba's
    parallel loops only add overhead there and collapse when the cores are
    shared. Restored afterwards so other test directories are unaffected.
    """
    import numba
    import torch

    n, t = numba.get_num_threads(), torch.get_num_threads()
    numba.set_num_threads(1)
    torch.set_num_threads(1)
    yield
    numba.set_num_threads(n)
    torch.set_num_threads(t)


# The furnished test room of docs/control.md: 3.0 x 2.4 m, impedance walls,
# a rigid sofa and an absorber panel.
FURNITURE = (Box(0.2, 1.9, 0.7, 2.25), Box(2.6, 0.1, 2.85, 0.5, 5))


def furnished_room(beta: float = 0.3) -> Room:
    return Room(size=(3.0, 2.4), beta=beta, boxes=FURNITURE)


@pytest.fixture(scope="session")
def make_room():
    """Factory for the furnished room with wall admittance ``beta``."""
    return furnished_room


@pytest.fixture(scope="module")
def zone_scene() -> tuple[Room, TransferSet, np.ndarray, np.ndarray]:
    """8-speaker line array, bright zone left, dark zone right, absorbing room."""
    room = furnished_room(0.3)
    speakers = np.array([[1.15 + 0.1 * i, 0.4] for i in range(8)])
    bright = disk_points(room, (1.0, 1.6), 0.15)
    dark = disk_points(room, (2.0, 1.6), 0.15)
    # Bright-zone centre first (the reference point of the broadband designs).
    c = int(np.argmin(np.linalg.norm(bright - [1.0, 1.6], axis=1)))
    bright = np.concatenate([bright[c : c + 1], np.delete(bright, c, axis=0)])
    ts = measure_transfer(room, speakers, np.concatenate([bright, dark]), duration=0.15)
    ib = np.arange(len(bright))
    id_ = np.arange(len(bright), len(bright) + len(dark))
    return room, ts, ib, id_
