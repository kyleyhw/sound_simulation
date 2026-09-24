"""Shared fixtures for the imaging tests (plan Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.simulation.dataset import synthetic_chirp


@pytest.fixture(scope="session")
def v2_drive() -> np.ndarray:
    """The v2 archive drive: 0.02 -> 0.45 chirp, amplitude 5, sampled at dt = 0.5."""
    u = synthetic_chirp(40000, 200.0, 0.02, 0.45)
    return 5.0 * u[::100].astype(np.float64)
