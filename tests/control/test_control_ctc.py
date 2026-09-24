"""Two-speaker crosstalk cancellation (plan 7.3)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.control.ctc import (
    TrackedCtc,
    displacement_sweep,
    ear_points,
    head_grid,
    kirkeby_inverse,
    stereo_separation_db,
    verify_ctc,
)
from acoustic_system.control.transfer import bandpass_pulse, measure_transfer

SPEAKERS = np.array([[1.35, 0.6], [1.65, 0.6]])  # 30 cm laptop span
HEAD = np.array([1.5125, 1.1])  # 50 cm in front; ears land on cells
EVAL = (300.0, 1500.0)


def _tracker(make_room, beta: float, lateral: float = 0.05):
    room = make_room(beta)
    pts = head_grid(room, HEAD, lateral, 0.0)
    ts = measure_transfer(room, SPEAKERS, pts, duration=0.3)
    return room, ts, TrackedCtc(ts, eval_band=EVAL)


@pytest.fixture(scope="module")
def absorbing(make_room):
    return _tracker(make_room, 0.3)


def test_kirkeby_inverse_limits() -> None:
    rng = np.random.default_rng(0)
    H = rng.standard_normal((5, 2, 2)) + 1j * rng.standard_normal((5, 2, 2))
    np.testing.assert_allclose(
        H @ kirkeby_inverse(H, 1e-12), np.broadcast_to(np.eye(2), H.shape), atol=1e-8
    )
    # Regularisation shrinks the filters.
    assert np.all(np.abs(kirkeby_inverse(H, 1.0)) <= np.abs(kirkeby_inverse(H, 1e-6)).max())


def test_ears_are_grid_points(make_room) -> None:
    room = make_room(0.3)
    e = ear_points(HEAD)
    assert e[1, 0] - e[0, 0] == pytest.approx(0.175)
    ij = room.cell_float(e)
    np.testing.assert_allclose(ij, np.rint(ij), atol=1e-9)


def test_ctc_separation_at_design_position(absorbing) -> None:
    """Headline (7.3.1): >= 15 dB separation in the absorbing room, played in the engine."""
    room, ts, tr = absorbing
    d = tr.design(HEAD)
    sep = tr.separation(d, HEAD)
    natural = stereo_separation_db(tr.plant(HEAD)[tr._eval])
    assert np.mean(natural) < 3.0  # two free-field points: plain stereo barely separates
    assert np.mean(sep) > 30.0
    assert np.mean(sep.min(axis=1) >= 15.0) > 0.8  # per-frequency, both programmes
    x = bandpass_pulse(ts.dt, EVAL, 511)
    for ch in (0, 1):
        chk = verify_ctc(room, ts, d.taps, HEAD, x, EVAL, channel=ch)
        assert chk.measured_db >= 15.0
        assert abs(chk.measured_db - chk.predicted_db) < 0.2


def test_ctc_every_frequency_with_absorbing_walls(make_room) -> None:
    """With beta = 1 walls the separation is >= 15 dB at every frequency."""
    _, _, tr = _tracker(make_room, 1.0, lateral=0.0)
    sep = tr.separation(tr.design(HEAD), HEAD)
    assert sep.min() >= 15.0


def test_head_displacement_and_tracking(absorbing) -> None:
    _, _, tr = absorbing
    offs = np.array([[k * 0.025, 0.0] for k in (-2, -1, 0, 1, 2)])
    fixed = displacement_sweep(tr, HEAD, offs)
    tracked = displacement_sweep(tr, HEAD, offs, tracked=True)
    assert fixed[2] == pytest.approx(tracked[2])
    assert fixed[0] < fixed[2] - 15.0 and fixed[4] < fixed[2] - 15.0  # 5 cm off: much worse
    assert np.all(tracked > 25.0)  # re-designed filters restore separation
