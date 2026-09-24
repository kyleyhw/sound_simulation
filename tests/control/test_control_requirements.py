"""Design in the estimated room, score in the true room (plan 7.5)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.control.requirements import (
    SensingError,
    SensingStudy,
    estimated_room,
    threshold_crossing,
    wall_errors,
)
from acoustic_system.control.transfer import Box, Room, disk_points


def test_estimated_room_geometry() -> None:
    room = Room(size=(3.0, 2.4), boxes=(Box(0.2, 1.9, 0.7, 2.25),), beta=0.2)
    est = estimated_room(room, SensingError(wall_m=0.05, signs=(1, 1, -1, -1)))
    assert est.origin == pytest.approx((-0.05, 0.05))
    assert est.size == pytest.approx((3.1, 2.3))
    assert est.shape == (125, 93)
    e2 = estimated_room(room, SensingError(beta_scale=2.0, box_shift_m=(0.1, 0.0)))
    assert e2.beta == pytest.approx(0.4)
    assert e2.boxes[0].x0 == pytest.approx(0.3)
    assert estimated_room(room, SensingError(drop_boxes=True)).boxes == ()
    errs = wall_errors([0.0, 0.025], n_patterns=2)
    assert len(errs) == 4 and errs[0].exact and not errs[2].exact


def test_threshold_crossing() -> None:
    assert threshold_crossing([0, 1, 2], [20, 12, 8], 10.0) == pytest.approx(1.5)
    assert threshold_crossing([0, 1], [20, 15], 10.0) == np.inf
    assert threshold_crossing([0, 1], [5, 4], 10.0) == 0.0


def test_model_error_costs_contrast() -> None:
    room = Room(size=(2.0, 1.6), beta=0.1, boxes=(Box(0.1, 1.2, 0.4, 1.5),))
    spk = np.array([[0.75 + 0.1 * i, 0.3] for i in range(6)])
    study = SensingStudy(
        room,
        spk,
        disk_points(room, (0.6, 1.1), 0.1),
        disk_points(room, (1.4, 1.1), 0.1),
        duration=0.2,
        freq_step=4,
    )
    exact = study.evaluate(SensingError())
    assert exact.band_db == pytest.approx(exact.predicted_db)
    assert exact.band_db > 15.0
    wall = study.evaluate(SensingError(wall_m=0.05))
    assert wall.band_db < exact.band_db - 5.0
    assert wall.predicted_db > wall.band_db + 5.0  # the wrong model is over-confident
