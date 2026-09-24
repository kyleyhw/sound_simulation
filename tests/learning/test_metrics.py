"""Threshold-free sensing metrics (plan 3.4.6)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.learning.metrics import (
    average_precision,
    boundary,
    boundary_f,
    info_gain_bits,
    iou,
    skill,
)


def _room() -> np.ndarray:
    m = np.zeros((16, 16), dtype=bool)
    m[4:9, 5:11] = True
    return m


def test_info_gain_is_zero_for_the_prior_and_positive_for_the_truth() -> None:
    t = _room()
    prior = np.full(t.shape, 0.1)
    assert info_gain_bits(prior, t, prior) == pytest.approx(0.0)
    assert info_gain_bits(np.where(t, 0.9, 0.05), t, prior) > 0
    # Confidently wrong is worse than the prior (proper scoring).
    assert info_gain_bits(np.where(t, 0.01, 0.99), t, prior) < 0


def test_average_precision_perfect_and_random() -> None:
    t = _room()
    assert average_precision(t.astype(float), t) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    ap_rand = np.mean([average_precision(rng.random(t.shape), t) for _ in range(50)])
    assert abs(ap_rand - t.mean()) < 0.05  # random ranking ~ prevalence
    assert np.isnan(average_precision(np.ones((4, 4)), np.zeros((4, 4), dtype=bool)))


def test_boundary_and_boundary_f() -> None:
    t = _room()
    b = boundary(t)
    assert b.sum() == 2 * 5 + 2 * 6 - 4  # perimeter ring of a 5x6 block
    assert boundary_f(t, t)[2] == pytest.approx(1.0)
    shifted = np.roll(t, 1, axis=1)
    assert boundary_f(shifted, t, tol=1)[2] == pytest.approx(1.0)
    assert boundary_f(shifted, t, tol=0)[2] < 1.0


def test_iou_and_skill() -> None:
    t = _room()
    assert iou(t, t) == pytest.approx(1.0)
    assert skill(0.1, 0.1) == 0.0 and skill(1.0, 0.1) == pytest.approx(1.0)
