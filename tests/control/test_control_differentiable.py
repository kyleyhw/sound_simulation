"""Drive signals optimised through the tensor engine (plan 7.6)."""

from __future__ import annotations

import numpy as np
import torch

from acoustic_system.control.beamforming import design_broadband
from acoustic_system.control.differentiable import DiffScene, gradient_check, optimise_drives
from acoustic_system.control.transfer import Room, measure_transfer

torch.set_num_threads(1)


def _scene(dtype=torch.float32, n_taps=48) -> DiffScene:
    room = Room(size=(1.4, 1.2), dx=0.05, beta=0.3)
    spk = np.array([[0.5 + 0.1 * i, 0.15] for i in range(4)])
    bright = np.array([[0.35, 0.9], [0.4, 0.9], [0.35, 0.95], [0.4, 0.95]])
    dark = np.array([[1.0, 0.9], [1.05, 0.9], [1.0, 0.95], [1.05, 0.95]])
    return DiffScene(
        room, spk, bright, dark, n_taps=n_taps, programme_taps=41, tail=0.01, dtype=dtype
    )


def test_gradient_matches_finite_difference() -> None:
    sc = _scene(torch.float64, n_taps=16)
    assert gradient_check(sc) < 1e-6


def test_optimisation_improves_contrast_and_replays_in_numba() -> None:
    sc = _scene()
    r = optimise_drives(sc, iters=15, lr=0.1)
    assert r.final_db > r.initial_db + 5.0
    assert abs(sc.verify_numba(r.z) - r.final_db) < 0.05


def test_optimisation_refines_acc() -> None:
    sc = _scene()
    pts = np.concatenate([sc.bright, sc.dark])
    ts = measure_transfer(sc.room, sc.speakers, pts, duration=0.06, band=(100.0, 1000.0))
    ib, id_ = np.arange(4), np.arange(4, 8)
    acc = design_broadband(ts, ib, id_, n_taps=sc.n_taps, band=sc.band, oversample=4).taps
    r = optimise_drives(sc, acc, iters=10, lr=0.05)
    assert abs(r.initial_db - sc.contrast_db(acc)) < 1e-3
    assert r.final_db > r.initial_db + 1.0
