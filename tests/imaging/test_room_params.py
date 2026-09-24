"""Room acoustic parameters (plan 6.5)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.imaging import room_params as rp


def test_schroeder_recovers_synthetic_t60():
    rng = np.random.default_rng(0)
    dt, t60 = 0.5, 400.0
    t = np.arange(4000) * dt
    h = rng.standard_normal(t.size) * 10 ** (-3 * t / t60)  # -60 dB at t60 (amplitude)
    assert rp.t60_schroeder(h, dt) == pytest.approx(t60, rel=0.05)
    assert rp.t60_schroeder(h, dt, (-5, -35)) == pytest.approx(t60, rel=0.05)


def test_drr_and_absorption_inversions():
    h = np.zeros(100)
    h[10] = 1.0
    h[30:] = 0.1
    assert rp.direct_to_reverberant(h, window=2) == pytest.approx(10 * np.log10(1 / (70 * 0.01)))
    a, S, L = 0.3, 3000.0, 220.0
    assert rp.alpha_eyring_2d(rp.t60_eyring_2d(a, S, L), S, L) == pytest.approx(a)
    assert rp.alpha_sabine_2d(rp.t60_sabine_2d(a, S, L), S, L) == pytest.approx(a)
    assert rp.alpha_diffuse_2d(1e-9) == pytest.approx(0.0, abs=1e-6)


def test_dc_free_chirp_moments():
    s = rp.dc_free_chirp(400, 0.5, 0.02, 0.25)
    n = np.arange(s.size)
    assert abs(s.sum()) < 1e-9 and abs((n * s).sum()) < 1e-6


def test_t60_of_simulated_absorbing_room_matches_diffuse_theory():
    """Chirp -> deconvolution -> Schroeder in an 80 x 60 room with known admittance."""
    shape, beta = (80, 60), 0.1
    drive = rp.dc_free_chirp(400, 0.5, 0.02, 0.25)
    mics = [(60, 43), (29, 46)]
    rec, dt = rp.simulate_absorbing_room(shape, beta, (20, 15), mics, drive, 8000)
    area, perim = 80 * 60, 2 * (80 + 60)
    ad = rp.alpha_diffuse_2d(beta)
    t_sab, t_eyr = rp.t60_sabine_2d(ad, area, perim), rp.t60_eyring_2d(ad, area, perim)
    for r in rec:
        est = rp.estimate_room_params(r, drive, dt, area, perim, band=(0.04, 0.2))
        # FDTD sits between the Eyring and Sabine predictions (docs/physics.md).
        assert 0.95 * t_eyr < est.t60 < 1.05 * t_sab
        assert est.alpha_eyring == pytest.approx(ad, rel=0.25)
        assert np.isfinite(est.drr_db)
