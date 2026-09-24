"""Cramér-Rao bounds (plan 6.4)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.imaging import crlb


def test_range_bound_scales_as_inverse_bandwidth_sqrt_snr():
    base = crlb.range_std(1000.0, 100.0)
    assert crlb.range_std(2000.0, 100.0) == pytest.approx(base / 2)
    assert crlb.range_std(1000.0, 400.0) == pytest.approx(base / 2)
    assert crlb.range_std(4000.0, 1600.0) == pytest.approx(base / 16)
    # Closed form for a flat band: sigma_R = (c/2) / (2 pi (B/sqrt 12) sqrt SNR).
    assert base == pytest.approx(0.5 * 343.0 / (2 * np.pi * (1000.0 / np.sqrt(12)) * 10.0))
    # Independent observations add information.
    assert crlb.range_std(1000.0, 100.0, n_obs=4) == pytest.approx(base / 2)


def test_coherent_bound_is_tighter_and_beta_matches_signals():
    env = crlb.range_std(1000.0, 100.0, centre=5000.0)
    coh = crlb.range_std(1000.0, 100.0, centre=5000.0, coherent=True)
    assert coh < env / 10
    # RMS bandwidth of a flat band from samples approaches B / sqrt(12).
    fs, n = 48000.0, 1 << 16
    rng = np.random.default_rng(0)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    X[(f < 4000) | (f > 6000)] = 0
    x = np.fft.irfft(X, n)
    assert crlb.signal_beta(x, fs) == pytest.approx(crlb.band_beta(4000, 6000), rel=0.03)
    assert crlb.signal_beta(x, fs, coherent=True) == pytest.approx(
        crlb.band_beta(4000, 6000, coherent=True), rel=0.03
    )


def test_bearing_bound_geometry():
    b = crlb.bearing_std(2000.0, 100.0, 0.2)
    assert crlb.bearing_std(2000.0, 100.0, 0.4) == pytest.approx(b / 2)
    assert crlb.bearing_std(2000.0, 100.0, 0.2, n_poses=9) == pytest.approx(b / 3)
    assert crlb.bearing_std(2000.0, 100.0, 0.2, theta=np.pi / 3) == pytest.approx(2 * b)
    # Two mics: sqrt(2) c sigma_tau / d; a pair's moment is d^2 / 2.
    s_tau = crlb.toa_std(2000.0 / np.sqrt(12), 100.0)
    assert b == pytest.approx(np.sqrt(2) * 343.0 * s_tau / 0.2)
    assert crlb.array_moment(2, 0.2) == pytest.approx(0.02)
    assert crlb.array_moment(4, 1.0) == pytest.approx(5.0)  # (-1.5..1.5)^2 summed


def test_ml_delay_estimator_attains_the_bound():
    """Monte Carlo: cross-correlation ML delay estimates reach the CRLB at high SNR."""
    rng = np.random.default_rng(1)
    fs, n = 1.0, 512
    t = np.arange(n)
    width = 3.0
    pulse = lambda d: np.exp(-0.5 * ((t - 200 - d) / width) ** 2)  # noqa: E731
    s = pulse(0.0)
    E = float((s**2).sum())
    sigma = 0.05
    snr = 2 * E / (2 * sigma**2)  # 2E/N0 with N0/2 = sigma^2 (unit sample spacing)
    beta = crlb.signal_beta(np.pad(s, 4096), fs, coherent=True)
    bound = float(crlb.toa_std(beta, snr))
    est = []
    for _ in range(300):
        d = rng.uniform(-0.5, 0.5)
        y = pulse(d) + sigma * rng.standard_normal(n)
        c = np.correlate(y, s, mode="full")[n - 10 : n + 10]
        k = int(np.argmax(c))
        a, b0, cc = c[k - 1], c[k], c[k + 1]
        frac = 0.5 * (a - cc) / (a - 2 * b0 + cc)
        est.append(k - 10 + frac - d)
    assert np.std(est) == pytest.approx(bound, rel=0.2)


def test_design_table_units_and_flags():
    rows = crlb.design_table(snr_db=20.0)
    names = [r["setup"] for r in rows]
    assert any("LAPTOP_ROOM" in n for n in names)
    ref = rows[0]
    # A 20 cm pair exceeds lambda_min / 2 = 10 cm at 1.7 kHz: coherent bearing is ambiguous.
    assert ref["grating_lobes"]
    assert ref["range_resolution_cm"] == pytest.approx(100 * 343.0 / (2 * 1400.0))
    wide = next(r for r in rows if r["setup"].startswith("laptop audible"))
    assert wide["range_env_mm"] < ref["range_env_mm"]
