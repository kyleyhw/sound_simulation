"""Impulse-response recovery (plan 6.2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.imaging.ir import (
    SampledWaveform,
    TikhonovDeconvolver,
    empty_room_response,
    envelope,
    free_field_green_2d,
    remove_direct_path_free_field,
    source_drive,
    tikhonov_deconvolve,
    wiener_deconvolve,
)
from acoustic_system.simulation.dataset import synthetic_chirp
from acoustic_system.simulation.setup import Driver
from acoustic_system.simulation.simulate import Simulate

DT = 0.5
T = 400


def test_source_drive_matches_engine_waveform(v2_drive):
    from acoustic_system.simulation.waveforms import AudioFileWaveform

    u = synthetic_chirp(40000, 200.0, 0.02, 0.45)
    wf = AudioFileWaveform.from_samples(samples=u, sample_rate=200.0, amplitude=5.0)
    ref = np.array([wf(n * DT) for n in range(T)])
    np.testing.assert_allclose(source_drive(u, T, DT, 200.0, 5.0), ref, atol=1e-6)
    np.testing.assert_allclose(v2_drive, ref, atol=1e-6)


def _synthetic_reflector(v2_drive):
    """Direct path at r_d plus a soft-wall (Gamma = -1) image source at r_i."""
    r_d, r_i = 10.0, 46.0
    h = free_field_green_2d(r_d, T, DT) - free_field_green_2d(r_i, T, DT)
    y = np.convolve(v2_drive, h)[:T]
    return y, h, r_d, r_i


@pytest.mark.parametrize("method", ["tikhonov", "wiener"])
def test_single_reflector_echo_delay(v2_drive, method):
    y, h, r_d, r_i = _synthetic_reflector(v2_drive)
    y = y + 1e-3 * np.abs(y).max() * np.random.default_rng(0).standard_normal(T)
    # Direct path removed with the analytic free-field model, fitted before the echo.
    res = remove_direct_path_free_field(y, v2_drive, r_d, DT, fit_window=int(r_i / DT) - 4)
    deconv = tikhonov_deconvolve if method == "tikhonov" else wiener_deconvolve
    env = envelope(deconv(res, v2_drive, 1e-2))
    # The echo is the strongest arrival left, at the image-source delay.
    assert abs(int(np.argmax(env)) - r_i / DT) <= 2
    # The direct path is gone: nothing comparable near its delay.
    j_d = int(r_d / DT)
    assert env[j_d - 3 : j_d + 4].max() < 0.2 * env.max()
    # And the full IR (direct included) peaks at the direct delay.
    full = envelope(deconv(y, v2_drive, 1e-2))
    assert abs(int(np.argmax(full)) - r_d / DT) <= 2


def test_tikhonov_inverts_the_truncated_convolution(v2_drive):
    h = np.zeros(T)
    h[40], h[130] = 1.0, -0.5
    y = TikhonovDeconvolver(v2_drive, T, lam=1e-6).S @ h
    est = tikhonov_deconvolve(y, v2_drive, lam=1e-6)
    # A delta is recovered as the chirp-band-limited pulse: right lags, signs and ratio.
    assert int(np.argmax(np.abs(est))) == 40
    assert est[40] > 0.8
    assert -0.6 < est[130] / est[40] < -0.4
    assert int(np.argmax(np.abs(est[100:]))) + 100 == 130


def test_engine_single_wall_residual_onset(v2_drive):
    """FDTD room with one straight soft wall: the scattered IR starts at the image delay."""
    n = 64
    src, mic = (32, 20), (32, 26)
    wall = np.zeros((n, n), dtype=bool)
    wall[8:56, 40] = True
    sim = Simulate((n, n))
    sim.set_obstacle_mask(wall)
    sim.set_drivers([Driver(src, SampledWaveform(v2_drive, sim.timestep))])
    y = np.zeros(T)
    for k in range(T):
        sim.step()
        y[k] = sim.p[mic]
    y0, _ = empty_room_response((n, n), src, v2_drive, [mic])
    residual = y - y0[0]
    # Specular path via the plane j = 40: image source at (32, 60), 34 cells from the mic.
    j_echo = 34.0 / DT
    # The residual is exactly zero until the scattered wave arrives (a few
    # steps of numerical precursor aside).
    onset = int(np.argmax(np.abs(residual) > 1e-3 * np.abs(residual).max()))
    assert j_echo - 6 <= onset <= j_echo + 1
    # The deconvolved scattered IR has its first echo at the image delay...
    env = envelope(tikhonov_deconvolve(residual, v2_drive, 1e-2))
    win = slice(int(j_echo) - 8, int(j_echo) + 9)
    peak = int(np.argmax(env[win])) + win.start
    assert abs(peak - j_echo) <= 5
    # ... and nothing comparable before it (the first lags carry a small
    # deconvolution edge effect, skipped).
    assert env[10 : int(j_echo) - 8].max() < 0.5 * env[peak]


def test_free_field_green_matches_engine():
    """Analytic 2D Green's function vs the engine before any wall echo (low band)."""
    n = 160
    src, mic = (80, 80), (80, 110)
    drive = 5.0 * synthetic_chirp(240, 2.0, 0.02, 0.15).astype(np.float64)
    y0, _ = empty_room_response((n, n), src, drive, [mic])
    model = np.convolve(drive, free_field_green_2d(30.0, 240, DT))[:240]
    # First wall echo: image across i = 0 is 2*80 + ... > 160 cells away; compare 0..220.
    keep = slice(0, 220)
    corr = np.corrcoef(y0[0, keep], model[keep])[0, 1]
    assert corr > 0.98
    ratio = np.abs(y0[0, keep]).max() / np.abs(model[keep]).max()
    assert 0.9 < ratio < 1.1
