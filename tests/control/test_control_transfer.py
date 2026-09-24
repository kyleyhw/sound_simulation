"""Transfer functions and metrics (plan 7.1)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.control.metrics import (
    acoustic_contrast_db,
    array_effort_db,
    band_contrast_db,
    band_energy,
    reproduction_error,
)
from acoustic_system.control.transfer import (
    Box,
    Room,
    bandpass_pulse,
    disk_points,
    measure_transfer,
    simulate_drives,
)


def _small_room() -> Room:
    return Room(size=(1.6, 1.2), beta=0.3, boxes=(Box(0.1, 0.9, 0.4, 1.1),))


def test_room_geometry() -> None:
    room = Room(size=(3.0, 2.4))
    assert room.shape == (121, 97)
    assert room.timestep == pytest.approx(0.5 * 0.025 / 343.0)
    ij = room.cells([[0.0, 0.0], [1.0, 0.5]])
    assert ij.tolist() == [[0, 0], [40, 20]]
    np.testing.assert_allclose(room.world(ij), [[0.0, 0.0], [1.0, 0.5]])
    # CPML rooms are padded: the interior keeps its size and coordinates.
    an = Room(size=(3.0, 2.4), boundary="cpml", cpml_cells=12)
    assert an.shape == (145, 121)
    np.testing.assert_allclose(an.world(an.cells([[1.0, 0.5]])), [[1.0, 0.5]])
    with pytest.raises(ValueError):
        room.cells([[3.5, 0.5]])
    furnished = _small_room()
    with pytest.raises(ValueError):
        furnished.cells([[0.2, 1.0]])  # inside the box
    pts = disk_points(room, (1.0, 1.0), 0.1)
    assert len(pts) == 49  # lattice points within 4 cells
    assert np.all(np.linalg.norm(pts - [1.0, 1.0], axis=1) <= 0.1 + 1e-9)


def test_pulse_is_band_limited_and_dc_free() -> None:
    dt = Room().timestep
    w = bandpass_pulse(dt, (100.0, 2000.0))
    assert abs(w.sum()) < 1e-9
    W = np.abs(np.fft.rfft(w, 8192))
    f = np.fft.rfftfreq(8192, dt)
    inband = W[(f > 300) & (f < 1800)]
    assert inband.min() > 0.9 * inband.max()
    assert W[f > 4000].max() < 1e-3 * W.max()


def test_transfer_functions_predict_engine_runs() -> None:
    """H measured with a pulse predicts an arbitrary multi-speaker run (7.1.1)."""
    room = _small_room()
    spk = [[0.6, 0.2], [0.9, 0.2]]
    pts = [[0.5, 0.8], [1.2, 0.8], [0.8, 0.6]]
    ts = measure_transfer(room, spk, pts, duration=0.1)
    assert ts.tail_db() < -60.0
    rng = np.random.default_rng(1)
    x = bandpass_pulse(ts.dt, (200.0, 1500.0), 301)
    drives = np.stack([np.convolve(x, rng.standard_normal(150)) for _ in spk])
    T = ts.steps
    rec = simulate_drives(room, spk, drives, pts, T).rec
    f = np.fft.rfftfreq(T, ts.dt)
    band = (f > 250) & (f < 1400)
    H = ts.at(f[band])
    pred = np.einsum("fms,sf->mf", H, np.fft.rfft(drives, n=T)[:, band])
    meas = np.fft.rfft(rec, n=T)[:, band]
    assert np.linalg.norm(meas - pred) / np.linalg.norm(meas) < 1e-4
    # Off-grid frequencies go through the direct DTFT; same answer.
    h1 = ts.at(f[band][:3])
    h2 = ts.at(f[band][:3] + 1e-9)
    np.testing.assert_allclose(h1, h2, rtol=1e-5)
    with pytest.raises(ValueError):
        ts.at([5000.0])


def test_torch_batch_matches_numba() -> None:
    room = Room(size=(1.0, 0.8), beta=0.3, boxes=(Box(0.1, 0.5, 0.2, 0.7),))
    spk = [[0.4, 0.1], [0.6, 0.1]]
    pts = [[0.3, 0.6], [0.7, 0.5]]
    a = measure_transfer(room, spk, pts, duration=0.03)
    b = measure_transfer(room, spk, pts, duration=0.03, engine="torch")
    assert np.abs(a.rec - b.rec).max() < 1e-4 * np.abs(a.rec).max()


def test_metrics() -> None:
    pb = np.array([1.0, 1.0j, -1.0])
    pd = 0.1 * pb
    assert acoustic_contrast_db(pb, pd) == pytest.approx(20.0)
    assert reproduction_error(pb, pb) == 0.0
    assert reproduction_error(2 * pb, pb) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    H = rng.standard_normal((4, 5, 3)) + 1j * rng.standard_normal((4, 5, 3))
    q = np.zeros((4, 3), complex)
    q[:, 1] = 3.0
    np.testing.assert_allclose(array_effort_db(q, H, ref=1), 0.0, atol=1e-12)
    assert band_contrast_db(H, 0.5 * H, q) == pytest.approx(10 * np.log10(4))
    # Band energy obeys Parseval over the full band.
    y = rng.standard_normal((2, 256))
    np.testing.assert_allclose(band_energy(y, 1.0, (0.0, 0.5)), np.sum(y**2, axis=-1), rtol=1e-12)
