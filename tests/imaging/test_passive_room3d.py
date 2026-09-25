"""Passive GCC-PHAT imaging (6.6.4) and the 3D demonstration (6.6.5)."""

from __future__ import annotations

import numpy as np

from acoustic_system.imaging.passive import gcc_phat, interferometric_image, passive_images
from acoustic_system.imaging.room3d import (
    drive_offsets,
    image_room_3d,
    random_pose_3d,
    random_room_3d,
    record_3d,
    ricker,
    travel_lags_3d,
)


def test_gcc_phat_does_not_depend_on_the_source():
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal(40) * np.exp(-np.arange(40) / 8)
    h2 = np.roll(h1, 7) + 0.3 * rng.standard_normal(40) * np.exp(-np.arange(40) / 5)
    n_fft = 512
    out = []
    for seed in (1, 2):
        s = np.random.default_rng(seed).standard_normal(200)
        y1, y2 = np.convolve(s, h1), np.convolve(s, h2)
        g, band = gcc_phat(y1, y2, n_fft, band=np.ones(n_fft // 2 + 1, dtype=bool))
        out.append(g)
    np.testing.assert_allclose(out[0], out[1], atol=1e-8)


def test_gcc_phat_peaks_at_the_delay():
    rng = np.random.default_rng(3)
    s = rng.standard_normal(300)
    y2 = s
    y1 = np.concatenate([np.zeros(9), s[:-9]])  # mic 1 hears it 9 samples later
    g, _ = gcc_phat(y1, y2, 1024)
    assert int(np.argmax(g)) - 512 == 9


def test_interferometric_image_hits_the_scatterer_delay():
    """A GCC envelope peaked at the direct-scattered lag of x lights up x."""
    grid = (32, 32)
    src, mics = np.array([6, 6]), np.array([[6, 20], [20, 6]])
    x = np.array([24, 24])
    t_s1 = (np.hypot(*(x - src)) + np.hypot(*(x - mics[0]))) / 0.5
    t_d2 = np.hypot(*(mics[1] - src)) / 0.5
    n_fft = 512
    tr = np.exp(-0.5 * ((np.arange(n_fft) - (n_fft // 2 + t_s1 - t_d2)) / 1.0) ** 2)
    img = interferometric_image(grid, src, mics, tr[None], 0.5, n_fft)
    assert img[tuple(x)] > 0.95 * img.max()


def test_passive_images_run_without_the_source(v2_drive):
    grid = (24, 24)
    rng = np.random.default_rng(0)
    rec = rng.standard_normal((1, 2, 120))
    raw, sub = passive_images(grid, np.array([[5, 5]]), np.array([[[5, 15], [15, 5]]]), rec, 0.5)
    assert raw.shape == grid and sub.shape == grid
    assert np.isfinite(raw).all() and np.isfinite(sub).all()


def test_ricker_offsets():
    d = ricker(200, 0.5, 0.12)
    peak, onset = drive_offsets(d)
    assert abs(peak - (1.2 / 0.12) / 0.5) <= 1
    assert 0 < onset < peak


def test_3d_carving_never_clears_the_obstacle():
    """Voxels inside the first-arrival ellipsoid are free, so the scatterer stays uncarved."""
    n, T = 18, 110
    shape = (n, n, n)
    mask = np.zeros(shape, dtype=bool)
    mask[11:14, 11:14, 8:11] = True
    rng = np.random.default_rng(0)
    src, mics = random_pose_3d(mask, rng, half_width=2)
    drive = 5.0 * ricker(T, 0.5, 0.15)
    peak, onset = drive_offsets(drive)
    res = record_3d(mask, src, mics, drive, shape) - record_3d(None, src, mics, drive, shape)
    im = image_room_3d(shape, src[None], mics[None], res[None], 0.5, peak, onset)
    assert im.carving.max() > 0
    # At most the voxel nearest the array is touched (the p = 0 surface sits on the
    # cell face and the FDTD precursor leads the wavefront by a few steps).
    assert (im.carving[mask] > 0).mean() <= 0.1
    assert (im.carving[~mask] > 0).mean() > 0.2
    assert np.isfinite(im.backprojection).all()
    assert travel_lags_3d(shape, src, mics[0], 0.5)[tuple(src)] > 0


def test_random_room_and_pose_3d():
    rng = np.random.default_rng(5)
    mask = random_room_3d(24, rng)
    assert mask.any() and not mask[:2].any() and not mask[-2:].any()
    s, m = random_pose_3d(mask, rng)
    assert m.shape == (4, 3)
    assert not mask[tuple(np.concatenate([s[None], m]).T)].any()
