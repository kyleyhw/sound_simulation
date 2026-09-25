"""Per-pose images, geometry channels and pose perturbation (plan 6.3, 6.6.1)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.imaging.ir import SampledWaveform, TikhonovDeconvolver
from acoustic_system.imaging.pipeline import ArchiveRoom, compute_room_images
from acoustic_system.imaging.pose_images import (
    POSE_CHANNELS,
    aggregate,
    compute_pose_images,
    geometry_channels,
    perturb_poses,
)
from acoustic_system.simulation.setup import Driver
from acoustic_system.simulation.simulate import Simulate

N = 40
T = 200


def _record(mask, src, mics, drive):
    sim = Simulate((N, N))
    sim.set_obstacle_mask(mask)
    sim.set_drivers([Driver(tuple(int(c) for c in src), SampledWaveform(drive, sim.timestep))])
    rec = np.zeros((len(mics), len(drive)))
    for k in range(len(drive)):
        sim.step()
        for i, m in enumerate(mics):
            rec[i, k] = sim.p[tuple(m)]
    return rec


@pytest.fixture(scope="module")
def small_room(v2_drive) -> ArchiveRoom:
    drive = v2_drive[:T]
    mask = np.zeros((N, N), dtype=bool)
    mask[24:28, 14:20] = True
    sources = np.array([[10, 10], [12, 30]])
    mics = np.array([[[8, 16], [14, 20]], [[8, 26], [16, 34]]])
    rec = np.stack([_record(mask, sources[k], mics[k], drive) for k in range(2)])
    return ArchiveRoom(mask, sources, mics, rec, drive, 0.5)


def test_pose_images_sum_to_room_images(small_room):
    dec = TikhonovDeconvolver(small_room.drive, T, lam=1e-2)
    pose = compute_pose_images(small_room, dec)
    room = compute_room_images(small_room, dec)
    agg = aggregate(pose.images)
    assert pose.images.shape == (2, len(POSE_CHANNELS), N, N)
    for name in POSE_CHANNELS:
        ref = getattr(room, name)
        np.testing.assert_allclose(agg[name], ref, atol=1e-9 * (1 + np.abs(ref).max()))
    np.testing.assert_allclose(pose.residual, room.residual)
    # The first pose alone is the K = 1 room image.
    one = aggregate(pose.images, 1)
    np.testing.assert_allclose(one["carving"], pose.images[0, 2])


def test_incident_fields_are_reused(small_room):
    from acoustic_system.imaging.ir import empty_room_response

    dec = TikhonovDeconvolver(small_room.drive, T, lam=1e-2)
    inc = []
    for k in range(2):
        y0, P = empty_room_response(
            (N, N), small_room.sources[k], small_room.drive, small_room.mics[k], return_field=True
        )
        assert P is not None
        inc.append((y0, P))
    a = compute_pose_images(small_room, dec, incident=inc).images
    b = compute_pose_images(small_room, dec).images
    np.testing.assert_allclose(a, b)


def test_geometry_channels():
    g = geometry_channels((N, N), np.array([10, 12]), np.array([[20, 12], [10, 30]]), 0.5)
    assert g.shape == (3, N, N)
    assert g[0, 10, 12] == 0.0
    # Bistatic lag at the first mic is the direct distance / dt, normalised by 4N.
    assert g[1, 20, 12] == pytest.approx((10 / 0.5) / (4 * N), rel=1e-6)
    assert np.all(g >= 0)


def test_perturb_poses(small_room):
    rng = np.random.default_rng(0)
    same, err0 = perturb_poses(small_room, 0.0, rng)
    np.testing.assert_array_equal(same.sources, small_room.sources)
    assert not err0.any()
    moved, err = perturb_poses(small_room, 3.0, rng)
    assert err.shape == (2, 3, 2) and err.dtype.kind == "i"
    devs = np.concatenate([moved.sources[:, None], moved.mics], 1)
    assert devs.min() >= 1 and devs.max() <= N - 2
    np.testing.assert_array_equal(moved.recordings, small_room.recordings)
    np.testing.assert_array_equal(
        devs - err, np.concatenate([small_room.sources[:, None], small_room.mics], 1)
    )
