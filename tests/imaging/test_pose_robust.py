"""Rigid pose error, rigid refinement and autofocus (``imaging/pose_robust.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.imaging import pose_robust as PR
from acoustic_system.imaging.ir import empty_room_response
from acoustic_system.imaging.pipeline import ArchiveRoom

N = 32
T = 160


def _pairwise(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.linalg.norm(p[:, None] - p[None], axis=-1)


def _room(drive: np.ndarray, sources, mics) -> ArchiveRoom:
    """An empty-box room whose recordings are the engine's, at the given (true) cells."""
    sources = np.asarray(sources, dtype=np.int64)
    mics = np.asarray(mics, dtype=np.int64)
    rec = np.stack(
        [empty_room_response((N, N), sources[k], drive, mics[k])[0] for k in range(len(sources))]
    )
    return ArchiveRoom(np.zeros((N, N), bool), sources, mics, rec, drive, 0.5)


@pytest.fixture(scope="module")
def drive(v2_drive):
    return v2_drive[:T]


def test_rigid_transform_is_rigid():
    p = np.array([[10.0, 4.0], [3.0, 9.0], [7.0, 20.0]])
    q = PR.rigid_transform(p, (1.5, -2.0), 17.0)
    assert np.allclose(_pairwise(p), _pairwise(q))
    assert np.allclose(q.mean(0), p.mean(0) + (1.5, -2.0))
    assert np.allclose(PR.rigid_transform(p, (0, 0), 0.0), p)


def test_perturb_poses_rigid(drive):
    rng = np.random.default_rng(0)
    room = ArchiveRoom(
        np.zeros((64, 64), bool),
        np.array([[20, 20], [40, 30]]),
        np.array([[[25, 30], [25, 42]], [[30, 10], [42, 12]]]),
        np.zeros((2, 2, 8)),
        drive[:8],
        0.5,
    )
    same, err0, _ = PR.perturb_poses_rigid(room, 0.0, 0.0, rng)
    assert np.all(err0 == 0) and np.array_equal(same.sources, room.sources)
    moved, err, params = PR.perturb_poses_rigid(room, 1.0, 3.0, rng)
    assert err.shape == (2, 3, 2) and params.shape == (2, 3)
    assert moved.recordings is room.recordings  # recordings stay at the true cells
    for k in range(2):
        a = np.concatenate([room.sources[k][None], room.mics[k]])
        b = np.concatenate([moved.sources[k][None], moved.mics[k]])
        # rigid up to rounding: each distance changes by at most one cell diagonal
        assert np.abs(_pairwise(a) - _pairwise(b)).max() <= np.sqrt(2) + 1e-9


def test_rigid_candidates_identity_first_and_unique():
    devs = np.array([[10, 10], [14, 20], [20, 16]])
    c = PR.rigid_candidates(devs, 1, [0.0], (N, N))
    assert len(c) == 9
    assert np.array_equal(c.devices[0], devs) and np.allclose(c.params[0], 0)
    c2 = PR.rigid_candidates(devs, 2, PR.candidate_thetas(3.0), (N, N))
    keys = {d.tobytes() for d in c2.devices}
    assert len(keys) == len(c2)
    assert c2.devices.min() >= 1 and c2.devices.max() <= N - 2
    edge = PR.rigid_candidates(np.array([[1, 5], [3, 9], [5, 5]]), 1, [0.0], (N, N))
    assert edge.devices.min() >= 1 and len(edge) == 6


def test_huber_cost():
    r = np.array([0.1, -0.5, 2.0])
    d = 0.5
    expected = 0.5 * 0.1**2 + 0.5 * 0.5**2 + d * (2.0 - 0.5 * d)
    assert np.isclose(PR.huber_cost(r, d), expected)


def test_rigid_refinement_recovers_a_translated_pose(drive):
    true_s = np.array([[9, 12]])
    true_m = np.array([[[15, 7], [20, 16]]])
    room = _room(drive, true_s, true_m)
    assumed = ArchiveRoom(
        room.mask, true_s + (1, -1), true_m + np.array([1, -1]), room.recordings, drive, 0.5
    )
    for loss in ("l2", "huber", "quiet"):
        fixed, idx = PR.refine_room_rigid(assumed, 1, [0.0], loss=loss)
        assert np.array_equal(fixed.sources, true_s), loss
        assert np.array_equal(fixed.mics, true_m), loss
        assert PR.device_error(fixed, room).max() == 0


def test_evaluate_candidates_shapes_and_zero_misfit(drive):
    room = _room(drive, [[9, 12]], [[[15, 7], [20, 16]]])
    cs = PR.analyse_room(room, 1, [0.0])
    c = cs[0]
    assert c.focus is not None and c.focus.shape == (len(c.candidates), 2, N, N)
    assert c.misfit["l2"][0] < 1e-8 * c.misfit["l2"][1:].min() + 1e-12
    rel = PR.relative_cost(c, "quiet")
    assert rel[0] == 0 and rel[1:].min() > 0
    f = PR.focus_features(c.focus, (0, 1), 1.0)
    norms = np.linalg.norm(f, axis=1)
    assert norms[0] < 1e-6  # the exact pose leaves no scattered residual in an empty box
    assert np.allclose(norms[1:], 1.0, atol=1e-6)


def _blob(center, n=24, s=1.5):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.exp(-((ii - center[0]) ** 2 + (jj - center[1]) ** 2) / (2 * s**2))


def test_autofocus_aligns_shifted_images():
    """Each pose sees the same two scatterers, displaced by its (unknown) pose error.

    Candidate c of pose k undoes shift ``shifts[c]``; the coherent choice
    undoes each pose's own error.
    """
    scat = [(8, 8), (15, 14)]
    shifts = [(0, 0), (2, 0), (0, 2), (-2, 0), (0, -2), (2, 2)]
    errors = [(2, 0), (0, -2), (0, 0), (-2, 0)]

    def image(offset):
        return sum(_blob((a + offset[0], b + offset[1])) for a, b in scat)

    feats = []
    for e in errors:
        imgs = np.stack([image((e[0] - s[0], e[1] - s[1]))[None] for s in shifts])  # (C, 1, H, W)
        feats.append(PR.focus_features(imgs, (0,), 0.0))
    idx = PR.autofocus_select(feats, metric="coherence")
    assert [shifts[i] for i in idx] == errors
    idx_e = PR.autofocus_select(feats, metric="entropy")
    assert [shifts[i] for i in idx_e] == errors
    # a strong prior on small corrections keeps every pose at the identity
    pens = [np.array([0.0] + [1.0] * (len(shifts) - 1))] * len(errors)
    assert PR.autofocus_select(feats, pens, metric="coherence", prior_weight=100.0) == [0] * 4
