"""Learned imaging models: invariances, migration and calibration (plan 6.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from acoustic_system.imaging import models as M
from acoustic_system.imaging.backprojection import backproject, travel_lags

torch.set_num_threads(1)
G = (32, 32)


def test_travel_lag_maps_match_numpy():
    src = np.array([[5.0, 7.0], [20.0, 11.0]])
    mics = np.array([[[9.0, 3.0], [1.0, 30.0]], [[25.0, 25.0], [12.0, 4.0]]])
    lags = M.travel_lag_maps(
        torch.tensor(src[None]), torch.tensor(mics[None]), G, 0.5, lag_offset=-2.0
    ).numpy()
    for k in range(2):
        for m in range(2):
            ref, _, _ = travel_lags(G, src[k], mics[k, m], 0.5)
            np.testing.assert_allclose(lags[0, k, m].reshape(G), ref - 2.0, atol=1e-9)


def test_migrate_equals_backprojection():
    """The differentiable delay-and-sum reproduces ``backprojection.backproject``."""
    rng = np.random.default_rng(0)
    src = np.array([[5, 7], [20, 11]])
    mics = np.array([[[9, 3], [1, 30]], [[25, 25], [12, 4]]])
    h = rng.standard_normal((2, 2, 150))
    ref = backproject(
        G,
        src,
        mics,
        h,
        0.5,
        kind="signed",
        spreading=False,
        normalise_traces=False,
        lag_offset=-2.0,
    )
    lags = M.travel_lag_maps(
        torch.tensor(src[None], dtype=torch.float64),
        torch.tensor(mics[None], dtype=torch.float64),
        G,
        0.5,
        lag_offset=-2.0,
    )
    out = M.migrate(torch.tensor(-h[None, :, :, None, :]), lags).sum(dim=(1, 2))
    np.testing.assert_allclose(out[0, 0].numpy().reshape(G), ref, atol=1e-9)


def test_untrained_models_return_the_prior():
    prior = torch.randn(2, 2, *G)
    x = torch.randn(2, 4, *G)
    for model, args in (
        (M.AlignedUNet(4, 4), (x, prior)),
        (M.PoseSetNet(7, 4, 4), (torch.randn(2, 3, 7, *G), prior)),
        (
            M.IRMigrationNet(G, 0.5, 4, 4),
            (
                torch.randn(2, 3, 2, 2, 64),
                torch.rand(2, 3, 2) * 30,
                torch.rand(2, 3, 2, 2) * 30,
                prior,
            ),
        ),
        (
            M.IRGlobalNet(2, G, 16, 4),
            (
                torch.randn(2, 3, 2, 2, 64),
                torch.rand(2, 3, 2) * 30,
                torch.rand(2, 3, 2, 2) * 30,
                prior,
            ),
        ),
    ):
        model.eval()
        with torch.no_grad():
            out = model(*args)
        assert out.shape == prior.shape
        torch.testing.assert_close(out, prior)


def test_pose_set_net_is_permutation_invariant_for_any_k():
    torch.manual_seed(0)
    net = M.PoseSetNet(7, 4, 4)
    with torch.no_grad():  # make the output non-trivial
        net.unet.out.weight.normal_()
    net.eval()
    prior = torch.zeros(1, 2, *G)
    x = torch.randn(1, 5, 7, *G)
    with torch.no_grad():
        a = net(x, prior)
        b = net(x[:, [3, 0, 4, 2, 1]], prior)
        c = net(x[:, :1], prior)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert c.shape == a.shape and not torch.allclose(a, c)


@pytest.mark.parametrize("t", range(8))
def test_d4_grid_and_points_agree(t):
    n = 16
    img = np.zeros((n, n))
    p = np.array([3, 11])
    img[p[0], p[1]] = 1.0
    q = M.d4_points(p, t, n)
    out = M.d4_grid(img, t)
    assert out[q[0], q[1]] == 1.0 and out.sum() == 1.0


def test_temperature_and_reliability():
    rng = np.random.default_rng(1)
    z = rng.normal(0.0, 3.0, 200_000)
    y = rng.random(z.size) < 1.0 / (1.0 + np.exp(-z / 2.0))
    assert M.fit_temperature(z, y) == pytest.approx(2.0, rel=0.05)
    p_cal = 1.0 / (1.0 + np.exp(-z / 2.0))
    *_, ece_cal = M.reliability(p_cal, y)
    *_, ece_over = M.reliability(1.0 / (1.0 + np.exp(-z)), y)
    assert ece_cal < 0.01 < ece_over


def test_fit_reduces_loss_and_mc_dropout_predicts():
    torch.manual_seed(0)
    net = M.AlignedUNet(1, 4, dropout=0.2)
    x = torch.randn(8, 1, *G)
    y = (x[:, :1] > 0.5).float().repeat(1, 2, 1, 1)
    prior = torch.zeros(8, 2, *G)

    def batches(_ep):
        yield (x, prior), y

    hist = M.fit(net, batches, 15, lr=1e-2, log=None)
    assert hist[-1] < hist[0]
    z = M.predict_logits([net], [(x, prior)], mc_samples=3)
    assert z.shape == (8, 2, *G) and np.isfinite(z).all()
