"""Generative mask model and multi-sample scores (plan 6.3.4)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from acoustic_system.imaging import generative as G
from acoustic_system.imaging import models as M

torch.set_num_threads(1)
H = 16


def _cond(b: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(b, 4, H, H, generator=g)
    prior = torch.randn(b, 2, H, H, generator=g) - 2.0
    return images, prior


def _randomise_bn(model: torch.nn.Module, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv2d):
            mod.weight.data = 0.1 * torch.randn(mod.weight.shape, generator=g)
        if isinstance(mod, torch.nn.BatchNorm2d):
            assert mod.running_mean is not None and mod.running_var is not None
            mod.running_mean.uniform_(-0.1, 0.1, generator=g)
            mod.running_var.uniform_(0.5, 1.5, generator=g)


def test_warm_start_reproduces_the_unet_marginal():
    """With the U-Net's weights and zeroed state channels, the output is the U-Net's fill logit."""
    unet = M.AlignedUNet(4, 8)
    _randomise_bn(unet)
    unet.eval()
    gen = G.MaskedDiffusionUNet(4, 8)
    G.warm_start_from_aligned(gen, unet.state_dict())
    gen.eval()
    images, prior = _cond(3)
    values = (torch.rand(3, 1, H, H) < 0.3).float()
    visible = (torch.rand(3, 1, H, H) < 0.5).float()
    with torch.no_grad():
        ref = unet(images, prior)[:, :1]
        out = gen(images, prior, values, visible)
    assert float(ref.abs().max()) > 0.1  # the check is not trivially zero
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_untrained_model_returns_the_prior():
    gen = G.MaskedDiffusionUNet(4, 8).eval()
    images, prior = _cond(2)
    with torch.no_grad():
        out = gen(images, prior, torch.zeros(2, 1, H, H), torch.zeros(2, 1, H, H))
    torch.testing.assert_close(out, prior[:, :1])


def test_state_channels_and_masked_loss():
    y = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    vis = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    s = G.state_channels(y, vis)
    assert s[0, 0].tolist() == [[1.0, -1.0], [0.0, 0.0]]
    assert s[0, 1].tolist() == vis[0, 0].tolist()
    # Wrong predictions on the visible pixels do not count.
    logits = torch.tensor([[[[-20.0, 20.0], [20.0, -20.0]]]])
    assert float(G.masked_bce(logits, y, vis)) < 1e-6
    assert float(G.masked_bce(logits, y, 1 - vis)) > 10.0


def test_corrupt_fraction_is_uniform_per_room():
    y = torch.zeros(400, 1, H, H)
    vis, a = G.corrupt(y, torch.Generator().manual_seed(0))
    frac = vis.mean((1, 2, 3))
    assert float((frac - a).abs().max()) < 0.15
    assert 0.4 < float(a.mean()) < 0.6


@pytest.mark.parametrize("steps", [1, 4, 16])
def test_reveal_schedule(steps):
    cum = G.reveal_schedule(256, steps)
    assert len(cum) == steps and cum[-1] == 256 and cum[0] >= 1
    assert np.all(np.diff(cum) > 0)
    if steps > 1:
        assert cum[0] < 256 / steps  # few pixels first


class _Oracle(torch.nn.Module):
    """Returns fixed logits regardless of the state."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    def forward(self, images, prior_logits, values, visible):
        return self.logits.expand(len(images), -1, -1, -1)


class _CopyNeighbour(torch.nn.Module):
    """A coherent toy: the whole map is one Bernoulli(1/2) draw.

    The first pixel gets logit 0; once any pixel is visible, every hidden
    pixel copies the visible value.
    """

    def forward(self, images, prior_logits, values, visible):
        n_vis = visible.sum((1, 2, 3), keepdim=True)
        mean_val = (values * visible).sum((1, 2, 3), keepdim=True) / n_vis.clamp_min(1)
        logit = torch.where(n_vis > 0, 40.0 * (mean_val - 0.5), torch.zeros_like(mean_val))
        return logit.expand_as(values)


def test_sampler_follows_a_state_independent_model():
    torch.manual_seed(0)
    p = torch.rand(1, 1, H, H)
    model = _Oracle(torch.logit(p))
    images, prior = _cond(2)
    s, rp = G.sample(model, images, prior, 200, steps=5, generator=torch.Generator().manual_seed(1))
    assert s.shape == (2, 200, H, H) and s.dtype == bool
    # The Rao-Blackwellised mean is exact here; the empirical mean agrees within noise.
    np.testing.assert_allclose(rp.mean(1), np.broadcast_to(p[0, 0].numpy(), (2, H, H)), atol=1e-5)
    assert np.abs(s.mean(1) - p[0, 0].numpy()).max() < 0.15
    # Deterministic given the generator.
    s2, _ = G.sample(model, images, prior, 200, steps=5, generator=torch.Generator().manual_seed(1))
    assert np.array_equal(s, s2)


def test_sampler_logit_bias_shifts_every_conditional():
    p = torch.full((1, 1, H, H), 0.5)
    images, prior = _cond(1)
    _, rp = G.sample(_Oracle(torch.logit(p)), images, prior, 4, steps=3, logit_bias=-1.0)
    np.testing.assert_allclose(rp, 1.0 / (1.0 + np.e), atol=1e-6)


def test_sampler_produces_coherent_maps():
    images, prior = _cond(1)
    s, rp = G.sample(
        _CopyNeighbour(), images, prior, 64, steps=24, generator=torch.Generator().manual_seed(0)
    )
    flat = s.reshape(64, -1)
    assert np.all(flat.all(1) | ~flat.any(1))  # each sample is all ones or all zeros
    assert 10 < flat.all(1).sum() < 54
    np.testing.assert_allclose(rp.mean(), 0.5, atol=0.1)


def test_fit_masked_reduces_loss():
    torch.manual_seed(0)
    images, prior = _cond(8)
    y = (images[:, :1] > 0.5).float()
    model = G.MaskedDiffusionUNet(4, 4)

    def batches(ep):
        for s in range(0, 8, 4):
            yield images[s : s + 4], prior[s : s + 4], y[s : s + 4]

    hist = G.fit_masked(model, batches, 6, lr=1e-2, log=None)
    assert hist[-1] < hist[0]


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def test_scores_vanish_for_a_perfect_point_forecast():
    rng = np.random.default_rng(0)
    t = rng.random((8, 8)) < 0.3
    s = np.repeat(t[None], 5, 0)
    assert G.energy_score(s, t) == pytest.approx(0.0)
    assert G.jaccard_kernel_score(s, t) == pytest.approx(0.0)
    assert G.pixel_crps(s, t) == pytest.approx(0.0)
    assert G.variogram_score(s, t) == pytest.approx(0.0)
    assert G.mean_pairwise_iou(s) == pytest.approx(1.0)
    assert G.best_of_n_iou(s, t, 5) == pytest.approx(1.0)


def test_energy_score_matches_brute_force():
    rng = np.random.default_rng(1)
    s = rng.random((4, 3, 3)) < 0.5
    t = rng.random((3, 3)) < 0.5
    d = lambda a, b: np.sqrt(np.sum(a != b))  # noqa: E731
    ref = np.mean([d(x, t) for x in s]) - sum(d(a, b) for a, b in itertools.permutations(s, 2)) / (
        2 * 4 * 3
    )
    assert G.energy_score(s, t) == pytest.approx(ref)


def test_pixel_crps_is_the_fair_brier_score():
    rng = np.random.default_rng(2)
    s = rng.random((6, 4, 4)) < 0.4
    t = rng.random((4, 4)) < 0.4
    ref = 0.0
    for i in range(4):
        for j in range(4):
            x = s[:, i, j].astype(float)
            y = float(t[i, j])
            ref += np.mean(np.abs(x - y)) - np.abs(x[:, None] - x[None]).sum() / (2 * 6 * 5)
    assert G.pixel_crps(s, t) == pytest.approx(ref)


def test_variogram_matches_brute_force():
    rng = np.random.default_rng(3)
    s = rng.random((5, 6, 6)) < 0.5
    t = rng.random((6, 6)) < 0.5
    ref = 0.0
    for i, j in itertools.product(range(6), range(6)):
        for di, dj in G.VARIOGRAM_OFFSETS:
            a, b = (i, j), (i + di, j + dj)
            if not (0 <= b[0] < 6 and 0 <= b[1] < 6):
                continue
            dy = float(t[a] != t[b])
            dx = np.mean(s[:, a[0], a[1]] != s[:, b[0], b[1]])
            ref += (dy - dx) ** 2 / np.hypot(di, dj)
    assert G.variogram_score(s, t) == pytest.approx(ref)
    assert len(G.VARIOGRAM_OFFSETS) == 12


def test_coherent_forecast_beats_decorrelated_one():
    """Truth is all-ones or all-zeros (p = 1/2). Coherent samples should score better
    on the energy, Jaccard and variogram scores; the pixel CRPS cannot tell them apart."""
    rng = np.random.default_rng(4)
    n, side, rooms = 32, 6, 60
    scores = {k: [0.0, 0.0] for k in ("es", "js", "vs", "crps")}
    for _ in range(rooms):
        t = np.full((side, side), rng.random() < 0.5)
        coh = np.repeat((rng.random(n) < 0.5)[:, None, None], side, 1).repeat(side, 2)
        dec = G.decorrelate(coh, rng)
        assert np.array_equal(coh.sum(0), dec.sum(0))  # same per-pixel marginals
        for i, s in enumerate((coh, dec)):
            scores["es"][i] += G.energy_score(s, t)
            scores["js"][i] += G.jaccard_kernel_score(s, t)
            scores["vs"][i] += G.variogram_score(s, t)
            scores["crps"][i] += G.pixel_crps(s, t)
    for k in ("es", "js", "vs"):
        assert scores[k][0] < scores[k][1], k
    assert scores["crps"][0] == pytest.approx(scores["crps"][1])


def test_diversity_and_best_of_n():
    rng = np.random.default_rng(5)
    t = rng.random((8, 8)) < 0.3
    s = rng.random((32, 8, 8)) < 0.3
    b = [G.best_of_n_iou(s, t, n) for n in (1, 8, 32)]
    assert b[0] <= b[1] <= b[2]
    assert b[0] == pytest.approx(G.sample_ious(s, t).mean())
    assert 0.0 < G.mean_pairwise_iou(s) < 0.5
    ind = G.independent_bernoulli(np.full((8, 8), 0.25), 4000, rng)
    assert abs(ind.mean() - 0.25) < 0.01
    with pytest.raises(ValueError):
        G.best_of_n_iou(s[:4], t, 8)


def test_jaccard_matrix_handles_empty_maps():
    x = np.zeros((2, 9), dtype=bool)
    x[1, :3] = True
    d = G.jaccard_matrix(x)
    assert d[0, 0] == 0.0 and d[0, 1] == 1.0 and d[1, 1] == 0.0
