"""A generative model of whole obstacle masks given the recordings (plan 6.3.4).

The aligned U-Net (``models.AlignedUNet``) gives calibrated *marginal*
probabilities :math:`q_i = P(M_i = 1 \\mid \\text{audio})`. Sampling each
pixel independently from :math:`q` gives salt-and-pepper maps, not rooms.
This module models the joint distribution of the 64x64 filled mask

.. math::
    p(\\mathbf{m} \\mid \\mathbf{c}), \\qquad
    \\mathbf{c} = (\\text{four aligned physics images}, \\text{prior logits}),

with an **absorbing-state (masked) discrete diffusion**, which is an
order-agnostic autoregressive model decoded in a few parallel steps:

* *Forward process.* Each pixel is independently hidden ("absorbed") with
  probability :math:`r \\sim \\mathcal{U}(0, 1)`; visible pixels keep their
  true value.
* *Reverse model.* :class:`MaskedDiffusionUNet` sees the conditioning, the
  visible values and the visibility mask and predicts, for every hidden
  pixel, :math:`P(m_i = 1 \\mid \\mathbf{m}_{\\text{visible}}, \\mathbf{c})`.
  It is trained with the binary cross-entropy on the hidden pixels.
* *Sampling.* All pixels start hidden. A uniformly random pixel order is
  drawn, and in each of :math:`T` steps the next block of pixels in that
  order (a cosine schedule: few pixels early, many late) is drawn from the
  model's current conditionals and becomes visible. With one pixel per step
  this is exactly the chain rule
  :math:`p(\\mathbf{m}) = \\prod_i p(m_{\\sigma(i)} \\mid m_{\\sigma(<i)})`;
  with :math:`T` steps the pixels revealed together are conditionally
  independent given the earlier ones.

With nothing visible the model is a marginal predictor, exactly the U-Net's
job. :func:`warm_start_from_aligned` copies the trained U-Net into it with
zero weights on the two new input channels, so fine-tuning starts from the
calibrated marginals and only has to learn the dependence.

The per-room *mean map* is Rao-Blackwellised: each pixel contributes the
probability it was drawn with, not the drawn bit. By the tower property
:math:`\\mathbb{E}[p_i^{\\text{reveal}}] = \\mathbb{E}[m_i]`, so it estimates
the same marginal with less variance and never returns an exact 0 or 1.

Scores (numpy, one room at a time; ``samples`` ``(N, H, W)`` bool, ``truth``
``(H, W)`` bool): :func:`energy_score` (Euclidean, i.e. square root of the
Hamming distance: strictly proper, sensitive to dependence),
:func:`jaccard_kernel_score` (the same with the Jaccard distance),
:func:`variogram_score` (neighbouring-pair disagreements, the score most
sensitive to spatial dependence), :func:`pixel_crps` (the pixelwise CRPS,
which for binary maps depends on the marginals only), plus
:func:`mean_pairwise_iou`, :func:`best_of_n_iou` and :func:`decorrelate`
(the control that keeps every pixel's marginal and destroys dependence).
See ``docs/imaging.md`` §9.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

from acoustic_system.imaging.models import CompactUNet, prior_features

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def state_channels(values: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """``(B, 2, H, W)`` state inputs: visible value in {-1, +1} (0 if hidden), visibility."""
    v = visible.to(values.dtype)
    return torch.cat([v * (2.0 * values - 1.0), v], 1)


class MaskedDiffusionUNet(nn.Module):
    """Reverse model of the absorbing-state diffusion over filled masks.

    ``forward(images, prior_logits, values, visible)``: ``images``
    ``(B, C, H, W)`` standardised pose-summed physics images,
    ``prior_logits`` ``(B, 2, H, W)`` (filled mask, illuminated boundary, as
    for ``AlignedUNet``), ``values`` and ``visible`` ``(B, 1, H, W)`` in
    {0, 1}. Returns ``(B, 1, H, W)`` logits of
    :math:`P(m_i = 1 \\mid \\mathbf{m}_{\\text{visible}}, \\mathbf{c})`, a
    residual on the filled-mask prior logit.
    """

    def __init__(self, n_images: int = 4, width: int = 12):
        super().__init__()
        self.n_images = n_images
        self.unet = CompactUNet(n_images + 2 + 2, 1, width)

    def forward(
        self,
        images: torch.Tensor,
        prior_logits: torch.Tensor,
        values: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([images, prior_features(prior_logits), state_channels(values, visible)], 1)
        return prior_logits[:, :1] + self.unet(x)


def warm_start_from_aligned(model: MaskedDiffusionUNet, aligned_state: dict) -> None:
    """Load an ``AlignedUNet`` state dict (same width) into ``model``.

    The first convolution gets zero weights on the two state channels and
    the output keeps only the filled-mask head, so with the model in eval
    mode its output equals the U-Net's filled-mask logit for any state.
    """
    new = {}
    for key, val in model.state_dict().items():
        src = aligned_state[key]  # both wrap a CompactUNet as ``self.unet``
        if key == "unet.e1.0.weight":
            w = torch.zeros_like(val)
            w[:, : src.shape[1]] = src
            new[key] = w
        elif key in ("unet.out.weight", "unet.out.bias"):
            new[key] = src[:1].clone()
        else:
            new[key] = src.clone()
    model.load_state_dict(new)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def corrupt(
    y: torch.Tensor, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward process: per room a visible fraction ``a ~ U(0, 1)``, per pixel visible w.p. ``a``.

    Returns ``(visible, a)`` with ``visible`` ``(B, 1, H, W)`` float.
    """
    b = y.shape[0]
    a = torch.rand(b, 1, 1, 1, generator=generator)
    visible = (torch.rand(y.shape, generator=generator) < a).to(y.dtype)
    return visible, a.view(-1)


def masked_bce(logits: torch.Tensor, y: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy over the hidden pixels of the batch."""
    hidden = 1.0 - visible
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (bce * hidden).sum() / hidden.sum().clamp_min(1.0)


GenBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def fit_masked(
    model: MaskedDiffusionUNet,
    batches: Callable[[int], Iterable[GenBatch]],
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    log: Callable[[str], None] | None = print,
) -> list[float]:
    """AdamW + cosine schedule on :func:`masked_bce`.

    ``batches(epoch)`` yields ``(images, prior_logits, y)`` with ``y``
    ``(B, 1, H, W)`` the true filled mask; the corruption is drawn here.
    Returns the mean training loss per epoch.
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    history = []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for images, prior_logits, y in batches(ep):
            visible, _ = corrupt(y, gen)
            loss = masked_bce(model(images, prior_logits, y, visible), y, visible)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
            n += len(y)
        sched.step()
        history.append(tot / max(n, 1))
        if log is not None:
            log(f"    epoch {ep + 1}/{epochs}: hidden-pixel BCE {history[-1]:.5f}")
    return history


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def reveal_schedule(n_pixels: int, steps: int) -> NDArray[np.int64]:
    """Cumulative number of visible pixels after each step (cosine schedule).

    ``out[s] = ceil(P (1 - cos(pi/2 (s+1)/T)))``, strictly increasing and
    ending at ``P``: a few pixels are drawn first, when the model knows
    least about the room, and most are drawn last, when they are nearly
    determined by the visible ones.
    """
    s = np.arange(1, steps + 1) / steps
    cum = np.ceil(n_pixels * (1.0 - np.cos(0.5 * np.pi * s))).astype(np.int64)
    cum[-1] = n_pixels
    for i in range(1, steps):
        cum[i] = max(cum[i], cum[i - 1] + 1)
    return np.minimum(cum, n_pixels)


@torch.no_grad()
def sample(
    model: nn.Module,
    images: torch.Tensor,
    prior_logits: torch.Tensor,
    n_samples: int,
    steps: int = 16,
    generator: torch.Generator | None = None,
    chunk: int = 128,
    logit_bias: float = 0.0,
) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    """Draw ``n_samples`` masks per room by ``steps``-step parallel decoding.

    Parameters
    ----------
    model
        A :class:`MaskedDiffusionUNet`, or any module with its call
        signature.
    images, prior_logits
        Conditioning for ``B`` rooms, as for the model.
    n_samples
        Samples per room.
    steps
        Decoding steps :math:`T`; each reveals a block of a uniformly
        random pixel order (:func:`reveal_schedule`).
    logit_bias
        A constant added to every conditional logit while sampling (0 = the
        model as trained). Parallel decoding drifts towards too much
        occupancy; a negative bias chosen on validation rooms corrects the
        mean area.

    Returns
    -------
    samples
        ``(B, N, H, W)`` bool.
    reveal_prob
        ``(B, N, H, W)`` float32: the probability each pixel was drawn
        with. Its mean over samples is the Rao-Blackwellised mean map.
    """
    model.eval()
    B, _, H, W = images.shape
    P = H * W
    cum = torch.from_numpy(reveal_schedule(P, steps))
    rows = [(b, n) for b in range(B) for n in range(n_samples)]
    out = np.zeros((B * n_samples, H, W), dtype=bool)
    rprob = np.zeros((B * n_samples, H, W), dtype=np.float32)
    for s0 in range(0, len(rows), chunk):
        idx = torch.tensor([b for b, _ in rows[s0 : s0 + chunk]])
        n = len(idx)
        img, pl = images[idx], prior_logits[idx]
        order = torch.argsort(torch.rand(n, P, generator=generator), dim=1)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(P).expand(n, P))
        step_of = torch.bucketize(rank, cum, right=True).view(n, 1, H, W)
        values = torch.zeros(n, 1, H, W)
        visible = torch.zeros(n, 1, H, W)
        rp = torch.zeros(n, 1, H, W)
        for s in range(steps):
            new = step_of == s
            p = torch.sigmoid(model(img, pl, values, visible) + logit_bias)
            draw = (torch.rand(p.shape, generator=generator) < p).to(values.dtype)
            values = torch.where(new, draw, values)
            rp = torch.where(new, p, rp)
            visible = visible + new.to(visible.dtype)
        out[s0 : s0 + n] = values[:, 0].numpy() > 0.5
        rprob[s0 : s0 + n] = rp[:, 0].numpy()
    return out.reshape(B, n_samples, H, W), rprob.reshape(B, n_samples, H, W)


def independent_bernoulli(
    prob: NDArray, n_samples: int, rng: np.random.Generator
) -> NDArray[np.bool_]:
    """``(N, H, W)`` maps with every pixel drawn independently from ``prob`` ``(H, W)``."""
    p = np.asarray(prob, dtype=np.float64)
    return rng.random((n_samples,) + p.shape) < p


def decorrelate(samples: NDArray, rng: np.random.Generator) -> NDArray[np.bool_]:
    """Permute the sample axis independently at every pixel.

    Keeps each pixel's empirical marginal (its count over the ``N``
    samples) exactly and destroys all dependence between pixels, so a score
    difference against the original set is due to the dependence alone.
    """
    s = np.asarray(samples, dtype=bool)
    n = s.shape[0]
    flat = s.reshape(n, -1)
    perm = np.argsort(rng.random(flat.shape), axis=0)
    return np.take_along_axis(flat, perm, axis=0).reshape(s.shape)


# ---------------------------------------------------------------------------
# Multi-sample scores (one room)
# ---------------------------------------------------------------------------


def _flat(samples: NDArray, truth: NDArray) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    y = np.asarray(truth, dtype=np.float64).ravel()
    return x, y


def _fair(d_xy: NDArray, d_xx: NDArray) -> float:
    """``mean d(X, y) - sum_{i != j} d(X_i, X_j) / (2 N (N - 1))`` (unbiased for N >= 2)."""
    n = len(d_xy)
    if n < 2:
        return float(np.mean(d_xy))
    return float(np.mean(d_xy) - d_xx.sum() / (2.0 * n * (n - 1)))


def hamming_matrix(x: NDArray) -> NDArray[np.float64]:
    """Pairwise Hamming distances of ``(N, P)`` binary rows."""
    x = np.asarray(x, dtype=np.float64)
    return x @ (1.0 - x).T + (1.0 - x) @ x.T


def energy_score(samples: NDArray, truth: NDArray) -> float:
    """Fair energy score with the Euclidean norm (= sqrt of the Hamming distance).

    :math:`\\mathrm{ES} = \\mathbb{E}\\lVert X - y\\rVert
    - \\tfrac12\\mathbb{E}\\lVert X - X'\\rVert`, lower is better. With the
    Euclidean norm (exponent :math:`\\beta = 1`) the score is strictly
    proper for the joint distribution; with :math:`\\beta = 2`
    (the Hamming distance itself on binary maps) it would depend on the
    marginals only.
    """
    x, y = _flat(samples, truth)
    d_xy = np.sqrt(np.abs(x - y).sum(1))
    d_xx = np.sqrt(np.maximum(hamming_matrix(x), 0.0))
    return _fair(d_xy, d_xx)


def jaccard_matrix(x: NDArray, y: NDArray | None = None) -> NDArray[np.float64]:
    """Pairwise Jaccard distances ``1 - |a & b| / |a | b|`` (0 for two empty maps)."""
    x = np.asarray(x, dtype=np.float64)
    yy = x if y is None else np.asarray(y, dtype=np.float64)
    inter = x @ yy.T
    union = x.sum(1)[:, None] + yy.sum(1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, 1.0 - inter / np.where(union > 0, union, 1.0), 0.0)


def jaccard_kernel_score(samples: NDArray, truth: NDArray) -> float:
    """Fair kernel score with the Jaccard distance (Tanimoto kernel), lower is better.

    The Tanimoto similarity is a positive-definite kernel on binary vectors,
    so the score is proper. It measures the set overlap the IoU measures.
    """
    x, y = _flat(samples, truth)
    return _fair(jaccard_matrix(x, y[None])[:, 0], jaccard_matrix(x))


def pixel_crps(samples: NDArray, truth: NDArray) -> float:
    """Pixelwise CRPS summed over pixels (fair estimator).

    For a binary pixel with ``k`` ones among ``N`` samples it is
    ``|k/N - y| - k (N - k) / (N (N - 1))``, whose expectation is the Brier
    score :math:`(q - y)^2`: it depends on the marginals only and cannot
    reward coherent samples.
    """
    x, y = _flat(samples, truth)
    n = len(x)
    k = x.sum(0)
    first = np.where(y > 0.5, 1.0 - k / n, k / n)
    second = k * (n - k) / (n * (n - 1)) if n > 1 else 0.0
    return float(np.sum(first - second))


VARIOGRAM_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (di, dj)
    for di in range(0, 3)
    for dj in range(-2, 3)
    if (di, dj) > (0, 0) and di * di + dj * dj <= 8
)


def variogram_score(
    samples: NDArray,
    truth: NDArray,
    offsets: tuple[tuple[int, int], ...] = VARIOGRAM_OFFSETS,
) -> float:
    """Variogram score of order 1 over neighbouring pixel pairs, lower is better.

    :math:`\\sum_{(a,b)} w_{ab}\\bigl(|y_a - y_b| - \\mathbb{E}|X_a - X_b|\\bigr)^2`
    over pairs at the given offsets, :math:`w_{ab} = 1/\\lVert a - b\\rVert`.
    It compares how often neighbouring pixels disagree in the samples and in
    the truth, so it is the score most sensitive to spatial coherence. It is
    proper (not strictly: it ignores the marginals).
    """
    s = np.asarray(samples, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    H, W = t.shape
    tot = 0.0
    for di, dj in offsets:
        a = (slice(0, H - di), slice(max(0, -dj), W - max(0, dj)))
        b = (slice(di, H), slice(max(0, dj), W - max(0, -dj)))
        dy = (t[a] != t[b]).astype(np.float64)
        dx = (s[:, a[0], a[1]] != s[:, b[0], b[1]]).mean(0)
        tot += float(np.sum((dy - dx) ** 2)) / math.hypot(di, dj)
    return tot


def iou_matrix(x: NDArray, eps: float = 1e-6) -> NDArray[np.float64]:
    """Pairwise IoU with the project convention ``(I + eps) / (U + eps)``."""
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    inter = x @ x.T
    union = x.sum(1)[:, None] + x.sum(1)[None, :] - inter
    return (inter + eps) / (union + eps)


def mean_pairwise_iou(samples: NDArray) -> float:
    """Mean IoU over the ``N (N - 1) / 2`` distinct sample pairs (1 = no diversity)."""
    m = iou_matrix(samples)
    n = len(m)
    return float((m.sum() - np.trace(m)) / (n * (n - 1))) if n > 1 else 1.0


def sample_ious(samples: NDArray, truth: NDArray, eps: float = 1e-6) -> NDArray[np.float64]:
    """IoU of each sample against the truth."""
    x, y = _flat(samples, truth)
    inter = x @ y
    union = x.sum(1) + y.sum() - inter
    return (inter + eps) / (union + eps)


def best_of_n_iou(samples: NDArray, truth: NDArray, n: int) -> float:
    """Expected oracle-selected IoU among ``n`` samples.

    The samples are split into ``len(samples) // n`` disjoint groups of
    ``n``; the best IoU of each group is averaged (for ``n = 1`` this is the
    mean single-sample IoU).
    """
    v = sample_ious(samples, truth)
    g = len(v) // n
    if g < 1:
        raise ValueError(f"need at least {n} samples, got {len(v)}")
    return float(v[: g * n].reshape(g, n).max(1).mean())


__all__ = [
    "MaskedDiffusionUNet",
    "warm_start_from_aligned",
    "state_channels",
    "corrupt",
    "masked_bce",
    "fit_masked",
    "reveal_schedule",
    "sample",
    "independent_bernoulli",
    "decorrelate",
    "energy_score",
    "jaccard_kernel_score",
    "pixel_crps",
    "variogram_score",
    "hamming_matrix",
    "jaccard_matrix",
    "iou_matrix",
    "mean_pairwise_iou",
    "sample_ious",
    "best_of_n_iou",
    "VARIOGRAM_OFFSETS",
]
