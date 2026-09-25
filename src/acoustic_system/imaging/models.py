"""Small learned models on top of the physics images (plan Task 6.3).

The physics baselines (``pipeline.py``, ``docs/imaging.md`` §7) fuse four
standardised images with the no-audio prior by a five-weight logistic
regression. The models here keep that structure, a residual on the prior
logit,

.. math::
    \\operatorname{logit} q(\\mathbf{x}) = \\operatorname{logit}\\hat\\pi(\\mathbf{x})
        + f_\\theta(\\text{inputs})(\\mathbf{x}),

with :math:`f_\\theta` initialised to zero (the last layer is zeroed), so
training starts exactly at the prior. Two output heads share the network:
the filled obstacle mask and the illuminated boundary (plan 6.1.1), each
with its own prior.

Models
------
:class:`AlignedUNet` (6.3.1)
    A compact U-Net on the grid-aligned images summed over poses.
:class:`IRMigrationNet` (6.3.2)
    A per-trace 1D encoder on the recovered impulse responses, whose
    output channels are migrated to the grid by a differentiable
    delay-and-sum (:func:`migrate`), then the same U-Net. It learns the
    detection filter that ``backprojection.py`` fixes by hand.
:class:`IRGlobalNet` (6.3.2, control)
    The Phase 2 layout with impulse responses in place of spectrograms:
    per-pose 1D encoder to a vector, pose coordinates appended, mean over
    poses, then a convolutional decoder. No spatial alignment.
:class:`PoseSetNet` (6.3.3)
    Per-pose images plus geometry channels, a shared per-pose encoder
    :math:`\\phi`, and permutation-invariant pooling over poses (per-pixel
    attention, mean and max), then the U-Net. Works for any :math:`K`.

Uncertainty (6.3.4): :func:`fit_temperature` and :func:`predict_proba`
(deep ensembles and Monte Carlo dropout).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from torch import nn

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convolutions with batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class CompactUNet(nn.Module):
    """Three-level U-Net (widths ``w, 2w, 4w``; bottleneck at ``H/8``).

    The output 1x1 convolution is zero-initialised, so an untrained network
    returns zeros (the prior, once added to the prior logit). ``dropout``
    applies ``Dropout2d`` at the bottleneck and the two coarse decoder
    levels, which is what Monte Carlo dropout samples.
    """

    def __init__(self, c_in: int, c_out: int = 2, width: int = 12, dropout: float = 0.0):
        super().__init__()
        w = width
        self.e1 = conv_block(c_in, w)
        self.e2 = conv_block(w, 2 * w)
        self.e3 = conv_block(2 * w, 4 * w)
        self.bott = conv_block(4 * w, 4 * w)
        self.d3 = conv_block(8 * w, 2 * w)
        self.d2 = conv_block(4 * w, w)
        self.d1 = conv_block(2 * w, w)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Conv2d(w, c_out, 1)
        with torch.no_grad():
            self.out.weight.zero_()
            assert self.out.bias is not None
            self.out.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.drop(self.bott(F.max_pool2d(e3, 2)))
        up = F.interpolate(b, scale_factor=2.0, mode="nearest")
        d3 = self.drop(self.d3(torch.cat([up, e3], 1)))
        up = F.interpolate(d3, scale_factor=2.0, mode="nearest")
        d2 = self.drop(self.d2(torch.cat([up, e2], 1)))
        up = F.interpolate(d2, scale_factor=2.0, mode="nearest")
        d1 = self.d1(torch.cat([up, e1], 1))
        return self.out(d1)


def prior_features(prior_logits: torch.Tensor) -> torch.Tensor:
    """Prior logits scaled to order one for use as network inputs."""
    return prior_logits / 5.0


# ---------------------------------------------------------------------------
# 6.3.1 Grid-aligned images
# ---------------------------------------------------------------------------


class AlignedUNet(nn.Module):
    """U-Net on standardised pose-summed physics images plus the prior logits.

    ``forward(images, prior_logits)`` with ``images`` ``(B, C, H, W)`` and
    ``prior_logits`` ``(B, 2, H, W)`` (filled mask, illuminated boundary)
    returns ``(B, 2, H, W)`` output logits.
    """

    def __init__(self, n_images: int = 4, width: int = 12, dropout: float = 0.0):
        super().__init__()
        self.unet = CompactUNet(n_images + 2, 2, width, dropout)

    def forward(self, images: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
        return prior_logits + self.unet(torch.cat([images, prior_features(prior_logits)], 1))


# ---------------------------------------------------------------------------
# 6.3.2 Impulse responses
# ---------------------------------------------------------------------------


def travel_lag_maps(
    sources: torch.Tensor,
    mics: torch.Tensor,
    grid_shape: tuple[int, int],
    dt: float,
    lag_offset: float = 0.0,
) -> torch.Tensor:
    """Bistatic travel lags ``(B, K, M, H*W)`` for sources ``(B, K, 2)``, mics ``(B, K, M, 2)``.

    :math:`t_{km}(\\mathbf{x}) = (\\lVert\\mathbf{x}-\\mathbf{s}_k\\rVert +
    \\lVert\\mathbf{x}-\\mathbf{m}_{km}\\rVert)/(c\\Delta t)` with :math:`c = 1`,
    in steps, plus ``lag_offset`` (as in ``backprojection.backproject``).
    """
    ii, jj = torch.meshgrid(
        torch.arange(grid_shape[0], dtype=sources.dtype),
        torch.arange(grid_shape[1], dtype=sources.dtype),
        indexing="ij",
    )
    pix = torch.stack([ii.reshape(-1), jj.reshape(-1)], -1)  # (P, 2)
    rs = torch.linalg.norm(pix[None, None] - sources[:, :, None, :], dim=-1)  # (B, K, P)
    rm = torch.linalg.norm(pix[None, None, None] - mics[..., None, :], dim=-1)  # (B, K, M, P)
    return (rs[:, :, None, :] + rm) / dt + lag_offset


def migrate(traces: torch.Tensor, lags: torch.Tensor) -> torch.Tensor:
    """Differentiable delay-and-sum of multichannel traces onto pixels.

    Parameters
    ----------
    traces
        ``(B, K, M, C, L)`` filtered traces.
    lags
        ``(B, K, M, P)`` fractional travel lags (steps).

    Returns
    -------
    ``(B, K, M, C, P)``: :math:`a_{kmc}(t_{km}(\\mathbf{x}))` by linear
    interpolation, zero outside ``[0, L-1)``, the same rule as
    ``backprojection.sample_trace``.
    """
    B, K, M, C, L = traces.shape
    P = lags.shape[-1]
    i0 = torch.floor(lags)
    frac = (lags - i0).unsqueeze(3)
    ok = ((i0 >= 0) & (i0 + 1 < L)).unsqueeze(3).to(traces.dtype)
    i0c = i0.clamp(0, L - 2).long().unsqueeze(3).expand(B, K, M, C, P)
    a = torch.gather(traces, 4, i0c)
    b = torch.gather(traces, 4, i0c + 1)
    return ((1 - frac) * a + frac * b) * ok


class TraceEncoder(nn.Module):
    """Shared 1D convolutional filter bank for (normalised IR, envelope) pairs."""

    def __init__(self, c_in: int = 2, c_out: int = 8, hidden: int = 16, kernel: int = 9):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, hidden, kernel, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, c_out, kernel, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IRMigrationNet(nn.Module):
    """Learned filters on impulse responses, migrated to the grid, then a U-Net.

    ``forward(traces, sources, mics, prior_logits)``: ``traces`` is
    ``(B, K, M, 2, L)`` (IR and its envelope, each divided by the trace
    RMS), ``sources`` ``(B, K, 2)`` and ``mics`` ``(B, K, M, 2)`` in cells.
    The migrated channels are averaged over poses and mics, so any
    :math:`K` works.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (64, 64),
        dt: float = 0.5,
        channels: int = 8,
        width: int = 12,
        lag_offset: float = -2.0,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.dt = dt
        self.lag_offset = lag_offset
        self.encoder = TraceEncoder(2, channels)
        self.unet = CompactUNet(channels + 2, 2, width)

    def forward(
        self,
        traces: torch.Tensor,
        sources: torch.Tensor,
        mics: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        B, K, M, c_in, L = traces.shape
        feat = self.encoder(traces.reshape(B * K * M, c_in, L)).reshape(B, K, M, -1, L)
        lags = travel_lag_maps(sources, mics, self.grid_shape, self.dt, self.lag_offset)
        img = migrate(feat, lags).mean(dim=(1, 2)).reshape(B, -1, *self.grid_shape)
        return prior_logits + self.unet(torch.cat([img, prior_features(prior_logits)], 1))


class IRGlobalNet(nn.Module):
    """Phase 2-style control: IR encoder to a vector, no spatial alignment.

    Each pose's ``(M * 2, L)`` traces are reduced to a ``latent`` vector
    by strided 1D convolutions and global pooling; the pose's device
    coordinates (divided by the grid size) are appended; an MLP embeds
    the pose; the embeddings are averaged over poses and decoded from
    ``H/8 x W/8`` to the grid.
    """

    def __init__(
        self,
        n_mics: int = 2,
        grid_shape: tuple[int, int] = (64, 64),
        latent: int = 64,
        width: int = 16,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.width = width
        self.start = (grid_shape[0] // 8, grid_shape[1] // 8)
        c = 2 * n_mics
        self.enc = nn.Sequential(
            nn.Conv1d(c, 32, 9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
        )
        self.pose_mlp = nn.Sequential(
            nn.Linear(64 * 8 + 2 * (1 + n_mics), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent),
        )
        self.to_grid = nn.Linear(latent, width * self.start[0] * self.start[1])
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            conv_block(width, width),
            nn.Upsample(scale_factor=2.0),
            conv_block(width, width),
            nn.Upsample(scale_factor=2.0),
        )
        head_out = nn.Conv2d(width, 2, 1)
        with torch.no_grad():
            head_out.weight.zero_()
            assert head_out.bias is not None
            head_out.bias.zero_()
        self.head = nn.Sequential(conv_block(width + 2, width), head_out)

    def forward(
        self,
        traces: torch.Tensor,
        sources: torch.Tensor,
        mics: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        B, K, M, c_in, L = traces.shape
        x = self.enc(traces.reshape(B * K, M * c_in, L)).reshape(B, K, -1)
        n = float(max(self.grid_shape))
        geo = torch.cat([sources[:, :, None, :], mics], 2).reshape(B, K, -1) / n
        z = self.pose_mlp(torch.cat([x, geo], -1)).mean(1)
        g = self.dec(self.to_grid(z).reshape(B, self.width, *self.start))
        return prior_logits + self.head(torch.cat([g, prior_features(prior_logits)], 1))


# ---------------------------------------------------------------------------
# 6.3.3 Pose-aware set model
# ---------------------------------------------------------------------------


class PoseSetNet(nn.Module):
    """DeepSets / attention fusion of per-pose images, then a U-Net.

    ``forward(pose_inputs, prior_logits)`` with ``pose_inputs``
    ``(B, K, C, H, W)``: per-pose physics images and geometry channels
    (``pose_images.geometry_channels``). A shared encoder :math:`\\phi`
    maps each pose to features :math:`F_k`; the pooled map is

    .. math::
        \\Bigl[\\sum_k \\alpha_k F_k,\\; \\tfrac1K \\sum_k F_k,\\;
        \\max_k F_k\\Bigr], \\qquad
        \\alpha_k(\\mathbf{x}) = \\operatorname{softmax}_k\\,
        a^\\top F_k(\\mathbf{x}),

    which is invariant to the order of the poses and defined for any
    :math:`K \\ge 1`.
    """

    def __init__(self, c_pose: int = 7, feat: int = 12, width: int = 12, dropout: float = 0.0):
        super().__init__()
        self.phi = conv_block(c_pose, feat)
        self.score = nn.Conv2d(feat, 1, 1)
        self.unet = CompactUNet(3 * feat + 2, 2, width, dropout)

    def pool(self, pose_inputs: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = pose_inputs.shape
        f = self.phi(pose_inputs.reshape(B * K, C, H, W))
        s = self.score(f).reshape(B, K, 1, H, W)
        f = f.reshape(B, K, -1, H, W)
        att = (torch.softmax(s, dim=1) * f).sum(1)
        return torch.cat([att, f.mean(1), f.amax(1)], 1)

    def forward(self, pose_inputs: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(pose_inputs)
        return prior_logits + self.unet(torch.cat([pooled, prior_features(prior_logits)], 1))


# ---------------------------------------------------------------------------
# Symmetry augmentation
# ---------------------------------------------------------------------------


def d4_grid(x: NDArray, t: int) -> NDArray:
    """Apply element ``t`` (0-7) of the square's symmetry group to the last two axes."""
    a = x
    if t & 4:
        a = np.swapaxes(a, -1, -2)
    if t & 1:
        a = a[..., ::-1, :]
    if t & 2:
        a = a[..., :, ::-1]
    return np.ascontiguousarray(a)


def d4_points(p: NDArray, t: int, n: int) -> NDArray:
    """The same transform on cell coordinates ``(..., 2)`` of an ``n x n`` grid."""
    q = np.array(p, copy=True)
    if t & 4:
        q = q[..., ::-1].copy()
    if t & 1:
        q[..., 0] = n - 1 - q[..., 0]
    if t & 2:
        q[..., 1] = n - 1 - q[..., 1]
    return q


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------

Batch = tuple[Sequence[torch.Tensor], torch.Tensor]


def fit(
    model: nn.Module,
    batches: Callable[[int], Iterable[Batch]],
    epochs: int,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    head_weights: Sequence[float] = (1.0, 1.0),
    log: Callable[[str], None] | None = print,
) -> list[float]:
    """Adam + cosine schedule on the summed per-head binary cross-entropy.

    ``batches(epoch)`` yields ``(inputs, targets)`` with ``targets``
    ``(B, 2, H, W)`` in :math:`\\{0, 1\\}`; ``model(*inputs)`` returns
    logits of the same shape. Returns the mean training loss per epoch.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    hw = torch.tensor(head_weights, dtype=torch.float32).view(1, -1, 1, 1)
    history = []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for inputs, y in batches(ep):
            logits = model(*inputs)
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * hw).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
            n += len(y)
        sched.step()
        history.append(tot / max(n, 1))
        if log is not None:
            log(f"    epoch {ep + 1}/{epochs}: loss {history[-1]:.5f}")
    return history


@torch.no_grad()
def predict_logits(
    models: Sequence[nn.Module],
    batches: Iterable[Sequence[torch.Tensor]],
    mc_samples: int = 0,
) -> NDArray[np.float64]:
    """Logit of the averaged probability over an ensemble (and MC-dropout draws).

    With one model and ``mc_samples = 0`` this is the model's logit. With
    ``mc_samples > 0`` the dropout layers stay active and each model is
    sampled that many times. Returns ``(N, 2, H, W)``.
    """
    out = []
    for inputs in batches:
        probs = []
        for m in models:
            m.eval()
            if mc_samples > 0:
                for mod in m.modules():
                    if isinstance(mod, nn.Dropout2d):
                        mod.train()
                for _ in range(mc_samples):
                    probs.append(torch.sigmoid(m(*inputs).double()))
            else:
                probs.append(torch.sigmoid(m(*inputs).double()))
        p = torch.stack(probs).mean(0).clamp(1e-7, 1 - 1e-7)
        out.append(torch.log(p / (1 - p)).numpy())
    return np.concatenate(out)


def fit_temperature(logits: NDArray, targets: NDArray) -> float:
    """Scalar temperature :math:`T` minimising the log-loss of :math:`\\sigma(z/T)`.

    Temperature scaling keeps the ranking (AP) and rescales confidence; it
    is fitted on training (validation) rooms only.
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()

    def nll(log_t: float) -> float:
        s = z / np.exp(log_t)
        return float(np.mean(np.logaddexp(0.0, s) - y * s))

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def reliability(
    prob: NDArray, truth: NDArray, n_bins: int = 15
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Reliability curve and expected calibration error.

    Bins are equal-mass (quantiles of the predicted probability), which
    suits the heavily skewed occupancy probabilities. Returns
    ``(mean predicted, observed frequency, bin weight, ECE)``.
    """
    p = np.asarray(prob, dtype=np.float64).ravel()
    y = np.asarray(truth, dtype=np.float64).ravel()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    cnt = np.bincount(idx, minlength=len(edges) - 1).astype(np.float64)
    ok = cnt > 0
    mp = np.bincount(idx, p, len(edges) - 1)[ok] / cnt[ok]
    fy = np.bincount(idx, y, len(edges) - 1)[ok] / cnt[ok]
    w = cnt[ok] / cnt.sum()
    return mp, fy, w, float(np.sum(w * np.abs(mp - fy)))


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


__all__ = [
    "CompactUNet",
    "AlignedUNet",
    "IRMigrationNet",
    "IRGlobalNet",
    "PoseSetNet",
    "TraceEncoder",
    "travel_lag_maps",
    "migrate",
    "d4_grid",
    "d4_points",
    "fit",
    "predict_logits",
    "fit_temperature",
    "reliability",
    "n_params",
]
