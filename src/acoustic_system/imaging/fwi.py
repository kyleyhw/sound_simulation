"""Full-waveform inversion of an occupancy map (plan Task 6.2.5).

A proof of concept on the differentiable engine
(``simulation/torch_engine.TorchFDTD``): find the occupancy
:math:`o(\\mathbf{x}) \\in [0, 1]` whose simulated recordings match the
measured ones for every pose,

.. math::
    \\min_\\theta\\; \\frac{\\sum_{k,m} \\lVert W * (y_{km}(o) - y^\\text{obs}_{km})
        \\rVert^2}{\\sum_{k,m} \\lVert W * (y^{(0)}_{km} - y^\\text{obs}_{km})\\rVert^2}
      + \\lambda_\\text{TV}\\,\\mathrm{TV}(o),
    \\qquad o = \\sigma(\\theta),

where :math:`y^{(0)}` is the empty-room response (so the data term starts
at 1), :math:`W` a low-pass filter whose cut-off rises in stages
(frequency continuation: long wavelengths first, to stay out of the
cycle-skipping basin), and :math:`\\mathrm{TV}(o) = \\sum_\\mathbf{x}
\\sqrt{|\\nabla o|^2 + \\epsilon^2}` the isotropic total variation, which
favours piecewise-constant maps with sharp edges, the shape of the true
masks. The sigmoid keeps :math:`o` in the physical range; the engine's
scrub :math:`p \\leftarrow (1 - o)\\,p` makes a fractional :math:`o` a
per-step absorber, so the map is initialised near zero (:math:`o_0 =
\\sigma(\\theta_0)` with :math:`\\theta_0 = -8` away from the prior) rather
than at the prior itself, which would damp the whole room.

The gradient :math:`\\partial J/\\partial\\theta` is the adjoint-state
gradient, i.e. at :math:`o = 0` it is the time-reversal image of
``time_reversal.py`` (checked in ``tests/imaging``). This module is kept
small on purpose: a few rooms and tens of iterations, run on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class FWIResult:
    occupancy: NDArray[np.float64]
    logits: NDArray[np.float64]
    history: list[float] = field(default_factory=list)


def total_variation(o, eps: float = 1e-2):
    """Isotropic TV of a ``(H, W)`` tensor (forward differences, zero-padded)."""
    import torch

    gi = torch.diff(o, dim=0, append=o[-1:, :])
    gj = torch.diff(o, dim=1, append=o[:, -1:])
    return torch.sqrt(gi * gi + gj * gj + eps * eps).sum()


def lowpass(x, f_cut: float, dt: float, width: float = 0.05):
    """Raised-cosine low-pass along the last axis (cycles per unit time)."""
    import torch

    T = x.shape[-1]
    n_fft = 2 * T
    X = torch.fft.rfft(x, n_fft, dim=-1)
    f = torch.fft.rfftfreq(n_fft, d=dt).to(x.dtype)
    lo, hi = f_cut - width / 2, f_cut + width / 2
    w = torch.clamp((hi - f) / (hi - lo), 0.0, 1.0)
    w = 0.5 - 0.5 * torch.cos(np.pi * w)
    return torch.fft.irfft(X * w, n_fft, dim=-1)[..., :T]


def invert_room(
    grid_shape: tuple[int, int],
    source_positions: NDArray,
    mic_positions: NDArray,
    drive: NDArray,
    recordings: NDArray,
    iterations: tuple[int, ...] = (15, 15, 20),
    cutoffs: tuple[float, ...] = (0.12, 0.25, 0.47),
    lr: float = 0.3,
    tv_weight: float = 2e-4,
    theta0: float = -8.0,
    init_logits: NDArray | None = None,
    courant: float = 0.5,
    seed: int = 0,
) -> FWIResult:
    """Invert one room's recordings for its occupancy map.

    Parameters
    ----------
    source_positions, mic_positions
        ``(K, 2)`` and ``(K, M, 2)`` cells.
    recordings
        ``(K, T, M)`` archive layout.
    iterations, cutoffs
        Adam steps per frequency-continuation stage and the stage cut-offs.
    init_logits
        Optional starting :math:`\\theta` (defaults to ``theta0`` everywhere).

    Device cells are pinned to free space (the laptop stands in air).
    """
    import torch

    from ..simulation.torch_engine import TorchFDTD, TorchGrid

    torch.manual_seed(seed)
    src = np.asarray(source_positions).reshape(-1, 2)
    mics = np.asarray(mic_positions).reshape(len(src), -1, 2)
    K, M = mics.shape[:2]
    grid = TorchGrid(tuple(grid_shape), courant=courant)
    eng = TorchFDTD(
        grid,
        torch.tensor(src[:, None, :], dtype=torch.long),
        torch.tensor(mics, dtype=torch.long),
    )
    drive_t = torch.tensor(np.asarray(drive, np.float32))[None, None, :].expand(K, 1, -1)
    obs = torch.tensor(np.transpose(np.asarray(recordings, np.float32), (0, 2, 1)))
    with torch.no_grad():
        empty = eng.run(drive_t)
    free = torch.zeros(grid_shape, dtype=torch.bool)
    for p in np.concatenate([src, mics.reshape(-1, 2)]):
        free[int(p[0]), int(p[1])] = True
    if init_logits is None:
        theta = torch.full(grid_shape, float(theta0))
    else:
        theta = torch.tensor(np.asarray(init_logits, np.float32))
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    history: list[float] = []
    dt = grid.timestep
    for n_it, f_cut in zip(iterations, cutoffs):
        with torch.no_grad():
            norm = (lowpass(empty - obs, f_cut, dt) ** 2).sum().clamp_min(1e-12)
            obs_f = lowpass(obs, f_cut, dt)
        for _ in range(int(n_it)):
            opt.zero_grad()
            occ = torch.sigmoid(theta).masked_fill(free, 0.0)
            pred = eng.run(drive_t, occ=occ[None].expand(K, *grid_shape))
            data = (lowpass(pred, f_cut, dt) - obs_f).pow(2).sum() / norm
            loss = data + tv_weight * total_variation(occ)
            loss.backward()
            opt.step()
            history.append(float(data.detach()))
    with torch.no_grad():
        occ = torch.sigmoid(theta).masked_fill(free, 0.0)
    return FWIResult(
        occupancy=occ.numpy().astype(np.float64),
        logits=theta.detach().numpy().astype(np.float64),
        history=history,
    )


__all__ = ["FWIResult", "total_variation", "lowpass", "invert_room"]
