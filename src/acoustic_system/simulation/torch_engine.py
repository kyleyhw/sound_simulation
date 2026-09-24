"""Differentiable, batched FDTD in PyTorch (plan Tasks 5.7.2, 5.7.3).

A tensor twin of ``Simulate`` for 2D and 3D grids with a leading batch
dimension B (many rooms per launch). The whole time loop is ordinary
autograd-tracked arithmetic, so gradients flow to:

* **source signals** ``src`` of shape (B, S, T): what the Phase 7
  controllers optimise;
* **obstacle occupancy** ``occ`` in [0, 1] of shape (B, *grid): a
  continuous relaxation of the p = 0 obstacle mask (``occ`` = 1 zeroes the
  cell exactly like ``Simulate``'s obstacle scrub). Full-waveform
  inversion (Phase 6) optimises this;
* **relative wave speed** ``speed`` (B, *grid).

Numerics follow the Python fast path, in the same order: fused leap-frog
with p = 0 outer faces, then the obstacle scrub ``p_next *= (1 - occ)``,
then additive (soft) source injection at t_n. Rigid, impedance and
absorbing-layer boundaries use the general-path coefficients from
``physics.build_coefficients`` (geometry fixed, non-differentiable), so
the tensor engine matches the numba engine on every path.

Memory: backpropagation through T steps stores O(T B N) activations.
``checkpoint_every`` recomputes the loop in segments
(``torch.utils.checkpoint``), trading compute for O(sqrt(T) B N) memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .physics import build_coefficients


@dataclass
class TorchGrid:
    shape: tuple[int, ...]
    wavespeed: float = 1.0
    gridstep: float = 1.0
    courant: float = 0.5

    @property
    def dims(self) -> int:
        return len(self.shape)

    @property
    def timestep(self) -> float:
        kappa = min(self.courant, 0.95 / np.sqrt(self.dims))
        return float(kappa * self.gridstep / self.wavespeed)

    @property
    def coeff(self) -> float:
        return float(np.float32((self.wavespeed * self.timestep / self.gridstep) ** 2))


def _neighbour_sum(p: torch.Tensor, dims: int) -> torch.Tensor:
    """Sum of the 2*dims axial neighbours with zero padding (B, *grid)."""
    pad = [1, 1] * dims
    q = F.pad(p, pad)
    if dims == 2:
        return q[:, :-2, 1:-1] + q[:, 2:, 1:-1] + q[:, 1:-1, :-2] + q[:, 1:-1, 2:]
    return (
        q[:, :-2, 1:-1, 1:-1]
        + q[:, 2:, 1:-1, 1:-1]
        + q[:, 1:-1, :-2, 1:-1]
        + q[:, 1:-1, 2:, 1:-1]
        + q[:, 1:-1, 1:-1, :-2]
        + q[:, 1:-1, 1:-1, 2:]
    )


def _edge_mask(shape: tuple[int, ...], device, dtype) -> torch.Tensor:
    m = torch.ones(shape, device=device, dtype=dtype)
    for a in range(len(shape)):
        idx: list[slice | int] = [slice(None)] * len(shape)
        idx[a] = 0
        m[tuple(idx)] = 0
        idx[a] = shape[a] - 1
        m[tuple(idx)] = 0
    return m


class TorchFDTD:
    """Batched differentiable FDTD.

    Parameters
    ----------
    grid
        Grid geometry and time step (same CFL rule as ``Simulate``).
    src_pos, mic_pos
        Integer cell positions: ``(B, S, dims)`` and ``(B, M, dims)`` long tensors.
    material
        Optional ``(B, *grid)`` uint8 material ids (``physics.MATERIALS``) and
        an outer boundary kind; anything beyond p = 0 walls routes through
        the general coefficients (fixed geometry).
    """

    def __init__(
        self,
        grid: TorchGrid,
        src_pos: torch.Tensor,
        mic_pos: torch.Tensor,
        *,
        material: Optional[np.ndarray] = None,
        boundary: str = "soft",
        boundary_beta: float = 1.0,
        sponge_cells: int = 24,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.grid = grid
        self.device = torch.device(device)
        self.dtype = dtype
        self.src_pos = src_pos.to(self.device)
        self.mic_pos = mic_pos.to(self.device)
        self.B = int(src_pos.shape[0])
        self.general = boundary != "soft" or (material is not None and bool((material > 1).any()))
        n = int(np.prod(grid.shape))
        self.n = n
        self._src_flat = self._flat(self.src_pos)
        self._mic_flat = self._flat(self.mic_pos)
        self.edge = _edge_mask(grid.shape, self.device, dtype)
        self.coef = None
        if self.general:
            if material is None:
                material = np.zeros((self.B,) + grid.shape, dtype=np.uint8)
            cs = [
                build_coefficients(
                    material[b],
                    None,
                    boundary,
                    boundary_beta,
                    sponge_cells,
                    grid.coeff,
                    grid.wavespeed,
                    grid.gridstep,
                    grid.timestep,
                )
                for b in range(self.B)
            ]
            if any(c.mur_edges for c in cs):
                raise NotImplementedError("Mur edges are not implemented in the tensor engine")

            def stack(name: str) -> torch.Tensor:
                return torch.tensor(
                    np.stack([getattr(c, name) for c in cs]), device=self.device, dtype=dtype
                )

            self.coef = {
                k: stack(k) for k in ("active", "k_air", "c2", "s", "qq", "qa", "inv_a", "ks")
            }
        elif material is not None:
            self.static_occ = torch.tensor(
                (material == 1).astype(np.float32), device=self.device, dtype=dtype
            )
        else:
            self.static_occ = None

    def _flat(self, pos: torch.Tensor) -> torch.Tensor:
        strides = []
        acc = 1
        for s in reversed(self.grid.shape):
            strides.append(acc)
            acc *= s
        strides = torch.tensor(list(reversed(strides)), device=pos.device, dtype=torch.long)
        return (pos.long() * strides).sum(-1)

    def _segment(self, p, pp, v, x, src_seg, occ_keep, c2, t0: int):
        dims = self.grid.dims
        recs = []
        B = p.shape[0]
        for k in range(src_seg.shape[-1]):
            nb = _neighbour_sum(p, dims)
            if self.coef is None:
                lap = nb - 2 * dims * p
                pn = (2 * p - pp + c2 * lap) * self.edge
                if occ_keep is not None:
                    pn = pn * occ_keep
            else:
                cf = self.coef
                lap = nb - cf["k_air"] * p
                sd = cf["s"]
                rhs = 2 * p - pp + c2 * lap + sd * pp
                q = cf["qa"]
                rhs_b = rhs - q * (0.5 * p - cf["ks"] * x) + cf["qq"] * v
                nxt_b = rhs_b / (1 + 0.5 * q + sd)
                nxt = torch.where(q != 0, nxt_b, rhs / (1 + sd))
                vn = torch.where(q != 0, (0.5 * (nxt + p) - cf["ks"] * x) * cf["inv_a"], v)
                x = x + self.grid.timestep * torch.where(q != 0, vn, torch.zeros_like(vn))
                v = vn
                pn = nxt * cf["active"]
                if occ_keep is not None:
                    pn = pn * occ_keep
            # Soft sources (added last), scatter-add into the flat field.
            flat = pn.reshape(B, -1)
            flat = flat.scatter_add(1, self._src_flat, src_seg[:, :, k])
            pn = flat.reshape(p.shape)
            pp, p = p, pn
            recs.append(torch.gather(p.reshape(B, -1), 1, self._mic_flat))
        return p, pp, v, x, torch.stack(recs, dim=-1)

    def run(
        self,
        src: torch.Tensor,
        *,
        occ: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
        checkpoint_every: int = 0,
        return_field: bool = False,
    ):
        """Run ``src.shape[-1]`` steps; return mic recordings (B, M, T).

        ``src[b, s, n]`` is the source value at time t_n (injected into
        p^{n+1}), i.e. the waveform sampled at ``n * dt``.
        """
        shape = (self.B,) + self.grid.shape
        p = torch.zeros(shape, device=self.device, dtype=self.dtype)
        pp = torch.zeros_like(p)
        v = torch.zeros_like(p)
        x = torch.zeros_like(p)
        keep = None
        if occ is not None:
            keep = 1.0 - occ
        else:
            static = getattr(self, "static_occ", None)
            if static is not None:
                keep = 1.0 - static
        c2 = self.grid.coeff if speed is None else self.grid.coeff * speed**2
        if self.coef is not None and speed is None:
            c2 = self.coef["c2"]
        T = src.shape[-1]
        seg = checkpoint_every if checkpoint_every > 0 else T
        recs = []
        for t0 in range(0, T, seg):
            chunk = src[..., t0 : t0 + seg]
            if checkpoint_every > 0 and torch.is_grad_enabled():
                res: Any = checkpoint(
                    self._segment, p, pp, v, x, chunk, keep, c2, t0, use_reentrant=False
                )
                p, pp, v, x, r = res
            else:
                p, pp, v, x, r = self._segment(p, pp, v, x, chunk, keep, c2, t0)
            recs.append(r)
        out = torch.cat(recs, dim=-1)
        return (out, p) if return_field else out


def sample_waveform(waveform, steps: int, dt: float) -> np.ndarray:
    """Waveform values at t_n = n dt (the engine's injection times)."""
    return np.array([waveform(n * dt) for n in range(steps)], dtype=np.float32)


__all__ = ["TorchFDTD", "TorchGrid", "sample_waveform"]
