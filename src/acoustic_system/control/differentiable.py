"""Differentiable sound-zone control through the tensor FDTD (plan 7.6).

Instead of designing per-frequency weights from transfer functions and
then approximating them with FIR filters (:mod:`.beamforming`), optimise
the speaker drive signals directly through the simulator with autograd
(``simulation.torch_engine.TorchFDTD``). The objective is the time-domain
energy contrast over the simulated window:

.. math::
    J(z) = -10\\log_{10}
        \\frac{\\frac{1}{M_B}\\sum_{m\\in B}\\sum_n y_m[n]^2}
             {\\frac{1}{M_D}\\sum_{m\\in D}\\sum_n y_m[n]^2},
    \\qquad u_s = z_s * x,

where :math:`x` is a band-limited programme pulse and :math:`z_s` are
per-speaker FIR filters, the optimisation variables. This is the same
structure as a broadband ACC design (filters applied to a programme), so
the FIR from :func:`.beamforming.design_broadband` is a natural starting
point. The contrast is invariant to the scale of :math:`z`, and Adam works
on :math:`z` directly.

Unlike ACC, this optimises exactly what is scored: finite filters,
transients and the finite window are all included. What it gives up:
only the scored window counts (energy that arrives later is not seen),
and every iteration costs a forward and a backward simulation. The tensor
engine supports p = 0, rigid, impedance and sponge walls, not CPML.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..simulation.torch_engine import TorchFDTD, TorchGrid
from .metrics import energy_contrast_db
from .transfer import Room, bandpass_pulse, simulate_drives


@dataclass
class DiffScene:
    """A small room, an array and two zones for the tensor engine."""

    room: Room
    speakers: np.ndarray  # (S, 2) world metres
    bright: np.ndarray  # (M_B, 2)
    dark: np.ndarray  # (M_D, 2)
    band: tuple[float, float] = (200.0, 800.0)
    n_taps: int = 256  # filter length L
    programme_taps: int = 101
    tail: float = 0.04  # seconds simulated after the drives end
    dtype: torch.dtype = torch.float32
    programme: np.ndarray = field(init=False)
    steps: int = field(init=False)

    def __post_init__(self) -> None:
        if self.room.boundary in ("cpml", "mur"):
            raise ValueError("the tensor engine has no CPML or Mur boundary")
        dt = self.room.timestep
        self.programme = bandpass_pulse(dt, self.band, self.programme_taps)
        drive_len = self.n_taps + len(self.programme) - 1
        self.steps = drive_len + int(np.ceil(self.tail / dt))
        room = self.room
        grid = TorchGrid(room.shape, wavespeed=room.c, gridstep=room.dx, courant=room.courant)
        mics = np.concatenate([self.bright, self.dark])
        self._nb = len(self.bright)
        self.engine = TorchFDTD(
            grid,
            torch.tensor(room.cells(self.speakers))[None],
            torch.tensor(room.cells(mics))[None],
            material=room.material_map()[None],
            boundary=room.boundary,
            boundary_beta=room.beta,
            dtype=self.dtype,
        )
        # conv1d kernel: flipped programme, so conv1d computes z * x.
        x = torch.tensor(self.programme[::-1].copy(), dtype=self.dtype)
        self._xk = x.view(1, 1, -1)

    @property
    def dt(self) -> float:
        return self.room.timestep

    def drives(self, z: torch.Tensor) -> torch.Tensor:
        """Drive signals (S, T) = z * x, zero padded to the run length."""
        S, L = z.shape
        K = self._xk.shape[-1]
        u = F.conv1d(F.pad(z.view(S, 1, L), (K - 1, K - 1)), self._xk).view(S, -1)
        return F.pad(u, (0, self.steps - u.shape[-1]))

    def record(self, z: torch.Tensor) -> torch.Tensor:
        """Mic recordings (M_B + M_D, T) for filters ``z`` (S, L)."""
        return self.engine.run(self.drives(z)[None])[0]

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        """Negative energy contrast in dB (the quantity minimised)."""
        y = self.record(z)
        eb = torch.mean(torch.sum(y[: self._nb] ** 2, dim=-1))
        ed = torch.mean(torch.sum(y[self._nb :] ** 2, dim=-1))
        return -10.0 * torch.log10(eb / ed)

    def contrast_db(self, z: "np.ndarray | torch.Tensor") -> float:
        with torch.no_grad():
            return float(-self.loss(torch.as_tensor(np.asarray(z), dtype=self.dtype)))

    def verify_numba(self, z: np.ndarray) -> float:
        """Energy contrast of filters ``z`` replayed in the numba engine."""
        with torch.no_grad():
            u = self.drives(torch.as_tensor(np.asarray(z), dtype=self.dtype)).numpy()
        mics = np.concatenate([self.bright, self.dark])
        rec = simulate_drives(self.room, self.speakers, u, mics, self.steps).rec
        return energy_contrast_db(rec[: self._nb], rec[self._nb :])


@dataclass
class OptimResult:
    z: np.ndarray  # optimised filters (S, L)
    history_db: np.ndarray  # contrast after each iteration
    initial_db: float
    final_db: float


def optimise_drives(
    scene: DiffScene,
    z0: "np.ndarray | None" = None,
    *,
    iters: int = 60,
    lr: float = 0.05,
    seed: int = 0,
) -> OptimResult:
    """Maximise the zone energy contrast with Adam through the tensor engine.

    ``z0`` (S, L) is the starting filter set (e.g. a broadband ACC design);
    by default small random filters. The filters are rescaled to unit RMS
    before the optimisation (the objective is scale invariant, and Adam's
    step ``lr`` is then relative to the filter size).
    """
    S, L = len(scene.speakers), scene.n_taps
    if z0 is None:
        z0 = np.random.default_rng(seed).standard_normal((S, L))
    z0 = np.asarray(z0, dtype=np.float64)
    z0 = z0 / np.sqrt(np.mean(z0**2))
    z = torch.tensor(z0, dtype=scene.dtype, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    hist = []
    best = (np.inf, z0.copy())
    for _ in range(iters):
        opt.zero_grad()
        loss = scene.loss(z)
        loss.backward()
        val = float(loss.detach())
        hist.append(-val)
        if val < best[0]:
            best = (val, z.detach().numpy().astype(np.float64).copy())
        opt.step()
    final = scene.contrast_db(z.detach())
    if -final < best[0]:
        best = (-final, z.detach().numpy().astype(np.float64).copy())
    hist.append(final)
    return OptimResult(best[1], np.array(hist), hist[0], -best[0])


def gradient_check(scene: DiffScene, seed: int = 0, eps: float = 1e-4) -> float:
    """Relative error between the autograd and central-difference derivatives.

    Compares :math:`\\nabla J \\cdot v` with
    :math:`(J(z + \\epsilon v) - J(z - \\epsilon v)) / 2\\epsilon` along a
    random unit direction :math:`v`. Use a float64 scene.
    """
    rng = np.random.default_rng(seed)
    S, L = len(scene.speakers), scene.n_taps
    z0 = rng.standard_normal((S, L))
    v = rng.standard_normal((S, L))
    v /= np.linalg.norm(v)
    z = torch.tensor(z0, dtype=scene.dtype, requires_grad=True)
    scene.loss(z).backward()
    assert z.grad is not None
    g = float(np.sum(z.grad.numpy() * v))
    with torch.no_grad():
        jp = float(scene.loss(torch.tensor(z0 + eps * v, dtype=scene.dtype)))
        jm = float(scene.loss(torch.tensor(z0 - eps * v, dtype=scene.dtype)))
    fd = (jp - jm) / (2 * eps)
    return abs(g - fd) / max(abs(fd), 1e-12)


__all__ = ["DiffScene", "OptimResult", "gradient_check", "optimise_drives"]
