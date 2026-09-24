"""Differentiable batched engine (plan 5.7.2/5.7.3): parity + gradients."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from acoustic_system.simulation.dataset import run_with_sensors
from acoustic_system.simulation.setup import Driver, Sensor
from acoustic_system.simulation.simulate import Simulate
from acoustic_system.simulation.torch_engine import TorchFDTD, TorchGrid, sample_waveform
from acoustic_system.simulation.waveforms import RickerWavelet

WF = RickerWavelet(1.0, 0.1, 15.0)


def _numba(shape, src, mics, steps, mask=None, boundary="soft", material=None):
    sim = Simulate(shape, boundary=boundary)
    if mask is not None:
        sim.set_obstacle([tuple(c) for c in np.argwhere(mask)])
    if material is not None:
        sim.set_material_map(material)
    sim.set_drivers([Driver(src, WF)])
    return run_with_sensors(sim, steps, [Sensor(position=m) for m in mics]).T


@pytest.mark.parametrize("dims", [2, 3])
def test_matches_numba_fast_path_batched(dims: int) -> None:
    shape = (40, 36) if dims == 2 else (20, 18, 16)
    rng = np.random.default_rng(0)
    B, steps = 3, 120
    masks = np.zeros((B,) + shape, dtype=np.uint8)
    srcs, mics = [], []
    for b in range(B):
        sl = tuple(slice(5 + b, 9 + b) for _ in shape)
        masks[b][sl] = 1
        srcs.append(tuple(int(v) for v in rng.integers(10, 15, size=dims)))
        mics.append(
            [tuple(int(rng.integers(2, s - 2)) for s in shape), tuple(s // 2 for s in shape)]
        )
    grid = TorchGrid(shape)
    eng = TorchFDTD(
        grid,
        torch.tensor([[s] for s in srcs]),
        torch.tensor([[m for m in ms] for ms in mics]),
        material=masks,
    )
    src = torch.tensor(np.stack([sample_waveform(WF, steps, grid.timestep)] * B))[:, None, :]
    rec = eng.run(src).numpy()
    for b in range(B):
        ref = _numba(shape, srcs[b], mics[b], steps, mask=masks[b])
        assert np.max(np.abs(rec[b] - ref)) < 1e-4 * max(1.0, np.abs(ref).max())


def test_matches_numba_general_path() -> None:
    shape = (40, 40)
    mat = np.zeros(shape, np.uint8)
    mat[10:30, 20] = 2
    mat[5:15, 30] = 4
    mat[30, 5:25] = 6
    grid = TorchGrid(shape)
    eng = TorchFDTD(
        grid,
        torch.tensor([[[20, 10]]]),
        torch.tensor([[[25, 33], [8, 8]]]),
        material=mat[None],
        boundary="absorb",
        boundary_beta=0.5,
    )
    src = torch.tensor(sample_waveform(WF, 200, grid.timestep))[None, None]
    rec = eng.run(src).numpy()[0]
    sim = Simulate(shape, boundary="absorb", boundary_beta=0.5)
    sim.set_material_map(mat)
    sim.set_drivers([Driver((20, 10), WF)])
    ref = run_with_sensors(sim, 200, [Sensor(position=(25, 33)), Sensor(position=(8, 8))]).T
    assert np.max(np.abs(rec - ref)) < 1e-4 * max(1.0, np.abs(ref).max())


def test_gradients_match_finite_differences() -> None:
    torch.manual_seed(0)
    grid = TorchGrid((14, 12))
    eng = TorchFDTD(grid, torch.tensor([[[5, 4]]]), torch.tensor([[[9, 8]]]), dtype=torch.float64)
    src = torch.randn(1, 1, 30, dtype=torch.float64, requires_grad=True)
    occ = torch.full((1, 14, 12), 0.1, dtype=torch.float64, requires_grad=True)

    def f(s, o):
        return (eng.run(s, occ=o) ** 2).sum()

    assert torch.autograd.gradcheck(f, (src, occ), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_checkpointing_gives_identical_gradients() -> None:
    grid = TorchGrid((20, 20))
    eng = TorchFDTD(grid, torch.tensor([[[5, 5]]]), torch.tensor([[[14, 14]]]))
    src = torch.randn(1, 1, 80, requires_grad=True)
    g1 = torch.autograd.grad((eng.run(src) ** 2).sum(), src)[0]
    g2 = torch.autograd.grad((eng.run(src, checkpoint_every=16) ** 2).sum(), src)[0]
    assert torch.allclose(g1, g2, atol=1e-5)
