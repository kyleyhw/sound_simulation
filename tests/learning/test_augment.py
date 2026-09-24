"""Sim-to-real device randomisation (plan 9.8)."""

from __future__ import annotations

import numpy as np
import torch

from acoustic_system.learning.augment import apply_fir, random_device_fir, randomize_device, shift


def test_fir_is_unit_energy_and_causal_convolution_matches_numpy():
    g = torch.Generator().manual_seed(0)
    h = random_device_fir(16, generator=g)
    assert torch.isclose(h.norm(), torch.tensor(1.0), atol=1e-5)
    x = torch.randn(3, 200, generator=g)
    y = apply_fir(x, h)
    ref = torch.stack([torch.from_numpy(np.convolve(xi.numpy(), h.numpy())[:200]) for xi in x])
    assert torch.allclose(y, ref.float(), atol=1e-4)


def test_randomize_device_shapes_shift_and_energy():
    g = torch.Generator().manual_seed(1)
    x = torch.zeros(2, 100)
    x[:, 10] = 1.0
    y = randomize_device(x, max_shift=0, channel_jitter=0.0, generator=g)
    assert y.shape == x.shape
    # Impulse in, device response out: unit energy, starting at the impulse.
    assert torch.allclose(y.norm(dim=-1), torch.ones(2), atol=1e-4)
    assert torch.all(y[:, :10].abs() < 1e-6)
    assert torch.equal(shift(x, 3)[:, 13], x[:, 10])
    joint = randomize_device(torch.randn(4, 2, 100, generator=g), generator=g)
    assert joint.shape == (4, 2, 100)
