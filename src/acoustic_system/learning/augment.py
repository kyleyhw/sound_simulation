"""Sim-to-real randomisation for the sensing models (plan 9.8).

Real laptops differ from the simulator in three ways the models can
overfit to: (1) each device colours the signal (speaker and mic
resonances, converter filters), (2) the playback-to-record latency is
unknown and varies by a few samples between runs, and (3) the per-channel
gains differ. Gain and additive noise are handled in
``ActiveSensingDataset``; this module adds the first two.

Device response. A random minimum-phase-ish FIR built from 1-3 damped
resonances plus a direct tap,

    h[n] = delta[n] + sum_k g_k exp(-n / tau_k) cos(2 pi f_k n + phi_k),

normalised to unit energy, applied per channel along time with a causal
convolution (the same response on every channel of one sample, because
the device is shared; a small per-channel variation is added on top).
Resonance frequencies are in cycles per sample, kept below Nyquist.

Latency. An integer circular-free shift in [0, max_shift] samples,
zero-filled at the start, identical across channels (one clock).
"""

from __future__ import annotations

import torch


def random_device_fir(
    n_taps: int = 16,
    n_res: tuple[int, int] = (1, 3),
    gain: tuple[float, float] = (0.1, 0.6),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One random device impulse response of length ``n_taps`` (unit energy)."""
    g = generator
    n = torch.arange(n_taps, dtype=torch.float32)
    h = torch.zeros(n_taps)
    h[0] = 1.0
    k = int(torch.randint(n_res[0], n_res[1] + 1, (1,), generator=g))
    for _ in range(k):
        f = 0.02 + 0.4 * torch.rand(1, generator=g)
        tau = 1.0 + 4.0 * torch.rand(1, generator=g)
        amp = gain[0] + (gain[1] - gain[0]) * torch.rand(1, generator=g)
        phi = 2 * torch.pi * torch.rand(1, generator=g)
        h = h + amp * torch.exp(-n / tau) * torch.cos(2 * torch.pi * f * n + phi)
    return h / h.norm()


def apply_fir(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal convolution of ``x`` (..., T) with ``h`` (L,) or (..., L) along time."""
    t = x.shape[-1]
    lead = x.shape[:-1]
    xf = x.reshape(-1, t)
    hf = h.expand(*lead, h.shape[-1]).reshape(-1, h.shape[-1])
    n = t + hf.shape[-1] - 1
    y = torch.fft.irfft(torch.fft.rfft(xf, n) * torch.fft.rfft(hf, n), n)[:, :t]
    return y.reshape(*lead, t)


def shift(x: torch.Tensor, s: int) -> torch.Tensor:
    """Delay along the last axis by ``s`` samples, zero-filled."""
    if s <= 0:
        return x
    out = torch.zeros_like(x)
    out[..., s:] = x[..., :-s]
    return out


def randomize_device(
    sensor: torch.Tensor,
    n_taps: int = 16,
    channel_jitter: float = 0.05,
    max_shift: int = 3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply a shared random device response, per-channel jitter and latency.

    ``sensor`` is ``(M, T)`` or ``(K, M, T)`` (joint mode: one device, all poses).
    """
    h = random_device_fir(n_taps, generator=generator)
    m = sensor.shape[-2]
    per_ch = h.expand(m, n_taps).clone()
    if channel_jitter > 0:
        per_ch = per_ch + channel_jitter * torch.randn(m, n_taps, generator=generator) / n_taps**0.5
        per_ch = per_ch / per_ch.norm(dim=-1, keepdim=True)
    y = apply_fir(sensor, per_ch)
    s = int(torch.randint(0, max_shift + 1, (1,), generator=generator)) if max_shift > 0 else 0
    return shift(y, s)
