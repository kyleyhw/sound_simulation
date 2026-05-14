"""DualInputCNN — sensor + source -> obstacle-mask CNN.

Architecture
------------
Two parallel spectrogram-encoder branches followed by concat fusion and
a transposed-conv mask decoder::

    sensor (B, n_mics, T_rec) ─── STFT ─── Encoder_s ──┐
                                                       ├── Concat ── 1x1 mix ── Decoder ── M̂ (B, 1, H, W)
    source (B, 1,      T_aud) ─── STFT ─── Encoder_u ──┘

Each encoder ends with an ``AdaptiveAvgPool2d(latent_size)`` so the
two branches converge to the same latent tensor shape regardless of
the per-branch STFT output dimensions. This decouples the model from
sensor-vs-source rate mismatches: each branch picks its own STFT
window/hop and the encoder normalises.

Why STFT-based front-end
------------------------
Room impulse responses are most naturally read in the time-frequency
plane (modal structure, direct-vs-reverberant separation). A 2D conv
on a log-magnitude spectrogram sees that structure directly; a 1D
conv on the raw waveform would have to spend capacity rediscovering
the FFT.

Why concat fusion (and not cross-attention)
-------------------------------------------
The model only has to compare ``sensor`` against ``source`` once per
sample, and the comparison is dominated by alignment in time and
frequency rather than by content semantics. Channel-concat + a 1x1
mix gives the decoder enough degrees of freedom to learn that
relationship; cross-attention is the natural upgrade if concat
plateaus, but it adds significant params and we are intentionally
keeping the model small for CPU inference.

Output target
-------------
``target_size`` square mask (default 64) of obstacle-probability logits.
The training loss applies sigmoid + BCE+Dice; eval thresholds at 0.5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpectrogramEncoder(nn.Module):
    """4-stage 2D conv encoder ending in an AdaptiveAvgPool2d.

    The pooling step is what makes the encoder agnostic to its input
    spectrogram size; the trade-off is that fine spatial structure
    finer than ``latent_size`` is averaged away. For the active-sensing
    task this is fine — the obstacle mask itself only resolves at
    ``target_size`` cells, so latent_size=8 is comfortably large.
    """

    def __init__(self, in_channels: int, base_ch: int = 16, latent_size: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(latent_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskDecoder(nn.Module):
    """Transposed-conv mask decoder: latent_size -> 4*latent_size.

    Three doubling stages take an ``(B, C_in, L, L)`` latent up to
    ``(B, 1, 8L, 8L)``. With latent_size=8 that's 8 -> 16 -> 32 -> 64,
    matching the default ``target_size``. Each transposed conv is
    followed by ReLU; the final layer emits raw logits (no activation).
    """

    def __init__(self, in_channels: int, mid_ch: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch * 2, mid_ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch, mid_ch // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch // 2, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualInputCNN(nn.Module):
    """Sensor + source -> obstacle mask.

    Hyperparameters tuned for "small enough to train on CPU in minutes,
    expressive enough to learn the inverse map for a 64x64 scene":
    ``base_ch=16`` gives a final encoder latent of 64 channels per
    branch (128 after concat), ``mid_ch=64`` decoder. Total ~200k
    parameters.

    STFT defaults
    -------------
    - Sensor: short window (n_fft=64, hop=16) because T_rec is small
      (a few hundred steps); we want enough time bins for the encoder.
    - Source: long window (n_fft=512, hop=256) because T_audio is
      typically 10-100x longer; we want roughly the same time-bin count.

    Both branches end at (B, 64, 8, 8) regardless of STFT shape.
    """

    def __init__(
        self,
        n_mics: int = 2,
        sensor_n_fft: int = 64,
        sensor_hop: int = 16,
        source_n_fft: int = 512,
        source_hop: int = 256,
        base_ch: int = 16,
        mid_ch: int = 64,
        latent_size: int = 8,
    ) -> None:
        super().__init__()
        self.sensor_spec = T.Spectrogram(n_fft=sensor_n_fft, hop_length=sensor_hop, power=2.0)
        self.source_spec = T.Spectrogram(n_fft=source_n_fft, hop_length=source_hop, power=2.0)
        self.encoder_s = SpectrogramEncoder(
            in_channels=n_mics, base_ch=base_ch, latent_size=latent_size
        )
        self.encoder_u = SpectrogramEncoder(in_channels=1, base_ch=base_ch, latent_size=latent_size)
        # Each branch outputs base_ch * 4 channels; concat -> 2 * base_ch * 4.
        self.decoder = MaskDecoder(in_channels=2 * base_ch * 4, mid_ch=mid_ch)

    def forward(self, sensor: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        # log1p(power) gives a numerically stable log-magnitude spectrogram.
        s_spec = torch.log1p(self.sensor_spec(sensor))
        u_spec = torch.log1p(self.source_spec(source))
        s_lat = self.encoder_s(s_spec)
        u_lat = self.encoder_u(u_spec)
        fused = torch.cat([s_lat, u_lat], dim=1)
        return self.decoder(fused)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
