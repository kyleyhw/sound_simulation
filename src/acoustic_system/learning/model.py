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

    def __init__(
        self,
        in_channels: int,
        base_ch: int = 16,
        latent_size: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Dropout2d zeros entire feature-map channels with probability p,
        # which works better than per-element Dropout in conv blocks
        # because adjacent pixels in a feature map are highly correlated
        # — independently zeroing them barely regularises the channel.
        layers: list[nn.Module] = [
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
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.AdaptiveAvgPool2d(latent_size))
        self.net = nn.Sequential(*layers)

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
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sensor_spec = T.Spectrogram(n_fft=sensor_n_fft, hop_length=sensor_hop, power=2.0)
        self.source_spec = T.Spectrogram(n_fft=source_n_fft, hop_length=source_hop, power=2.0)
        self.encoder_s = SpectrogramEncoder(
            in_channels=n_mics, base_ch=base_ch, latent_size=latent_size, dropout=dropout
        )
        self.encoder_u = SpectrogramEncoder(
            in_channels=1, base_ch=base_ch, latent_size=latent_size, dropout=dropout
        )
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


class PassiveCNN(nn.Module):
    """Sensor-only obstacle-mask CNN for passive sensing (Task 2.2).

    The blind-deconvolution setting: the source signal $u(t)$ is
    unknown, so the model must infer geometry from the stereo recording
    alone — inter-channel level/spectral structure and the reverberant
    tail are the only cues. Architecturally this is ``DualInputCNN``
    with the source branch amputated: the same ``SpectrogramEncoder``
    on the ``n_mics``-channel sensor spectrogram, and the same
    ``MaskDecoder`` (fed the single branch's ``base_ch * 4`` channels
    instead of the concat's ``2 * base_ch * 4``). Keeping every other
    hyperparameter identical makes the active-vs-passive comparison a
    controlled experiment: the performance difference isolates the
    value of knowing the source.

    Known limitation, shared with the active model: the ``power=2.0``
    spectrogram front-end discards phase, i.e. inter-channel time
    differences (TDOA). A GCC-PHAT-style cross-correlation input
    channel is the designated upgrade if the magnitude-only passive
    model proves too weak; noted in ``docs/learning.md``.

    ``forward`` accepts and ignores an optional ``source`` argument so
    the training/eval plumbing can stay model-agnostic (the dataset
    always yields the triplet; a passive model simply never looks at
    the middle element).
    """

    def __init__(
        self,
        n_mics: int = 2,
        sensor_n_fft: int = 64,
        sensor_hop: int = 16,
        base_ch: int = 16,
        mid_ch: int = 64,
        latent_size: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sensor_spec = T.Spectrogram(n_fft=sensor_n_fft, hop_length=sensor_hop, power=2.0)
        self.encoder_s = SpectrogramEncoder(
            in_channels=n_mics, base_ch=base_ch, latent_size=latent_size, dropout=dropout
        )
        self.decoder = MaskDecoder(in_channels=base_ch * 4, mid_ch=mid_ch)

    def forward(self, sensor: torch.Tensor, source: torch.Tensor | None = None) -> torch.Tensor:
        s_spec = torch.log1p(self.sensor_spec(sensor))
        return self.decoder(self.encoder_s(s_spec))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class JointPoseCNN(nn.Module):
    """K-pose sensor + source -> obstacle mask (Task 2.1.4c).

    The learned counterpart of the Bayes fusion that passed the 2.1.4b
    decision gate (tests/reports/multipose_2026_07_10.md): there, K
    single-pose predictions fused by a prior-corrected logit sum lifted
    held-out IoU 2.4x. Here the fusion happens in latent space and is
    trained end-to-end:

        sensor (B, K, n_mics, T) ── shared STFT+Encoder_s per pose ── (B, K, C, L, L)
                                                    │ mean over K
                                                    ▼
                                              (B, C, L, L) ──┐
        source (B, 1, T_aud)     ── STFT ── Encoder_u ────────┴─ concat ─ Decoder ─ logits

    Design choices:

    - **Shared encoder** across poses: pose index carries no meaning
      (placements are i.i.d.), so per-pose weights would only multiply
      parameters K-fold and break permutation symmetry.
    - **Mean pooling** over the pose axis: permutation-invariant and
      K-agnostic, so a model trained at K=4 evaluates at any K (the
      2.1.4b curve was still rising at K=8 — inference can exploit
      more poses than training saw). Mean rather than sum keeps the
      latent scale K-independent; the evidence-accumulation role of
      the Bayes logit *sum* is available to the network through the
      decoder's weights, without hard-coding a fusion rule.
    - Everything else (STFT settings, encoder/decoder shapes) is
      identical to ``DualInputCNN``, so single-pose vs joint-pose is a
      controlled comparison — same parameter count, same capacity;
      only the pose information differs.

    ``forward`` also accepts single-pose input ``(B, n_mics, T)`` and
    treats it as K=1, so the model degrades gracefully to the
    single-pose setting.
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
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sensor_spec = T.Spectrogram(n_fft=sensor_n_fft, hop_length=sensor_hop, power=2.0)
        self.source_spec = T.Spectrogram(n_fft=source_n_fft, hop_length=source_hop, power=2.0)
        self.encoder_s = SpectrogramEncoder(
            in_channels=n_mics, base_ch=base_ch, latent_size=latent_size, dropout=dropout
        )
        self.encoder_u = SpectrogramEncoder(
            in_channels=1, base_ch=base_ch, latent_size=latent_size, dropout=dropout
        )
        self.decoder = MaskDecoder(in_channels=2 * base_ch * 4, mid_ch=mid_ch)

    def forward(self, sensor: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if sensor.dim() == 3:  # (B, n_mics, T) -> K=1
            sensor = sensor.unsqueeze(1)
        b, k, m, t = sensor.shape
        # Fold poses into the batch for one shared encoder pass, then
        # mean-pool the per-pose latents back out.
        s_spec = torch.log1p(self.sensor_spec(sensor.reshape(b * k, m, t)))
        s_lat = self.encoder_s(s_spec)
        s_lat = s_lat.reshape(b, k, *s_lat.shape[1:]).mean(dim=1)
        u_lat = self.encoder_u(torch.log1p(self.source_spec(source)))
        return self.decoder(torch.cat([s_lat, u_lat], dim=1))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(model_type: str, n_mics: int = 2, dropout: float = 0.0) -> nn.Module:
    """Construct a model by checkpoint tag: 'dual', 'passive', or 'joint'."""
    if model_type == "dual":
        return DualInputCNN(n_mics=n_mics, dropout=dropout)
    if model_type == "passive":
        return PassiveCNN(n_mics=n_mics, dropout=dropout)
    if model_type == "joint":
        return JointPoseCNN(n_mics=n_mics, dropout=dropout)
    raise ValueError(f"unknown model_type {model_type!r}")
