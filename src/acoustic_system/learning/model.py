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


class StereoPhaseFrontEnd(nn.Module):
    """Complex-STFT stereo front-end with inter-channel phase (Task 2.3a).

    The magnitude-only front-end of the earlier models discards phase,
    and with it the inter-channel time difference (TDOA) — for a stereo
    pair the strongest geometric cue there is. This front-end keeps it:
    from the complex STFTs $X_1, X_2$ of the two mics it emits four
    channels per time-frequency bin,

    $$ \\bigl[\\, \\log(1{+}|X_1|^2),\\ \\log(1{+}|X_2|^2),\\
       \\cos\\varphi,\\ \\sin\\varphi \\,\\bigr], \\qquad
       \\varphi(f, t) = \\arg\\bigl(X_1 X_2^*\\bigr), $$

    where $\\varphi$ is the inter-channel phase difference (the phase of
    the GCC-PHAT cross-spectrum): for a pure arrival-time difference
    $\\tau$, $\\varphi = 2\\pi f \\tau$ — a plane in $(f, t)$ whose slope
    encodes direction. cos/sin rather than raw $\\varphi$ avoids the
    $\\pm\\pi$ wrap discontinuity. In near-silent bins $\\varphi$ is
    noise; the adjacent magnitude channels give the network exactly the
    information needed to gate it, so no explicit masking is applied.
    """

    def __init__(self, n_fft: int = 64, hop: int = 16) -> None:
        super().__init__()
        self.spec = T.Spectrogram(n_fft=n_fft, hop_length=hop, power=None)

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        x = self.spec(sensor)  # (B, 2, F, T') complex
        mag = torch.log1p(x.abs() ** 2)
        cross = x[:, 0] * x[:, 1].conj()  # (B, F, T')
        phase = torch.angle(cross)
        return torch.cat([mag, torch.cos(phase).unsqueeze(1), torch.sin(phase).unsqueeze(1)], dim=1)


class MultiScaleEncoder(nn.Module):
    """Three conv stages returning features at full, 1/2 and 1/4 scale."""

    def __init__(self, in_channels: int, base_ch: int = 32, dropout: float = 0.0) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        self.stage1 = block(in_channels, base_ch)
        self.stage2 = nn.Sequential(nn.MaxPool2d(2), block(base_ch, base_ch * 2))
        self.stage3 = nn.Sequential(nn.MaxPool2d(2), block(base_ch * 2, base_ch * 4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3


class SkipSensingCNN(nn.Module):
    """K-pose stereo sensing with phase channels and multi-scale skips.

    Task 2.3 (a) + (d). Differences from ``JointPoseCNN``:

    - **Front-end** (a): ``StereoPhaseFrontEnd`` — 4 channels including
      the inter-channel phase, instead of 2 magnitude channels.
    - **Multi-scale skips** (d): the earlier models squeeze everything
      the decoder learns about the room through one 8x8x64 latent —
      a hard ceiling on map detail. Here the encoder's three stages are
      each adaptive-pooled to the decoder resolution they feed (8x8,
      16x16, 32x32) and concatenated in, U-Net-style. Note this is
      *multi-resolution conditioning*, not a geometric U-Net: encoder
      pixels live in time-frequency, decoder pixels in room space, so
      there is no spatial correspondence to exploit — the skips widen
      the information bottleneck, nothing more.
    - Pose handling is unchanged from ``JointPoseCNN``: the sensor
      encoder is shared across poses and each scale's features are
      mean-pooled over the pose axis (permutation-invariant,
      K-agnostic). The source branch is the familiar magnitude
      ``SpectrogramEncoder`` joining at the 8x8 bottleneck.

    ~0.9M parameters (vs 232k) — still comfortably CPU-inference-sized,
    but no longer parameter-matched to the v1 models; comparisons
    against them measure the (a)+(d) package, not capacity alone. The
    stereo phase channel requires ``n_mics == 2`` (the hardware target).
    """

    def __init__(
        self,
        sensor_n_fft: int = 64,
        sensor_hop: int = 16,
        source_n_fft: int = 512,
        source_hop: int = 256,
        base_ch: int = 32,
        source_base_ch: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.front = StereoPhaseFrontEnd(n_fft=sensor_n_fft, hop=sensor_hop)
        self.encoder_s = MultiScaleEncoder(in_channels=4, base_ch=base_ch, dropout=dropout)
        self.source_spec = T.Spectrogram(n_fft=source_n_fft, hop_length=source_hop, power=2.0)
        self.encoder_u = SpectrogramEncoder(
            in_channels=1, base_ch=source_base_ch, latent_size=8, dropout=dropout
        )
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        src_c = source_base_ch * 4
        # Decoder: 8 -> 16 -> 32 -> 64 with a skip concat at each stage.
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 + src_c, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, c1 // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 2, 1, kernel_size=1),
        )

    def _pool_poses(self, f: torch.Tensor, b: int, k: int, size: int) -> torch.Tensor:
        """Adaptive-pool to (size, size), then mean over the pose axis."""
        pooled = nn.functional.adaptive_avg_pool2d(f, size)
        return pooled.reshape(b, k, *pooled.shape[1:]).mean(dim=1)

    def forward(self, sensor: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if sensor.dim() == 3:  # (B, 2, T) -> K=1
            sensor = sensor.unsqueeze(1)
        b, k, m, t = sensor.shape
        if m != 2:
            raise ValueError(f"SkipSensingCNN requires 2 mics (stereo phase); got {m}")
        f1, f2, f3 = self.encoder_s(self.front(sensor.reshape(b * k, m, t)))
        s3 = self._pool_poses(f3, b, k, 8)
        s2 = self._pool_poses(f2, b, k, 16)
        s1 = self._pool_poses(f1, b, k, 32)
        u = self.encoder_u(torch.log1p(self.source_spec(source)))  # (B, src_c, 8, 8)
        d = self.dec3(torch.cat([s3, u], dim=1))  # -> (B, c2, 16, 16)
        d = self.dec2(torch.cat([d, s2], dim=1))  # -> (B, c1, 32, 32)
        return self.dec1(torch.cat([d, s1], dim=1))  # -> (B, 1, 64, 64)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(model_type: str, n_mics: int = 2, dropout: float = 0.0) -> nn.Module:
    """Construct a model by checkpoint tag: 'dual', 'passive', 'joint', or 'skip'."""
    if model_type == "dual":
        return DualInputCNN(n_mics=n_mics, dropout=dropout)
    if model_type == "passive":
        return PassiveCNN(n_mics=n_mics, dropout=dropout)
    if model_type == "joint":
        return JointPoseCNN(n_mics=n_mics, dropout=dropout)
    if model_type == "skip":
        return SkipSensingCNN(dropout=dropout)
    raise ValueError(f"unknown model_type {model_type!r}")
