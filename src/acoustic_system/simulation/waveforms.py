from dataclasses import dataclass, field

import numpy as np


class Waveform:
    """Base class for callable driver waveforms p_src(t)."""

    def __call__(self, t: float) -> float:
        raise NotImplementedError


@dataclass
class Cosine(Waveform):
    """Continuous monochromatic source: p_src(t) = A cos(2 pi f t).

    For a leap-frog FDTD with timestep dt, the source must be sampled
    well below Nyquist: f * dt < 0.5; ten or more samples per period
    (f * dt < 0.1) is comfortable.
    """

    frequency: float = 0.05
    amplitude: float = 1.0

    def __call__(self, t: float) -> float:
        return self.amplitude * np.cos(2.0 * np.pi * self.frequency * t)


@dataclass
class GaussianPulse(Waveform):
    """Single-lobe Gaussian: p_src(t) = A exp(-(t - t0)^2 / (2 sigma^2)).

    Generates a low-pass DC-rich impulse. Spectrally bright but contains a
    persistent DC offset, which can leave a quasi-static residual in
    closed (hard-walled) domains.
    """

    amplitude: float = 1.0
    center_time: float = 0.5
    width: float = 0.1

    def __call__(self, t: float) -> float:
        return self.amplitude * np.exp(
            -((t - self.center_time) ** 2) / (2.0 * self.width**2)
        )


@dataclass
class RickerWavelet(Waveform):
    """Ricker (Mexican-hat) wavelet — second derivative of a Gaussian.

    p_src(t) = A (1 - 2 (pi f (t - t0))^2) exp(-(pi f (t - t0))^2)

    Standard FDTD excitation: zero mean (no DC pump-up), single dominant
    frequency f, finite duration ~ 2/f around t0. Choose t0 >= 1/f so the
    pulse starts near zero and ends near zero, leaving the field clean.
    """

    amplitude: float = 1.0
    frequency: float = 0.1
    delay: float = 20.0

    def __call__(self, t: float) -> float:
        arg = np.pi * self.frequency * (t - self.delay)
        arg2 = arg * arg
        return self.amplitude * (1.0 - 2.0 * arg2) * np.exp(-arg2)


@dataclass
class AudioFileWaveform(Waveform):
    """Drive the simulation from an audio file (WAV).

    Mathematical formulation
    ------------------------
    Given a discrete audio signal $u_k$ sampled at rate $f_a$ (samples per
    unit of simulation time), the continuous source value at time $t$ is
    constructed by linear interpolation between adjacent samples:

    $$
    p_{\\text{src}}(t) = A \\cdot \\bigl[(1 - \\alpha)\\, u_{k} + \\alpha\\, u_{k+1}\\bigr]
    \\quad\\text{with}\\quad k = \\lfloor f_a (t - t_0) \\rfloor,
    \\quad \\alpha = f_a (t - t_0) - k,
    $$

    for $t \\ge t_0$ and $k + 1 < N$ (audio length). Outside that support
    window the source returns 0.

    Notes
    -----
    - Audio sample-rate vs simulation rate: the simulation's timestep
      ``dt_sim`` and the audio's native sample rate ``fs_native`` need not
      coincide. The audio is read once at its native rate; the constructor
      converts to a sample rate ``fs_audio`` expressed in *simulation time
      units* via ``fs_audio = fs_native * sim_time_per_audio_second``. The
      ratio is encoded in the constructor argument ``sim_time_per_second``
      (default 1.0, which means one second of audio corresponds to one unit
      of simulation time — appropriate for the dimensionless ``c=1, dx=1``
      sandbox). For physical units, pass the actual simulation seconds per
      audio second.
    - Linear interpolation is preferred over nearest-neighbour because the
      FDTD timestep is generally a non-integer multiple of the audio
      sample period; nearest-neighbour introduces audible quantisation in
      the high-frequency content of the source.
    - The audio array is stored on a private (underscore-prefixed)
      attribute so it is excluded from the HDF5 attribute dump performed
      by ``data_io.SaveSimulationResults`` (which iterates
      ``waveform.__dict__`` and would otherwise try to write the entire
      raw audio array as an attribute, exceeding HDF5's 64 KiB attr size
      limit on any non-trivial clip).
    """

    path: str = ""
    amplitude: float = 1.0
    delay: float = 0.0
    # Conversion factor: simulation-time seconds per second of audio.
    # 1.0 means audio plays at native speed in dimensionless sim units.
    sim_time_per_second: float = 1.0

    # Mutable internal state — kept off the dataclass field list so it
    # does not participate in repr/eq, and kept underscore-private so
    # data_io.py skips it. ``init=False`` lets __post_init__ populate it.
    _samples: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32),
        init=False,
        repr=False,
    )
    _fs_audio: float = field(default=0.0, init=False, repr=False)
    _native_fs: float = field(default=0.0, init=False, repr=False)
    _duration: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.path:
            return
        # Local import: scipy.io is heavy and only needed when the user
        # actually opts into audio-file sources.
        from scipy.io import wavfile

        fs_native, samples = wavfile.read(self.path)

        # Normalise to float32 in [-1, 1] regardless of source bit depth.
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype == np.int32:
            samples = samples.astype(np.float32) / 2147483648.0
        elif samples.dtype == np.uint8:
            samples = (samples.astype(np.float32) - 128.0) / 128.0
        else:
            samples = samples.astype(np.float32)

        # Stereo -> mono by averaging channels (preserves DC, keeps
        # amplitude in roughly the same range as the loudest channel).
        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        self._samples = np.ascontiguousarray(samples, dtype=np.float32)
        self._native_fs = float(fs_native)
        # Audio samples per *simulation-time* unit.
        self._fs_audio = self._native_fs * float(self.sim_time_per_second)
        # Audible duration in simulation time.
        self._duration = (
            len(self._samples) / self._fs_audio if self._fs_audio > 0 else 0.0
        )

    def __call__(self, t: float) -> float:
        if self._samples.size == 0 or self._fs_audio <= 0.0:
            return 0.0
        t_audio = t - self.delay
        if t_audio < 0.0 or t_audio >= self._duration:
            return 0.0
        idx_f = t_audio * self._fs_audio
        idx0 = int(idx_f)
        # idx_f < (N-1) because t_audio < duration; safety clamp anyway.
        if idx0 >= self._samples.shape[0] - 1:
            return 0.0
        frac = idx_f - idx0
        a = self._samples[idx0]
        b = self._samples[idx0 + 1]
        return float(self.amplitude * ((1.0 - frac) * a + frac * b))


waveform_registry = {
    "Cosine": Cosine,
    "GaussianPulse": GaussianPulse,
    "RickerWavelet": RickerWavelet,
    "AudioFileWaveform": AudioFileWaveform,
}
