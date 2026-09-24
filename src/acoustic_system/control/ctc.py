"""Crosstalk cancellation with two laptop speakers: virtual headphones (plan 7.3).

The plant :math:`H(f)` is the 2x2 matrix from the speakers (columns, left
then right) to the ears (rows, left then right). Here it is measured in
the FDTD room (:mod:`.transfer`) instead of the free-field model of the
browser version (``web/src/control/ctc.ts``). Like the browser version, the
ears are two points :math:`2a = 17.5` cm apart with no head in the field
(head shadowing would only add natural separation).

Filters (Kirkeby et al. 1998), with the same conventions as ``ctc.ts``:

.. math::
    C(f) = H^H \\bigl(H H^H + \\beta_f I\\bigr)^{-1} e^{-j 2\\pi f \\tau},
    \\qquad \\beta_f = \\beta\\, \\overline{|H_{es}(f)|^2},

with the modelling delay :math:`\\tau = L/2` samples, a Hann window over
the whole L-tap filter, plain stereo (:math:`C = I e^{-j 2\\pi f \\tau}`)
outside the design band, and ``taps[speaker, programme]``. As
:math:`\\beta \\to 0`, :math:`H C = I e^{-j2\\pi f\\tau}`, so each ear hears
only its own programme. :math:`\\beta` bounds the filter gain where
:math:`H` is nearly singular (for a laptop: low frequencies, where the
two paths to an ear are almost equal).

Channel separation for programme :math:`p` at frequency :math:`f` is the
level at the intended ear over the level at the other ear,

.. math::
    \\text{sep}_p(f) = 20 \\log_{10}
        \\frac{|(H C)_{pp}|}{|(H C)_{\\bar p p}|}\\;\\text{dB}.

Robustness (7.3.2) is measured by evaluating fixed filters with the plant
of a displaced head. Tracking (7.3.3) re-designs the filters from the
plant at the tracked head position. Both use one measurement: the ears
are free-field points, so recording a grid of points around the head
gives the plant for every head position on the grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .beamforming import apply_fir, fir_from_weights, fir_response
from .metrics import band_energy
from .transfer import Point, Room, TransferSet, dtft, simulate_drives

EAR_RADIUS = 0.0875  # m, half the 17.5 cm ear spacing (same as ctc.ts)


def ear_points(head: Point, ear_radius: float = EAR_RADIUS) -> np.ndarray:
    """Left and right ear positions (2, 2) for a head facing +y."""
    x, y = float(head[0]), float(head[1])
    return np.array([[x - ear_radius, y], [x + ear_radius, y]])


def kirkeby_inverse(H: np.ndarray, beta: float = 0.005) -> np.ndarray:
    """Regularised inverse :math:`H^H (H H^H + \\beta_f I)^{-1}` for (F, 2, 2) plants."""
    Hh = np.conj(np.swapaxes(H, -1, -2))
    b = beta * np.mean(np.abs(H) ** 2, axis=(-1, -2))
    A = H @ Hh + b[:, None, None] * np.eye(2)
    return Hh @ np.linalg.inv(A)


def design_ctc(
    H: np.ndarray,
    bins: np.ndarray,
    n_taps: int,
    dt: float,
    *,
    beta: float = 0.005,
    delay: float | None = None,
    window: str = "hann",
    n_fft: int | None = None,
) -> np.ndarray:
    """CTC filters ``taps[speaker, programme]`` (2, 2, L).

    ``H`` (F, 2, 2) is the plant at the in-band FFT ``bins`` of the
    ``n_fft``-point grid (default ``L``). Outside the band the filters are
    plain stereo with the same modelling delay.
    """
    N = int(n_taps) if n_fft is None else int(n_fft)
    C = np.zeros((N // 2 + 1, 2, 2), dtype=np.complex128)
    C[:] = np.eye(2)
    C[bins] = kirkeby_inverse(H, beta)
    return fir_from_weights(
        C, np.arange(N // 2 + 1), n_taps, dt, delay=delay, window=window, taper_bins=0, n_fft=N
    )


def realised_plant(taps: np.ndarray, H: np.ndarray, freqs: np.ndarray, dt: float) -> np.ndarray:
    """Ear responses :math:`H(f) C(f)` (F, ear, programme) of filters on plant ``H``."""
    Cf = fir_response(taps, freqs, dt)  # (F, 2, 2)
    return H @ Cf


def separation_db(taps: np.ndarray, H: np.ndarray, freqs: np.ndarray, dt: float) -> np.ndarray:
    """Channel separation (F, 2) for programmes left and right."""
    E = np.abs(realised_plant(taps, H, freqs, dt))
    left = E[:, 0, 0] / np.maximum(E[:, 1, 0], 1e-300)
    right = E[:, 1, 1] / np.maximum(E[:, 0, 1], 1e-300)
    return 20 * np.log10(np.stack([left, right], axis=1))


def stereo_separation_db(H: np.ndarray) -> np.ndarray:
    """Natural separation (F, 2) of plain stereo (no CTC)."""
    E = np.abs(H)
    return 20 * np.log10(np.stack([E[:, 0, 0] / E[:, 1, 0], E[:, 1, 1] / E[:, 0, 1]], axis=1))


# --------------------------------------------------------------------------- #
# Plants for a grid of head positions
# --------------------------------------------------------------------------- #


def head_grid(
    room: Room,
    head: Point,
    lateral: float = 0.15,
    forward: float = 0.10,
    ear_radius: float = EAR_RADIUS,
) -> np.ndarray:
    """Cells around both ears (world coordinates) for head offsets up to ``+-lateral`` (x)
    and ``+-forward`` (y), on the grid."""
    ears = ear_points(head, ear_radius)
    ij = np.rint(room.cell_float(ears)).astype(int)
    k = int(round(lateral / room.dx))
    m = int(round(forward / room.dx))
    xs = np.arange(ij[0, 0] - k, ij[1, 0] + k + 1)
    ys = np.arange(ij[0, 1] - m, ij[0, 1] + m + 1)
    gi, gj = np.meshgrid(xs, ys, indexing="ij")
    return room.world(np.stack([gi.ravel(), gj.ravel()], axis=1))


def ear_indices(
    points: np.ndarray, head: Point, ear_radius: float = EAR_RADIUS, tol: float = 1e-3
) -> np.ndarray:
    """Indices of the left and right ear in ``points`` (raises if not on the grid)."""
    idx = []
    for e in ear_points(head, ear_radius):
        d = np.linalg.norm(points - e, axis=1)
        k = int(np.argmin(d))
        if d[k] > tol:
            raise ValueError(f"ear {e} is not a measured point (nearest {d[k] * 100:.2f} cm)")
        idx.append(k)
    return np.array(idx)


def plant_at(ts: TransferSet, freqs: np.ndarray, ears: np.ndarray) -> np.ndarray:
    """Plant (F, ear, speaker) from a transfer set and the two ear indices."""
    return ts.at(freqs)[:, ears, :]


@dataclass
class CtcDesign:
    """Filters and design-grid data for one head position."""

    taps: np.ndarray  # (2, 2, L)
    head: np.ndarray  # (2,)
    freqs: np.ndarray  # in-band design frequencies [Hz]
    H: np.ndarray  # (F, 2, 2) design plant


class TrackedCtc:
    """CTC filters that follow a tracked head (plan 7.3.3).

    Holds the transfer functions from both speakers to a grid of points
    around the nominal head (:func:`head_grid`) and re-designs the filters
    for any head position on that grid. The design plant at the new
    position is read from the measurement, so no re-simulation is needed.

    The defaults differ from the ``ctc.ts`` convention (L/2 delay, Hann):
    in a room the inverse has a long causal tail, and L = 4096 taps
    (150 ms at 27.4 kHz), an L/4 delay, a Tukey window and a 4x
    oversampled design grid keep more of it (``docs/control.md``).
    Separation is evaluated on :attr:`eval_freqs`, the oversampled design
    grid inside ``eval_band`` (1.7 Hz spacing by default, four points per
    FIR bin, so it also sees the response between the FIR's own bins).
    ``eval_band`` sits inside the design band so the switch to plain stereo
    at the design band edges does not count.
    """

    def __init__(
        self,
        ts: TransferSet,
        *,
        n_taps: int = 4096,
        band: tuple[float, float] = (200.0, 1800.0),
        beta: float = 0.001,
        delay: float | None = None,
        window: str = "tukey",
        oversample: int = 4,
        eval_band: tuple[float, float] = (300.0, 1500.0),
        ear_radius: float = EAR_RADIUS,
    ) -> None:
        self.ts = ts
        self.n_taps = int(n_taps)
        self.n_fft = self.n_taps * int(oversample)
        self.band = band
        self.beta = float(beta)
        self.delay = self.n_taps / 4 if delay is None else float(delay)
        self.window = window
        self.ear_radius = float(ear_radius)
        f = np.fft.rfftfreq(self.n_fft, ts.dt)
        self.bins = np.where((f >= band[0]) & (f <= band[1]))[0]
        self.freqs = f[self.bins]
        self._H = ts.at(self.freqs)  # (F, M, 2)
        self._eval = (self.freqs >= eval_band[0]) & (self.freqs <= eval_band[1])
        self.eval_freqs = self.freqs[self._eval]
        self._cache: dict[tuple[float, float], CtcDesign] = {}

    def plant(self, head: Point, freqs: "np.ndarray | None" = None) -> np.ndarray:
        ears = ear_indices(self.ts.points, head, self.ear_radius)
        if freqs is None:
            return self._H[:, ears, :]
        return plant_at(self.ts, freqs, ears)

    def design(self, head: Point) -> CtcDesign:
        key = (round(float(head[0]), 6), round(float(head[1]), 6))
        if key not in self._cache:
            H = self.plant(head)
            taps = design_ctc(
                H,
                self.bins,
                self.n_taps,
                self.ts.dt,
                beta=self.beta,
                delay=self.delay,
                window=self.window,
                n_fft=self.n_fft,
            )
            self._cache[key] = CtcDesign(taps, np.asarray(head, float), self.freqs, H)
        return self._cache[key]

    def separation(
        self, design: CtcDesign, head: Point, freqs: "np.ndarray | None" = None
    ) -> np.ndarray:
        """Separation (F, 2) of ``design`` when the head is actually at ``head``.

        Evaluated on :attr:`eval_freqs` unless ``freqs`` is given.
        """
        if freqs is None:
            f = self.eval_freqs
            H = self.plant(head)[self._eval]
        else:
            f = np.asarray(freqs, dtype=np.float64)
            H = self.plant(head, f)
        return separation_db(design.taps, H, f, self.ts.dt)


def displacement_sweep(
    tracker: TrackedCtc,
    head0: Point,
    offsets: np.ndarray,
    *,
    tracked: bool = False,
) -> np.ndarray:
    """Mean in-band separation (dB, both programmes) for head offsets (K, 2).

    ``tracked=False`` keeps the filters designed at ``head0``;
    ``tracked=True`` re-designs them at each displaced position.
    """
    base = tracker.design(head0)
    out = np.empty(len(offsets))
    for k, off in enumerate(np.asarray(offsets, dtype=np.float64)):
        head = np.asarray(head0, dtype=np.float64) + off
        d = tracker.design(head) if tracked else base
        out[k] = float(np.mean(tracker.separation(d, head)))
    return out


# --------------------------------------------------------------------------- #
# Time-domain verification through the engine
# --------------------------------------------------------------------------- #


@dataclass
class CtcCheck:
    """Separation of a CTC design measured by playing it in the engine."""

    predicted_db: float  # band-energy separation from H(f) C(f) and the programme
    measured_db: float  # band-energy separation of the engine recordings
    freqs: np.ndarray
    measured_spectrum_db: np.ndarray  # per-frequency separation of the recordings
    rec: np.ndarray  # (2, T) ear recordings


def verify_ctc(
    room: Room,
    ts: TransferSet,
    taps: np.ndarray,
    head: Point,
    programme: np.ndarray,
    band: tuple[float, float],
    *,
    channel: int = 0,
    ear_radius: float = EAR_RADIUS,
    tail: float = 0.15,
) -> CtcCheck:
    """Play ``programme`` on one channel through the CTC filters in the engine."""
    ears = ear_points(head, ear_radius)
    drives = apply_fir(taps[:, channel], programme)  # (2 speakers, T)
    steps = drives.shape[1] + int(np.ceil(tail / ts.dt))
    rec = simulate_drives(room, ts.speakers, drives, ears, steps).rec
    other = 1 - channel
    meas = 10 * np.log10(
        band_energy(rec[channel], ts.dt, band) / band_energy(rec[other], ts.dt, band)
    )
    f_all = np.fft.rfftfreq(ts.steps, ts.dt)
    f = f_all[(f_all >= band[0]) & (f_all <= band[1])]
    idx = ear_indices(ts.points, head, ear_radius)
    E = realised_plant(taps, plant_at(ts, f, idx), f, ts.dt)[:, :, channel]
    w = np.abs(dtft(programme, f, ts.dt)) ** 2
    pred = 10 * np.log10(
        np.sum(w * np.abs(E[:, channel]) ** 2) / np.sum(w * np.abs(E[:, other]) ** 2)
    )
    n = rec.shape[1]
    Y = np.fft.rfft(rec.astype(np.float64), n=n, axis=-1)
    fr = np.fft.rfftfreq(n, ts.dt)
    sel = (fr >= band[0]) & (fr <= band[1])
    spec = 20 * np.log10(np.abs(Y[channel, sel]) / np.maximum(np.abs(Y[other, sel]), 1e-300))
    return CtcCheck(float(pred), float(meas), fr[sel], spec, rec)


__all__ = [
    "EAR_RADIUS",
    "CtcCheck",
    "CtcDesign",
    "TrackedCtc",
    "design_ctc",
    "displacement_sweep",
    "ear_indices",
    "ear_points",
    "head_grid",
    "kirkeby_inverse",
    "plant_at",
    "realised_plant",
    "separation_db",
    "stereo_separation_db",
    "verify_ctc",
]
