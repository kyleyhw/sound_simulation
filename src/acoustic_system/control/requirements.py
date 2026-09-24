"""Sensing requirements for sound-zone control (plan 7.5).

A controller designed from the room it *believes in* is played in the
room that *exists*. Sensing (Phase 6) delivers an estimate of the room's
walls, absorption and furniture. The controller is designed from
transfer functions simulated in that estimate, :math:`\\hat H`, and the
achieved contrast is scored with the true transfer functions :math:`H`:

.. math::
    q = \\operatorname{ACC}(\\hat H_B, \\hat H_D), \\qquad
    C_{\\text{achieved}} = 10\\log_{10}
       \\frac{\\sum_f \\|H_B(f) q(f)\\|^2 / M_B}{\\sum_f \\|H_D(f) q(f)\\|^2 / M_D}.

Sweeping the size of one kind of error at a time gives a curve of
achieved contrast against sensing error. Its crossing of a target (e.g.
10 dB) is the accuracy that sensing must deliver (7.5.2). Errors:

* **wall position**: every wall moves by :math:`\\pm\\delta` (a fixed or
  random pattern of outward and inward moves). This is quantised to the
  grid (2.5 cm cells).
* **absorption**: the estimated admittance :math:`\\hat\\beta = s\\,\\beta`.
* **furniture**: boxes shifted by a vector, or missing from the estimate.

Speakers and zones stay at the same world positions in every estimate:
the array and the listener are located relative to each other, and the
error is in the room around them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from .beamforming import acoustic_contrast_control
from .metrics import band_contrast_db, contrast_spectrum_db
from .transfer import Room, TransferSet, measure_transfer


@dataclass(frozen=True)
class SensingError:
    """One sensing error applied to a room estimate.

    ``wall_m`` moves each wall by ``signs[i] * wall_m`` (positive is
    outward), in the order x-low, x-high, y-low, y-high.
    """

    wall_m: float = 0.0
    signs: tuple[int, int, int, int] = (1, -1, -1, 1)
    beta_scale: float = 1.0
    box_shift_m: tuple[float, float] = (0.0, 0.0)
    drop_boxes: bool = False
    label: str = ""

    @property
    def exact(self) -> bool:
        """True when the estimate equals the true room."""
        return (
            self.wall_m == 0.0
            and self.beta_scale == 1.0
            and tuple(self.box_shift_m) == (0.0, 0.0)
            and not self.drop_boxes
        )


def estimated_room(room: Room, err: SensingError) -> Room:
    """The room as a sensing system with error ``err`` would report it."""
    d = float(err.wall_m)
    s = err.signs
    x0 = room.origin[0] - s[0] * d
    x1 = room.origin[0] + room.size[0] + s[1] * d
    y0 = room.origin[1] - s[2] * d
    y1 = room.origin[1] + room.size[1] + s[3] * d
    boxes: tuple = ()
    if not err.drop_boxes:
        boxes = tuple(b.shifted(*err.box_shift_m) for b in room.boxes)
    return replace(
        room,
        origin=(x0, y0),
        size=(x1 - x0, y1 - y0),
        beta=room.beta * float(err.beta_scale),
        boxes=boxes,
    )


@dataclass
class ErrorResult:
    """Contrast achieved in the true room by a design from an estimate."""

    error: SensingError
    band_db: float  # broadband contrast in the true room
    predicted_db: float  # what the (wrong) model predicted for itself
    spectrum_db: np.ndarray  # per-frequency contrast in the true room
    sub_band_db: dict[str, float] = field(default_factory=dict)


@dataclass
class SensingStudy:
    """A true room, a speaker array and two zones; evaluates room estimates."""

    room: Room
    speakers: np.ndarray
    bright: np.ndarray  # (M_B, 2) world points
    dark: np.ndarray  # (M_D, 2)
    band: tuple[float, float] = (300.0, 1500.0)
    reg: float = 1e-3
    duration: float = 0.15
    excitation: tuple[float, float] = (100.0, 2000.0)
    freq_step: int = 2  # use every n-th FFT bin of the records
    sub_bands: tuple[tuple[float, float], ...] = ((300.0, 600.0), (600.0, 1000.0), (1000.0, 1500.0))

    def __post_init__(self) -> None:
        self.points = np.concatenate([self.bright, self.dark])
        self.ib = np.arange(len(self.bright))
        self.id = np.arange(len(self.bright), len(self.points))
        self.truth = self.measure(self.room)
        f = np.fft.rfftfreq(self.truth.steps, self.truth.dt)
        sel = np.where((f >= self.band[0]) & (f <= self.band[1]))[0][:: self.freq_step]
        self.freqs = f[sel]
        self.H_true = self.truth.at(self.freqs)

    def measure(self, room: Room) -> TransferSet:
        return measure_transfer(
            room, self.speakers, self.points, duration=self.duration, band=self.excitation
        )

    def design(self, H: np.ndarray) -> np.ndarray:
        return acoustic_contrast_control(H[:, self.ib], H[:, self.id], reg=self.reg)

    def score(self, q: np.ndarray, H: "np.ndarray | None" = None) -> tuple[float, np.ndarray]:
        H = self.H_true if H is None else H
        Hb, Hd = H[:, self.ib], H[:, self.id]
        return band_contrast_db(Hb, Hd, q), contrast_spectrum_db(Hb, Hd, q)

    def estimate(self, err: SensingError) -> np.ndarray:
        """Transfer functions (F, M, S) simulated in the room estimate ``err``."""
        if err.exact:
            return self.H_true
        return self.measure(estimated_room(self.room, err)).at(self.freqs)

    def evaluate(self, err: SensingError) -> ErrorResult:
        """Design from the estimate ``err`` describes, score in the true room."""
        return self.evaluate_model(self.estimate(err), err)

    def evaluate_model(self, H_est: np.ndarray, err: SensingError) -> ErrorResult:
        q = self.design(H_est)
        band_db, spec = self.score(q)
        pred, _ = self.score(q, H_est)
        subs = {}
        for lo, hi in self.sub_bands:
            m = (self.freqs >= lo) & (self.freqs <= hi)
            if m.any():
                Hb, Hd = self.H_true[m][:, self.ib], self.H_true[m][:, self.id]
                subs[f"{lo:.0f}-{hi:.0f}"] = band_contrast_db(Hb, Hd, q[m])
        return ErrorResult(err, band_db, pred, spec, subs)

    def sweep(self, errors: Sequence[SensingError]) -> list[ErrorResult]:
        return [self.evaluate(e) for e in errors]


def wall_errors(
    deltas_m: "Sequence[float] | np.ndarray", n_patterns: int = 1, seed: int = 0
) -> list[SensingError]:
    """Wall-position errors: for each delta, ``n_patterns`` random sign patterns.

    The first pattern is the fixed default (+, -, -, +): the room is
    estimated shifted, not only resized.
    """
    rng = np.random.default_rng(seed)
    out = []
    for d in deltas_m:
        for k in range(n_patterns):
            if k == 0:
                signs = (1, -1, -1, 1)
            else:
                a, b, c, e = (int(v) for v in rng.choice([-1, 1], 4))
                signs = (a, b, c, e)
            out.append(SensingError(wall_m=float(d), signs=signs, label=f"wall {d * 100:.1f} cm"))
    return out


def threshold_crossing(
    x: "Sequence[float] | np.ndarray", y: "Sequence[float] | np.ndarray", target: float
) -> float:
    """Smallest x where the (piecewise-linear) curve y(x) first drops below ``target``.

    Returns ``inf`` if it never does and ``x[0]`` if it starts below.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if ys[0] < target:
        return float(xs[0])
    for k in range(1, len(xs)):
        if ys[k] < target:
            t = (ys[k - 1] - target) / (ys[k - 1] - ys[k])
            return float(xs[k - 1] + t * (xs[k] - xs[k - 1]))
    return float("inf")


__all__ = [
    "ErrorResult",
    "SensingError",
    "SensingStudy",
    "estimated_room",
    "threshold_crossing",
    "wall_errors",
]
