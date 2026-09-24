"""Speaker-to-point transfer functions from the FDTD engine (plan 7.1.1).

Every controller in this package is designed from a matrix of transfer
functions :math:`H_{ms}(f)`, the complex pressure at control point
:math:`m` per unit drive of speaker :math:`s`. They are measured the way
an acoustician would measure a room, but in the simulator:

1. drive speaker :math:`s` alone with a band-limited impulse :math:`w[n]`;
2. record the pressure :math:`y_{ms}[n]` at every point;
3. deconvolve by the source spectrum,
   :math:`H_{ms}(f) = Y_{ms}(f) / W(f)`, inside the band where
   :math:`|W|` is well above zero.

Drive convention (the engine's soft source): the drive sample
:math:`u[n]` is *added* to :math:`p^{n+1}` at the speaker cell after the
wall update, and the recording :math:`y[n]` is :math:`p^{n+1}` at the
probe. The scheme is linear and time invariant in :math:`u`, so
:math:`y = h * u` exactly, with :math:`h` the response to a unit
injection at :math:`n = 0`, and :math:`H(f)` is the DTFT of :math:`h`:

.. math::
    H(f) = \\sum_{n \\ge 0} h[n]\\, e^{-j 2\\pi f n \\Delta t}.

Any drive signal designed from :math:`H` therefore reproduces the
prediction when it is played through the engine, up to float32 round-off
and the truncation of :math:`h` at the recording length (reported by
:meth:`TransferSet.tail_db`).

Scenes are 2D and in SI units: a :class:`Room` maps world coordinates in
metres to cells (axis 0 is x, axis 1 is y) with cell size ``dx``
(default 2.5 cm, ``units.LAPTOP_ROOM``), sound speed 343 m/s and Courant
number 0.5, so :math:`\\Delta t = 36.4\\,\\mu s` (27.4 kHz). Walls are
the engine's general-path boundaries: ``"absorb"`` (locally reacting
impedance, normalised admittance :math:`\\beta`), ``"rigid"``, ``"soft"``
or ``"cpml"`` (anechoic; the grid is padded by the layer so the room's
interior is unchanged). Furniture and wall treatment are :class:`Box`
material blocks (``physics.MATERIALS`` ids).

Two engines: ``"numba"`` (``Simulate``, the reference) runs the speakers
one after another; ``"torch"`` (``TorchFDTD``) runs all speakers as one
batch, which is what a GPU would do, but supports only p = 0, rigid,
impedance and sponge walls.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import firwin

from ..simulation.physics import MATERIALS, RIGID
from ..simulation.simulate import Simulate
from ..simulation.units import LAPTOP_ROOM, PhysicalScale

# A world position (x, y) in metres: a pair of floats or a length-2 array.
Point = Union[Sequence[float], np.ndarray]

# --------------------------------------------------------------------------- #
# Scene geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Box:
    """Axis-aligned block of one material, in world metres (x0 < x1, y0 < y1)."""

    x0: float
    y0: float
    x1: float
    y1: float
    material: int = RIGID

    def shifted(self, dx: float, dy: float) -> "Box":
        return replace(self, x0=self.x0 + dx, x1=self.x1 + dx, y0=self.y0 + dy, y1=self.y1 + dy)


@dataclass(frozen=True)
class Room:
    """A 2D room in SI units.

    Cell ``(i, j)`` sits at world position ``origin + (i, j) * dx``. The
    outer walls lie just beyond the first and last cells, so the room spans
    ``[origin, origin + size]`` to within half a cell. ``boundary`` is any
    ``Simulate`` outer kind; ``beta`` is the normalised admittance of
    ``"absorb"`` walls (normal-incidence absorption
    :math:`1 - ((1-\\beta)/(1+\\beta))^2`: 0.33 for 0.1, 0.71 for 0.3).
    """

    size: tuple[float, float] = (3.0, 2.4)
    dx: float = LAPTOP_ROOM.dx
    boundary: str = "absorb"
    beta: float = 0.3
    boxes: tuple[Box, ...] = ()
    origin: tuple[float, float] = (0.0, 0.0)
    c: float = LAPTOP_ROOM.c
    courant: float = LAPTOP_ROOM.courant
    cpml_cells: int = 12

    @property
    def scale(self) -> PhysicalScale:
        return PhysicalScale(dx=self.dx, c=self.c)

    @property
    def pad(self) -> int:
        # Absorbing layers eat cells inside the grid; pad so the room's
        # interior (the part the scene is drawn in) keeps its size.
        return self.cpml_cells if self.boundary in ("cpml", "sponge") else 0

    @property
    def shape(self) -> tuple[int, int]:
        nx = int(round(self.size[0] / self.dx)) + 1 + 2 * self.pad
        ny = int(round(self.size[1] / self.dx)) + 1 + 2 * self.pad
        return (nx, ny)

    @property
    def timestep(self) -> float:
        # Same rule as Simulate(timestep=None): kappa = min(courant, 0.95/sqrt 2).
        kappa = min(self.courant, 0.95 / np.sqrt(2.0))
        return float(kappa * self.dx / self.c)

    @property
    def fs(self) -> float:
        return 1.0 / self.timestep

    def cell_float(self, xy: ArrayLike) -> np.ndarray:
        """Fractional cell coordinates of world points (..., 2)."""
        xy = np.asarray(xy, dtype=np.float64)
        return (xy - np.asarray(self.origin)) / self.dx + self.pad

    def cells(self, xy: ArrayLike) -> np.ndarray:
        """Integer cells (M, 2) of world points (M, 2); raises if outside the air."""
        ij = np.rint(self.cell_float(np.atleast_2d(xy))).astype(np.int64)
        nx, ny = self.shape
        lo = self.pad
        bad = (ij[:, 0] < lo) | (ij[:, 0] >= nx - lo) | (ij[:, 1] < lo) | (ij[:, 1] >= ny - lo)
        if bad.any():
            raise ValueError(f"points outside the room: {np.atleast_2d(xy)[bad][:3]}")
        mat = self.material_map()
        solid = mat[ij[:, 0], ij[:, 1]] != 0
        if solid.any():
            raise ValueError(f"points inside a wall or box: {np.atleast_2d(xy)[solid][:3]}")
        return ij

    def world(self, ij: np.ndarray) -> np.ndarray:
        """World coordinates of integer cells (M, 2)."""
        return (np.asarray(ij, dtype=np.float64) - self.pad) * self.dx + np.asarray(self.origin)

    def material_map(self) -> np.ndarray:
        """uint8 material ids (``physics.MATERIALS``) for the boxes."""
        m = np.zeros(self.shape, dtype=np.uint8)
        if not self.boxes:
            return m
        nx, ny = self.shape
        xs = self.world(np.stack([np.arange(nx), np.zeros(nx)], axis=1))[:, 0]
        ys = self.world(np.stack([np.zeros(ny), np.arange(ny)], axis=1))[:, 1]
        eps = 1e-9
        for b in self.boxes:
            if not 0 <= b.material < len(MATERIALS):
                raise ValueError(f"unknown material id {b.material}")
            ix = (xs >= b.x0 - eps) & (xs <= b.x1 + eps)
            iy = (ys >= b.y0 - eps) & (ys <= b.y1 + eps)
            m[np.ix_(ix, iy)] = b.material
        return m

    def make_sim(self) -> Simulate:
        """A fresh ``Simulate`` for this room (general path unless p = 0 walls)."""
        sim = Simulate(
            self.shape,
            wavespeed=self.c,
            gridstep=self.dx,
            courant=self.courant,
            boundary=self.boundary,
            boundary_beta=self.beta,
            cpml_cells=self.cpml_cells,
        )
        if self.boxes:
            sim.set_material_map(self.material_map())
        return sim


def disk_points(room: Room, centre: Point, radius: float) -> np.ndarray:
    """World coordinates (M, 2) of every air cell within ``radius`` of ``centre``."""
    c = room.cell_float(centre)
    r = radius / room.dx
    i0, i1 = int(np.floor(c[0] - r)), int(np.ceil(c[0] + r))
    j0, j1 = int(np.floor(c[1] - r)), int(np.ceil(c[1] + r))
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    keep = (ii - c[0]) ** 2 + (jj - c[1]) ** 2 <= r * r + 1e-9
    ij = np.stack([ii[keep], jj[keep]], axis=1)
    mat = room.material_map()
    nx, ny = room.shape
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    ij = ij[inside]
    ij = ij[mat[ij[:, 0], ij[:, 1]] == 0]
    return room.world(ij)


def grid_points(room: Room, centre: Point, half_widths: Point) -> np.ndarray:
    """World coordinates of every cell in a rectangle ``centre +- half_widths``."""
    c = np.rint(room.cell_float(centre)).astype(int)
    hx = int(round(half_widths[0] / room.dx))
    hy = int(round(half_widths[1] / room.dx))
    ii, jj = np.meshgrid(np.arange(-hx, hx + 1), np.arange(-hy, hy + 1), indexing="ij")
    ij = np.stack([ii.ravel() + c[0], jj.ravel() + c[1]], axis=1)
    return room.world(ij)


# --------------------------------------------------------------------------- #
# Running the engine with arbitrary drive signals
# --------------------------------------------------------------------------- #


@dataclass
class DriveResult:
    """Recordings of one engine run with several driven speakers."""

    rec: np.ndarray  # (M, steps) float32 pressure at the probes, y[n] = p^{n+1}
    energy: "np.ndarray | None" = None  # (nx, ny) sum of p^2 over the energy window
    dt: float = 0.0


def simulate_drives(
    room: Room,
    sources: "np.ndarray | Sequence[Sequence[float]]",
    drives: np.ndarray,
    probes: "np.ndarray | Sequence[Sequence[float]]",
    steps: int | None = None,
    energy_window: "tuple[int, int] | None" = None,
) -> DriveResult:
    """Run the numba engine with speaker ``s`` playing ``drives[s]``.

    Parameters
    ----------
    room
        The scene.
    sources, probes
        World coordinates (S, 2) and (M, 2).
    drives
        (S, T) drive samples; sample :math:`u_s[n]` is added to
        :math:`p^{n+1}` at the speaker cell (after the wall update, exactly
        like a ``Driver``). Drives are zero after ``T``.
    steps
        Number of steps (default ``T``).
    energy_window
        ``(n0, n1)``: also return :math:`\\sum_{n_0 \\le n < n_1} (p^{n+1})^2`
        over the whole grid (loudness maps, quiet-zone sizes).
    """
    drives = np.atleast_2d(np.asarray(drives, dtype=np.float32))
    src = room.cells(sources)
    prb = room.cells(probes)
    if drives.shape[0] != len(src):
        raise ValueError(f"{len(src)} sources but {drives.shape[0]} drive signals")
    steps = drives.shape[1] if steps is None else int(steps)
    unique = len({tuple(c) for c in src}) == len(src)
    si, sj = src[:, 0], src[:, 1]
    pi, pj = prb[:, 0], prb[:, 1]
    sim = room.make_sim()
    rec = np.empty((len(prb), steps), dtype=np.float32)
    energy = np.zeros(room.shape, dtype=np.float64) if energy_window is not None else None
    n0, n1 = energy_window if energy_window is not None else (0, 0)
    T = drives.shape[1]
    for n in range(steps):
        sim.step()
        p = sim.p
        if n < T:
            if unique:
                p[si, sj] += drives[:, n]
            else:
                np.add.at(p, (si, sj), drives[:, n])
        rec[:, n] = p[pi, pj]
        if energy is not None and n0 <= n < n1:
            energy += np.square(p, dtype=np.float64)
    return DriveResult(rec=rec, energy=energy, dt=room.timestep)


def bandpass_pulse(dt: float, band: tuple[float, float], taps: int = 511) -> np.ndarray:
    """Band-limited excitation: a Blackman-windowed band-pass FIR, unit peak.

    Its spectrum is flat inside ``band`` and falls off smoothly outside it,
    so the deconvolution :math:`Y/W` is well conditioned in the band while
    the grid's badly dispersed high frequencies are never excited.
    """
    fs = 1.0 / dt
    n = int(taps) | 1
    w = firwin(n, list(band), pass_zero=False, fs=fs, window="blackman")
    # The windowed design leaks about -40 dB at DC. A closed 2D room
    # integrates DC (the recorded tail then decays very slowly), so remove
    # it exactly with a matching Blackman bump.
    bump = np.blackman(n)
    w = w - bump * (w.sum() / bump.sum())
    return (w / np.abs(w).max()).astype(np.float64)


# --------------------------------------------------------------------------- #
# Transfer functions
# --------------------------------------------------------------------------- #


def dtft(x: np.ndarray, freqs: np.ndarray, dt: float, chunk: int = 64) -> np.ndarray:
    """DTFT of ``x`` (..., T) at arbitrary frequencies [Hz]: (..., F) complex."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[-1]
    flat = x.reshape(-1, T)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    out = np.empty((flat.shape[0], len(freqs)), dtype=np.complex128)
    n = np.arange(T)
    for k in range(0, len(freqs), chunk):
        f = freqs[k : k + chunk]
        E = np.exp(-2j * np.pi * np.outer(n, f) * dt)
        out[:, k : k + chunk] = flat @ E
    return out.reshape(x.shape[:-1] + (len(freqs),))


@dataclass
class TransferSet:
    """Measured transfer functions from S speakers to M points.

    ``rec[s, m]`` is the recording at point ``m`` when speaker ``s`` played
    ``pulse``. :meth:`at` returns :math:`H(f)` of shape (F, M, S).
    """

    dt: float
    speakers: np.ndarray  # (S, 2) world metres
    points: np.ndarray  # (M, 2) world metres
    rec: np.ndarray  # (S, M, T) float32
    pulse: np.ndarray  # (T,) excitation, zero padded to the record length
    band: tuple[float, float]  # excitation band [Hz]
    room: "Room | None" = None
    _rfft: "np.ndarray | None" = field(default=None, repr=False)

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def steps(self) -> int:
        return int(self.rec.shape[-1])

    def _spectra(self, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Exact DTFT samples. On the record's own FFT grid (k / (T dt)) use
        # the cached rfft; on any coarser-spaced grid k / (N dt) with N >= T
        # use a zero-padded rfft (still exact); otherwise a direct DTFT.
        T = self.steps

        def on(n: int) -> bool:
            b = freqs * n * self.dt
            return bool(np.all(np.abs(b - np.rint(b)) < 1e-6) and np.all(b >= 0))

        if on(T):
            if self._rfft is None:
                self._rfft = np.fft.rfft(self.rec.astype(np.float64), axis=-1)
            k = np.rint(freqs * T * self.dt).astype(int)
            return self._rfft[..., k], np.fft.rfft(self.pulse, n=T)[k]
        u = np.unique(freqs)
        spacing = float(np.min(np.diff(u))) if len(u) > 1 else float(u[0])
        n = int(round(1.0 / (spacing * self.dt))) if spacing > 0 else 0
        if n >= T and on(n):
            k = np.rint(freqs * n * self.dt).astype(int)
            Y = np.fft.rfft(self.rec.astype(np.float64), n=n, axis=-1)[..., k]
            return Y, np.fft.rfft(self.pulse, n=n)[k]
        return dtft(self.rec, freqs, self.dt), dtft(self.pulse, freqs, self.dt)

    def at(self, freqs: "np.ndarray | Point") -> np.ndarray:
        """Transfer functions :math:`H[f, m, s]` at ``freqs`` [Hz] (complex)."""
        freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        Y, W = self._spectra(freqs)
        wmax = np.abs(np.fft.rfft(self.pulse)).max()
        if np.any(np.abs(W) < 1e-3 * wmax):
            raise ValueError(
                f"frequencies outside the excitation band {self.band} (|W| < -60 dB): "
                f"{freqs[np.abs(W) < 1e-3 * wmax][:3]}"
            )
        H = Y / W  # (S, M, F)
        return np.transpose(H, (2, 1, 0))

    def impulse_responses(self, n: int | None = None, eps: float = 1e-4) -> np.ndarray:
        """Band-limited impulse responses (S, M, n) by regularised deconvolution.

        :math:`h = \\mathcal{F}^{-1}[G\\, Y W^* / (|W|^2 + \\epsilon \\max|W|^2)]`,
        with :math:`G` a raised-cosine band mask that is 1 inside
        ``self.band`` and falls to 0 over the half octave outside it. The
        mask matters: a 2D room's response grows without bound towards DC,
        so an unmasked inverse amplifies the pulse's tiny near-DC content
        into a slowly decaying tail that wraps around the FFT.
        """
        T = self.steps
        n = T if n is None else int(n)
        Y = np.fft.rfft(self.rec.astype(np.float64), n=T, axis=-1)
        W = np.fft.rfft(self.pulse, n=T)
        f = np.fft.rfftfreq(T, self.dt)
        lo, hi = self.band
        G = np.ones_like(f)
        low = f < lo
        G[low] = 0.5 - 0.5 * np.cos(np.pi * np.clip((f[low] - lo / 2) / (lo / 2), 0.0, 1.0))
        high = f > hi
        G[high] = 0.5 + 0.5 * np.cos(np.pi * np.clip((f[high] - hi) / (hi / 2), 0.0, 1.0))
        inv = np.conj(W) / (np.abs(W) ** 2 + eps * np.abs(W).max() ** 2)
        h = np.fft.irfft(Y * (G * inv), n=T, axis=-1)
        return h[..., :n]

    def select(self, idx: "np.ndarray | Sequence[int] | slice") -> "TransferSet":
        """The same measurement restricted to a subset of points."""
        return TransferSet(
            self.dt,
            self.speakers,
            self.points[idx],
            self.rec[:, idx],
            self.pulse,
            self.band,
            self.room,
        )

    def valid_band(self, floor_db: float = -40.0) -> tuple[float, float]:
        """Frequency range where the excitation is within ``floor_db`` of its peak."""
        W = np.abs(np.fft.rfft(self.pulse, n=self.steps))
        f = np.fft.rfftfreq(self.steps, self.dt)
        ok = np.where(W >= W.max() * 10 ** (floor_db / 20))[0]
        return float(f[ok[0]]), float(f[ok[-1]])

    def tail_db(self, fraction: float = 0.1) -> float:
        """Energy in the last ``fraction`` of the records relative to the total (dB).

        A truncation diagnostic: well below -40 dB means the records caught
        essentially the whole room response.
        """
        k = max(1, int(self.steps * fraction))
        e = np.square(self.rec.astype(np.float64))
        return float(10 * np.log10(e[..., -k:].sum() / max(e.sum(), 1e-300) + 1e-300))


def measure_transfer(
    room: Room,
    speakers: "np.ndarray | Sequence[Sequence[float]]",
    points: "np.ndarray | Sequence[Sequence[float]]",
    *,
    duration: float = 0.15,
    band: tuple[float, float] = (100.0, 2000.0),
    pulse_taps: int = 511,
    engine: str = "numba",
) -> TransferSet:
    """Measure :math:`H[f, m, s]` by driving each speaker in turn.

    Parameters
    ----------
    room
        The scene.
    speakers, points
        World coordinates (S, 2) and (M, 2).
    duration
        Record length in seconds, rounded up to a multiple of 512 steps
        (so FIR design grids of up to 512 taps fall on the FFT bins).
    band
        Excitation band [Hz]; keep the top below about
        :math:`c / (6\\Delta x)` (2.3 kHz at 2.5 cm) where the scheme's
        dispersion is small.
    engine
        ``"numba"`` (reference, any boundary) or ``"torch"`` (one batched
        ``TorchFDTD`` launch for all speakers; no CPML/Mur).
    """
    speakers = np.atleast_2d(np.asarray(speakers, dtype=np.float64))
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    dt = room.timestep
    steps = int(np.ceil(duration / dt / 512.0)) * 512
    w = bandpass_pulse(dt, band, pulse_taps)
    pulse = np.zeros(steps)
    pulse[: len(w)] = w
    S, M = len(speakers), len(points)
    if engine == "numba":
        rec = np.empty((S, M, steps), dtype=np.float32)
        for s in range(S):
            rec[s] = simulate_drives(room, speakers[s : s + 1], w[None, :], points, steps).rec
    elif engine == "torch":
        rec = _record_torch(room, speakers, points, pulse, steps)
    else:
        raise ValueError(f"engine must be 'numba' or 'torch', got {engine!r}")
    return TransferSet(dt, speakers, points, rec, pulse, band, room)


def _record_torch(
    room: Room, speakers: np.ndarray, points: np.ndarray, pulse: np.ndarray, steps: int
) -> np.ndarray:
    import torch

    from ..simulation.torch_engine import TorchFDTD, TorchGrid

    S = len(speakers)
    grid = TorchGrid(room.shape, wavespeed=room.c, gridstep=room.dx, courant=room.courant)
    src = torch.tensor(room.cells(speakers))[:, None, :]  # (S, 1, 2): one speaker per batch
    mic = torch.tensor(room.cells(points))[None].expand(S, -1, -1).contiguous()
    material = np.broadcast_to(room.material_map(), (S,) + room.shape).copy()
    eng = TorchFDTD(
        grid,
        src,
        mic,
        material=material,
        boundary=room.boundary,
        boundary_beta=room.beta,
    )
    drive = torch.tensor(np.broadcast_to(pulse, (S, 1, steps)).copy(), dtype=torch.float32)
    with torch.no_grad():
        rec = eng.run(drive)
    return rec.numpy().astype(np.float32)


__all__ = [
    "Box",
    "DriveResult",
    "Room",
    "TransferSet",
    "bandpass_pulse",
    "disk_points",
    "dtft",
    "grid_points",
    "measure_transfer",
    "simulate_drives",
]
