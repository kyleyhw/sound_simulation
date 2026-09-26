"""Information limits of active room sensing (study 2026-09-26).

How much of a room map can a sensing configuration deliver *at all*,
whatever the estimator? This module linearises the engine's forward map
around the empty room, whitens it by the measurement noise, and counts the
modes the data resolve better than a weak prior.

Forward model (discrete Born sensitivity)
-----------------------------------------
The engine advances :math:`p^{n+1} = 2p^n - p^{n-1} + C\\,\\mathcal{L}p^n
+ q^n` (``simulate.py``, general path), with :math:`C = (c\\Delta t /
\\Delta x)^2` per cell, :math:`\\mathcal{L}` the 5-point Laplacian over open
faces, and :math:`q^n` the source samples added to :math:`p^{n+1}`. The
recorded trace sample :math:`n` is :math:`p^{n+1}` at a mic. Let
:math:`G_{\\mathbf a}[k](\\mathbf x)` be the field at :math:`\\mathbf x`
recorded :math:`k` steps after a unit (Kronecker) injection at
:math:`\\mathbf a` (one simulation per position), and :math:`P[k] =
p^{k+1}` the background field of a shot. Perturbing the scheme at a cell
adds an injection there, which then propagates with :math:`G`. The
scheme is reciprocal (:math:`G_{\\mathbf x \\to \\mathbf m} =
G_{\\mathbf m \\to \\mathbf x}`, ``pose_refine.py``), so one simulation from
each mic gives the receiver side for every cell:

* **monopole** (compressibility, :math:`\\delta C / C` at cell
  :math:`\\mathbf x`): injection :math:`\\delta C\\,\\mathcal{L}p^j`, so

  .. math:: \\delta y_m[n] = \\frac{\\delta C}{C}\\sum_j
      G_{\\mathbf m}[n-j](\\mathbf x)\\,(C\\mathcal{L}P)[j-1](\\mathbf x);

* **dipole** (face conductance, the density term): removing the face
  between :math:`\\mathbf a` and :math:`\\mathbf b` to first order gives

  .. math:: \\delta y_m[n] = \\sum_j
      C\\bigl(G_{\\mathbf m}(\\mathbf a) - G_{\\mathbf m}(\\mathbf b)\\bigr)[n-j]
      \\bigl(P(\\mathbf a) - P(\\mathbf b)\\bigr)[j-1].

A coarse pixel sums the monopole kernels of its cells and the dipole
kernels of its boundary faces (a rigid pixel closes those faces). The
kernel of an *occupied rigid pixel* is fitted as :math:`a K_M + b K_D`
against exact engine runs with that pixel made rigid
(:func:`fit_rigid_kernel`). The Jacobian :math:`J` then maps pixel
occupancy (0 = air, 1 = rigid) to the recorded traces of every shot and
mic, sampled as the loop records them.

Any emission scheme fits this frame: a shot's background :math:`P` is
simulated with all its drives at once, so a simultaneous, summed,
band-split, coded or steered emission is exactly the Born map of that
shot.

Information measures
--------------------
With white noise :math:`\\sigma` per sample and a Gaussian prior
:math:`\\theta \\sim \\mathcal N(\\bar\\theta, \\tau^2 I)`, the Fisher matrix is
:math:`F = J^\\top J / \\sigma^2`. With eigenvalues :math:`\\lambda_i` of
:math:`J^\\top J` and :math:`x_i = \\lambda_i \\tau^2 / \\sigma^2`:

* recoverable degrees of freedom: :math:`\\#\\{x_i \\ge 1\\}`, the modes the
  data measure better than the prior;
* degrees of freedom for signal (Rodgers):
  :math:`\\mathrm{tr}\\,R = \\sum x_i / (1 + x_i)`;
* information content :math:`\\tfrac12 \\sum \\log_2 (1 + x_i)` bits;
* per-pixel posterior (Cramér-Rao with prior) standard deviation
  :math:`\\sqrt{\\mathrm{diag}\\,(F + I/\\tau^2)^{-1}}`;
* resolution matrix :math:`R = (F + I/\\tau^2)^{-1} F`, whose column
  :math:`j` is the point-spread function of pixel :math:`j`.

Emission algebra
----------------
Per frequency, a scheme with shot drive vectors :math:`w_k(f) \\in
\\mathbb C^S` gives :math:`F = \\sum_f \\sum_m K_m^H M(f) K_m` with the
emission Gram :math:`M(f) = \\sum_k \\bar w_k w_k^\\top`
(:func:`emission_gram`). Speakers in turn have :math:`M = |r|^2 I`. So a
summed or steered emission (:math:`M \\preceq \\lambda_{\\max}(M) I`) can
never resolve a mode the speakers in turn miss, and codes whose spectra do
not overlap add exactly the per-speaker information
(:func:`drive_rows`, tested in ``tests/imaging/test_information.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from ..simulation.simulate import Simulate

DT = 0.5  # engine timestep at Courant 0.5, c = dx = 1
F0 = 0.08  # loop Ricker peak frequency, cycles per unit time (0.04 per step)
LISTEN = 622  # closedLoop.senseSteps for the 100 x 100 room
DEVICE_ROW = 86
GRID_N = 100
WALL = 12  # CPML thickness of the loop room; reflective rooms use a wall slab this thick

Pos = tuple[int, int]

ROOM_MATERIAL = {"rigid": 2, "plaster": 3, "wood": 4}


# --------------------------------------------------------------------- rooms


@dataclass(frozen=True)
class Room:
    """The loop's 100 x 100 room: CPML, or walls of a material at the CPML interface.

    ``kind`` is ``"cpml"`` (anechoic, as the loop), ``"rigid"``
    (reflection +1), ``"plaster"`` (beta 0.1, R = 0.82 at normal incidence)
    or ``"wood"`` (beta 0.3, R = 0.54). The reflective rooms fill the
    outer ``wall`` cells with the material, so the air interior is the
    same as the CPML room's (cells ``wall`` to ``n - wall - 1``).
    """

    kind: str = "cpml"
    n: int = GRID_N
    wall: int = WALL

    def materials(self) -> NDArray[np.uint8]:
        m = np.zeros((self.n, self.n), dtype=np.uint8)
        if self.kind != "cpml":
            w = self.wall
            mid = ROOM_MATERIAL[self.kind]
            m[:w] = mid
            m[-w:] = mid
            m[:, :w] = mid
            m[:, -w:] = mid
        return m

    def simulate(
        self, extra_rigid: NDArray[np.bool_] | None = None, speed: NDArray | None = None
    ) -> Simulate:
        if self.kind == "cpml":
            sim = Simulate((self.n, self.n), boundary="cpml", cpml_cells=self.wall)
        else:
            sim = Simulate((self.n, self.n))
        mat = self.materials()
        if extra_rigid is not None:
            mat[extra_rigid] = 2
        if mat.any():
            sim.set_material_map(mat)
        if speed is not None:
            sim.set_speed_map(speed)
        return sim

    @property
    def coeff(self) -> float:
        return float((DT) ** 2)  # (c dt / dx)^2 at Courant 0.5


# -------------------------------------------------------------------- pixels


@dataclass(frozen=True)
class PixelGrid:
    """Square pixels of ``size`` cells over rows ``r0:r1`` and cols ``c0:c1``.

    The default is the loop room's air interior (rows 12-75, cols 12-87)
    in 2 x 2 pixels (32 x 38 = 1216 pixels). The loop places objects over
    rows 8-74 and cols 8-91; the outer 4-cell rim lies inside the CPML,
    where the scheme is stretched and objects are half absorbed, so it is
    left out.
    """

    r0: int = WALL
    r1: int = 76  # the loop keeps objects at rows <= 74, 12 cells clear of the device row
    c0: int = WALL
    c1: int = GRID_N - WALL
    size: int = 2

    def __post_init__(self) -> None:
        if (self.r1 - self.r0) % self.size or (self.c1 - self.c0) % self.size:
            raise ValueError("region must be a whole number of pixels")

    @property
    def shape(self) -> tuple[int, int]:
        return ((self.r1 - self.r0) // self.size, (self.c1 - self.c0) // self.size)

    @property
    def n_pix(self) -> int:
        a, b = self.shape
        return a * b

    @property
    def window(self) -> tuple[slice, slice]:
        """Region plus a one-cell margin (for the Laplacian and the boundary faces)."""
        return slice(self.r0 - 1, self.r1 + 1), slice(self.c0 - 1, self.c1 + 1)

    def centres(self) -> NDArray[np.float64]:
        """Pixel centres in cell coordinates, ``(n_pix, 2)`` (row, col)."""
        a, b = self.shape
        s = self.size
        rr = self.r0 + s * np.arange(a) + (s - 1) / 2
        cc = self.c0 + s * np.arange(b) + (s - 1) / 2
        R, Cc = np.meshgrid(rr, cc, indexing="ij")
        return np.stack([R.ravel(), Cc.ravel()], 1)

    def cells(self, j: int) -> tuple[slice, slice]:
        a, b = self.shape
        pr, pc = divmod(int(j), b)
        s = self.size
        return (
            slice(self.r0 + s * pr, self.r0 + s * pr + s),
            slice(self.c0 + s * pc, self.c0 + s * pc + s),
        )

    def mask(self, j: int, n: int = GRID_N) -> NDArray[np.bool_]:
        m = np.zeros((n, n), dtype=bool)
        m[self.cells(j)] = True
        return m

    def index(self, row: float, col: float) -> int:
        a, b = self.shape
        pr = int(np.clip((row - self.r0) // self.size, 0, a - 1))
        pc = int(np.clip((col - self.c0) // self.size, 0, b - 1))
        return pr * b + pc


# ----------------------------------------------------------------- emissions


def ricker_samples(length: int, f0: float = F0, delay: float | None = None) -> NDArray:
    """The loop's Ricker as injected per step: ``w[k] = ricker(k dt)`` (amplitude 1)."""
    d = 1.5 / f0 if delay is None else delay
    a = np.pi * f0 * (np.arange(length) * DT - d)
    return (1.0 - 2.0 * a * a) * np.exp(-a * a)


def rc_step(f: NDArray, a: float, b: float) -> NDArray:
    """Raised-cosine step: 0 below ``a``, 1 above ``b``."""
    x = np.clip((f - a) / (b - a), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


BAND_LOW_EDGE = (0.065, 0.075)
BAND_HIGH_EDGE = (0.085, 0.095)
BAND_DELAY = 120
BAND_LEN = 240


def band_pulses() -> tuple[NDArray, NDArray]:
    """Low / high halves of a Ricker (``web/src/twospeaker/device.ts`` ``bandPulses``)."""
    n = 1024
    r = ricker_samples(n, F0, BAND_DELAY * DT)
    R = np.fft.fft(r)
    k = np.arange(n)
    f = np.minimum(k, n - k) / (n * DT)
    lo = np.real(np.fft.ifft(R * (1.0 - rc_step(f, *BAND_LOW_EDGE))))
    hi = np.real(np.fft.ifft(R * rc_step(f, *BAND_HIGH_EDGE)))
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(BAND_LEN) / (BAND_LEN - 1))
    return lo[:BAND_LEN] * w, hi[:BAND_LEN] * w


CODE_BAND = (0.01, 0.22)


def chirp_code(length: int, down: bool = False) -> NDArray:
    """Linear chirp over ``CODE_BAND`` with 10 % raised-cosine tapers (``device.ts`` ``chirpCode``)."""
    fa, fb = (CODE_BAND[1], CODE_BAND[0]) if down else CODE_BAND
    T = (length - 1) * DT
    kk = (fb - fa) / T
    edge = 0.1 * T
    t = np.arange(length) * DT
    env = np.ones(length)
    lo = t < edge
    hi = t > T - edge
    env[lo] = 0.5 - 0.5 * np.cos(np.pi * t[lo] / edge)
    env[hi] = 0.5 - 0.5 * np.cos(np.pi * (T - t[hi]) / edge)
    return env * np.sin(2 * np.pi * (fa + 0.5 * kk * t) * t)


BEAM_DELAY = 12
BEAM_SHOTS = (
    ((1.0, 0), (1.0, 0)),
    ((1.0, 0), (-1.0, 0)),
    ((1.0, 0), (1.0, BEAM_DELAY)),
    ((1.0, BEAM_DELAY), (1.0, 0)),
)


def delayed(x: NDArray, d: int, g: float, length: int) -> NDArray:
    out = np.zeros(length)
    m = min(len(x), length - d)
    if m > 0:
        out[d : d + m] = g * x[:m]
    return out


def energy(x: NDArray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


RICKER_ENERGY = energy(ricker_samples(LISTEN))


def unit_energy(x: NDArray, e: float = RICKER_ENERGY) -> NDArray:
    """Scale a drive to energy ``e`` (default: the loop's amplitude-1 Ricker)."""
    return np.asarray(x, dtype=np.float64) * np.sqrt(e / energy(x))


# -------------------------------------------------------------------- shots


@dataclass(eq=False)
class Shot:
    """One emission: drives ``(position, samples)`` played together; ``mics`` record ``steps`` samples."""

    drives: list[tuple[Pos, NDArray]]
    mics: list[Pos]
    steps: int


@dataclass(eq=False)
class Config:
    name: str
    shots: list[Shot]
    room: Room = field(default_factory=Room)
    label: str = ""

    @property
    def n_traces(self) -> int:
        return sum(len(s.mics) for s in self.shots)

    @property
    def time_steps(self) -> int:
        return sum(s.steps for s in self.shots)

    def positions(self) -> tuple[list[Pos], list[Pos]]:
        sp: list[Pos] = []
        mi: list[Pos] = []
        for s in self.shots:
            for p, _ in s.drives:
                if p not in sp:
                    sp.append(p)
            for p in s.mics:
                if p not in mi:
                    mi.append(p)
        return sp, mi


def laptop(
    centre: int, spacing: int = 12, inset: int = 2, row: int = DEVICE_ROW
) -> tuple[list[Pos], list[Pos]]:
    """Two speakers ``spacing`` apart and two mics ``inset`` cells inboard (``device.ts`` ``laptop``)."""
    a = centre - spacing // 2
    b = a + spacing
    return [(row, a), (row, b)], [(row, a + inset), (row, b - inset)]


BAR = [(DEVICE_ROW, 31 + 4 * k) for k in range(8)]
PLACEMENTS = {1: [45], 2: [38, 52], 4: [24, 38, 52, 66], 8: [18 + 8 * k for k in range(8)]}


def seq_shots(
    speakers: Sequence[Pos], mics: Sequence[Pos], drive: NDArray, steps: int = LISTEN
) -> list[Shot]:
    return [Shot([(s, drive)], list(mics), steps) for s in speakers]


def scheme_shots(
    kind: str, speakers: Sequence[Pos], mics: Sequence[Pos], normalise: bool = True
) -> list[Shot]:
    """Shots of one placement for the two-speaker emission schemes of ``device.ts``.

    ``seq`` in turn, ``sum`` same pulse at once, ``band`` disjoint half
    bands at once, ``code``/``code128`` up/down chirps at once (622 or 128
    steps), ``chirp_seq``/``chirp128_seq`` the same chirps in turn (the
    perfectly separated reference), ``beams`` four steered same-pulse
    shots. With ``normalise`` every speaker emission has the Ricker's
    energy (equal energy per speaker per shot).
    """
    A, B = speakers
    ric = ricker_samples(LISTEN)
    nz = unit_energy if normalise else (lambda x: np.asarray(x, dtype=np.float64))
    if kind == "seq":
        return seq_shots(speakers, mics, ric)
    if kind == "sum":
        return [Shot([(A, ric), (B, ric)], list(mics), LISTEN)]
    if kind == "band":
        lo, hi = band_pulses()
        steps = LISTEN + int(np.ceil(BAND_DELAY - 1.5 / F0 / DT))
        return [Shot([(A, nz(lo)), (B, nz(hi))], list(mics), steps)]
    if kind in ("code", "code128", "chirp_seq", "chirp128_seq"):
        L = 128 if "128" in kind else LISTEN
        up = nz(chirp_code(L))
        dn = nz(chirp_code(L, down=True))
        if kind.startswith("code"):
            return [Shot([(A, up), (B, dn)], list(mics), L + LISTEN)]
        return [Shot([(A, up)], list(mics), L + LISTEN), Shot([(B, dn)], list(mics), L + LISTEN)]
    if kind == "beams":
        n = LISTEN + BEAM_DELAY
        r = ricker_samples(n)
        return [
            Shot([(A, delayed(r, da, ga, n)), (B, delayed(r, db, gb, n))], list(mics), n)
            for (ga, da), (gb, db) in BEAM_SHOTS
        ]
    raise ValueError(f"unknown scheme {kind!r}")


def standard_configs() -> dict[str, Config]:
    """Every configuration of the study (see ``docs/imaging.md`` section 11)."""
    ric = ricker_samples(LISTEN)
    cfg: dict[str, Config] = {}

    def add(name: str, shots: list[Shot], room: str = "cpml", label: str = "") -> None:
        cfg[name] = Config(name, shots, Room(room), label or name)

    add("bar8", seq_shots(BAR, BAR, ric), label="8-element bar, in turn (64 traces)")
    sp, mi = laptop(45)
    add("seq", scheme_shots("seq", sp, mi), label="2 spk + 2 mic, 12 apart, in turn")
    wsp, wmi = laptop(45, 28)
    add("wide_seq", scheme_shots("seq", wsp, wmi), label="2 spk + 2 mic, 28 apart, in turn")
    for kind, lab in (
        ("sum", "same pulse at once"),
        ("band", "disjoint half bands at once"),
        ("code", "up/down chirps at once (622)"),
        ("code128", "up/down chirps at once (128)"),
        ("chirp_seq", "the same chirps in turn (622)"),
        ("chirp128_seq", "the same chirps in turn (128)"),
        ("beams", "4 steered same-pulse shots"),
    ):
        add(kind, scheme_shots(kind, sp, mi), label=f"narrow, {lab}")
    add(
        "band_raw",
        scheme_shots("band", sp, mi, normalise=False),
        label="narrow, half bands, drives as trained",
    )
    for k in (2, 4, 8):
        shots: list[Shot] = []
        for c in PLACEMENTS[k]:
            s2, m2 = laptop(c)
            shots += seq_shots(s2, m2, ric)
        add(f"seq_k{k}", shots, label=f"narrow device at K = {k} placements")
    for f0 in (0.04, 0.12, 0.16):
        d = unit_energy(ricker_samples(LISTEN, f0))
        add(
            f"seq_f{int(round(f0 * 100)):02d}",
            seq_shots(sp, mi, d),
            label=f"narrow, Ricker f0 = {f0}",
        )
    for room in ("rigid", "plaster", "wood"):
        add(f"seq_{room}", scheme_shots("seq", sp, mi), room, f"narrow, {room} walls")
    add("wide_rigid", scheme_shots("seq", wsp, wmi), "rigid", "wide, rigid walls")
    add("bar8_rigid", seq_shots(BAR, BAR, ric), "rigid", "bar, rigid walls")
    return cfg


# ---------------------------------------------------------------- simulation


def run_shot(
    room: Room,
    drives: Sequence[tuple[Pos, NDArray]],
    steps: int,
    window: tuple[slice, slice] | None = None,
    mics: Sequence[Pos] = (),
    extra_rigid: NDArray[np.bool_] | None = None,
    speed: NDArray | None = None,
) -> tuple[NDArray[np.float32] | None, NDArray[np.float64]]:
    """Simulate one shot; return the windowed field per step and the mic traces.

    ``drives[i][1][n]`` is added to :math:`p^{n+1}` at the drive cell (the
    engine's soft-source convention), and sample ``n`` of every output is
    :math:`p^{n+1}`.
    """
    sim = room.simulate(extra_rigid, speed)
    fields = None
    if window is not None:
        h = window[0].stop - window[0].start
        w = window[1].stop - window[1].start
        fields = np.empty((steps, h, w), dtype=np.float32)
    traces = np.empty((len(mics), steps))
    mr = np.array([m[0] for m in mics], dtype=np.int64)
    mc = np.array([m[1] for m in mics], dtype=np.int64)
    dr = [(p, np.asarray(d, dtype=np.float64)) for p, d in drives]
    for n in range(steps):
        sim.step()
        for p, d in dr:
            if n < d.shape[0]:
                sim.p[p] += np.float32(d[n])
        if fields is not None and window is not None:
            fields[n] = sim.p[window]
        if len(mics):
            traces[:, n] = sim.p[mr, mc]
    return fields, traces


def impulse(pos: Pos) -> list[tuple[Pos, NDArray]]:
    return [(pos, np.ones(1))]


# ------------------------------------------------------------ Born kernels


def _open_faces(room: Room, grid: PixelGrid) -> tuple[NDArray, NDArray]:
    """Open-face masks in the window: vertical (between rows) and horizontal faces."""
    mat = room.materials()[grid.window]
    air = mat == 0
    return (air[1:] & air[:-1]).astype(np.float32), (air[:, 1:] & air[:, :-1]).astype(np.float32)


@dataclass
class _Spectra:
    """Time-FFTs of a windowed field: region cells, and pixel-boundary face differences."""

    cells: NDArray[np.complex64]  # (bins, nr*s, nc*s)
    fv: NDArray[np.complex64]  # (bins, nr+1, nc*s) faces between rows on pixel boundaries
    fh: NDArray[np.complex64]  # (bins, nr*s, nc+1) faces between cols on pixel boundaries


def _spectra(
    F: NDArray[np.float32],
    grid: PixelGrid,
    nfft: int,
    faces: tuple[NDArray, NDArray],
    laplacian: bool,
) -> _Spectra:
    s = grid.size
    ov, oh = faces
    dv = (F[:, 1:, :] - F[:, :-1, :]) * ov
    dh = (F[:, :, 1:] - F[:, :, :-1]) * oh
    if laplacian:
        cells = dv[:, 1:, 1:-1] - dv[:, :-1, 1:-1] + dh[:, 1:-1, 1:] - dh[:, 1:-1, :-1]
    else:
        cells = F[:, 1:-1, 1:-1]
    fv = dv[:, ::s, 1:-1]  # rows 0, s, 2s, ... of the window faces are the pixel boundaries
    fh = dh[:, 1:-1, ::s]

    def ft(x: NDArray) -> NDArray[np.complex64]:
        return np.fft.rfft(x, n=nfft, axis=0).astype(np.complex64)

    return _Spectra(ft(cells), ft(fv), ft(fh))


def _pixel_sum(x: NDArray, nr: int, nc: int, s: int) -> NDArray:
    b = x.shape[0]
    return x.reshape(b, nr, s, nc, s).sum(axis=(2, 4)).reshape(b, nr * nc)


def born_pair(
    g: _Spectra, p: _Spectra, grid: PixelGrid, coeff: float, nfft: int, steps: int
) -> tuple[NDArray, NDArray]:
    """Monopole and dipole Born rows ``(n_pix, steps)`` of one (shot, mic) trace.

    ``g`` holds the mic's Green's function spectra, ``p`` the shot's
    background (cells hold :math:`\\mathcal L P`).
    """
    nr, nc = grid.shape
    s = grid.size
    bins = g.cells.shape[0]
    shift = np.exp(-2j * np.pi * np.arange(bins) / nfft).astype(np.complex64)[:, None]
    mono = _pixel_sum(g.cells * p.cells, nr, nc, s) * (coeff * shift)
    v = (g.fv * p.fv).reshape(bins, nr + 1, nc, s).sum(3)  # (bins, nr+1, nc)
    h = (g.fh * p.fh).reshape(bins, nr, s, nc + 1).sum(2)  # (bins, nr, nc+1)
    dip = ((v[:, :-1] + v[:, 1:]) + (h[:, :, :-1] + h[:, :, 1:])).reshape(bins, nr * nc) * (
        coeff * shift
    )
    km = np.fft.irfft(mono, n=nfft, axis=0)[:steps].T
    kd = np.fft.irfft(dip, n=nfft, axis=0)[:steps].T
    return km, kd


def born_rows(
    config: Config, grid: PixelGrid, log: Callable[[str], None] | None = None
) -> Iterator[tuple[int, Pos, NDArray, NDArray]]:
    """Yield ``(shot index, mic, K_M, K_D)`` for every trace of ``config``."""
    room = config.room
    faces = _open_faces(room, grid)
    tmax = max(s.steps for s in config.shots)
    nfft = 2 * tmax
    _, mics = config.positions()
    gspec: dict[Pos, _Spectra] = {}
    for m in mics:
        F, _ = run_shot(room, impulse(m), tmax, grid.window)
        assert F is not None
        gspec[m] = _spectra(F, grid, nfft, faces, laplacian=False)
    for k, shot in enumerate(config.shots):
        F, _ = run_shot(room, shot.drives, shot.steps, grid.window)
        assert F is not None
        ps = _spectra(F, grid, nfft, faces, laplacian=True)
        for m in shot.mics:
            km, kd = born_pair(gspec[m], ps, grid, room.coeff, nfft, shot.steps)
            yield k, m, km, kd
        if log is not None:
            log(f"  {config.name}: shot {k + 1}/{len(config.shots)}")


@dataclass
class Grams:
    """Gram blocks of the monopole / dipole rows of one configuration."""

    mm: NDArray[np.float64]
    md: NDArray[np.float64]
    dd: NDArray[np.float64]

    def combine(self, a: float, b: float) -> NDArray[np.float64]:
        """:math:`J^\\top J` for :math:`J = a K_M + b K_D`."""
        return a * a * self.mm + a * b * (self.md + self.md.T) + b * b * self.dd


def config_grams(
    config: Config, grid: PixelGrid, log: Callable[[str], None] | None = None
) -> Grams:
    n = grid.n_pix
    mm = np.zeros((n, n))
    md = np.zeros((n, n))
    dd = np.zeros((n, n))
    for _, _, km, kd in born_rows(config, grid, log):
        mm += km @ km.T
        md += km @ kd.T
        dd += kd @ kd.T
    return Grams(mm, md, dd)


def config_rows(
    config: Config, grid: PixelGrid, pixels: Sequence[int] | NDArray
) -> tuple[NDArray, NDArray]:
    """Born rows restricted to ``pixels``: ``K_M, K_D`` of shape ``(len(pixels), total samples)``."""
    idx = np.asarray(pixels)
    ms, ds = [], []
    for _, _, km, kd in born_rows(config, grid):
        ms.append(km[idx])
        ds.append(kd[idx])
    return np.concatenate(ms, 1), np.concatenate(ds, 1)


# ---------------------------------------------------------- exact responses


def empty_traces(config: Config) -> list[NDArray]:
    return [run_shot(config.room, s.drives, s.steps, None, s.mics)[1] for s in config.shots]


def scattered_traces(
    config: Config, rigid: NDArray[np.bool_], empty: list[NDArray] | None = None
) -> NDArray:
    """Exact residual (room with ``rigid`` cells minus empty room), all traces concatenated."""
    e = empty if empty is not None else empty_traces(config)
    out = []
    for s, y0 in zip(config.shots, e):
        _, y = run_shot(config.room, s.drives, s.steps, None, s.mics, extra_rigid=rigid)
        out.append((y - y0).ravel())
    return np.concatenate(out)


def fit_rigid_kernel(config: Config, grid: PixelGrid, pixels: Sequence[int] | NDArray) -> dict:
    """Fit an occupied rigid pixel's response as ``a K_M + b K_D`` (least squares over ``pixels``).

    Returns the coefficients, the fraction of the exact residual energy
    explained (overall and per pixel), and the same for the monopole alone.
    """
    km, kd = config_rows(config, grid, pixels)
    e = empty_traces(config)
    ys = np.stack([scattered_traces(config, grid.mask(j), e) for j in pixels])
    A = np.stack([km.ravel(), kd.ravel()], 1)
    coef, *_ = np.linalg.lstsq(A, ys.ravel(), rcond=None)
    a, b = float(coef[0]), float(coef[1])
    res = ys - (a * km + b * kd)
    am = float(np.sum(km * ys) / np.sum(km * km))
    res_m = ys - am * km
    tot = np.sum(ys**2, 1)
    return {
        "a": a,
        "b": b,
        "explained": float(1 - np.sum(res**2) / np.sum(tot)),
        "explained_per_pixel": (1 - np.sum(res**2, 1) / tot).tolist(),
        "a_mono_only": am,
        "explained_mono_only": float(1 - np.sum(res_m**2) / np.sum(tot)),
        "pixels": [int(j) for j in pixels],
    }


def reference_echo(
    config: Config, rows: tuple[int, int] = (37, 45), cols: tuple[int, int] = (45, 56)
) -> float:
    """Peak |residual| of a reference rigid block (default 8 x 11 cells mid-room) over all traces."""
    m = np.zeros((config.room.n, config.room.n), dtype=bool)
    m[rows[0] : rows[1], cols[0] : cols[1]] = True
    return float(np.abs(scattered_traces(config, m)).max())


# ------------------------------------------------------ information measures


def eig_gram(gram: NDArray) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues (descending, clipped at 0) and eigenvectors of a Gram matrix."""
    lam, V = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(lam)[::-1]
    return np.clip(lam[order], 0.0, None), V[:, order]


def noise_sigma(snr_db: float, echo: float) -> float:
    """Per-sample noise standard deviation at ``snr_db`` below the reference echo peak."""
    return float(echo * 10.0 ** (-snr_db / 20.0))


def snr_modes(lam: NDArray, sigma: float, tau: float) -> NDArray:
    """:math:`x_i = \\lambda_i \\tau^2 / \\sigma^2` (mode SNR relative to the prior)."""
    return np.asarray(lam) * (tau * tau) / (sigma * sigma)


def dof(lam: NDArray, sigma: float, tau: float) -> int:
    """Number of modes measured better than the prior (:math:`x_i \\ge 1`)."""
    return int(np.sum(snr_modes(lam, sigma, tau) >= 1.0))


def dfs(lam: NDArray, sigma: float, tau: float) -> float:
    """Degrees of freedom for signal, :math:`\\mathrm{tr} R = \\sum x/(1+x)`."""
    x = snr_modes(lam, sigma, tau)
    return float(np.sum(x / (1.0 + x)))


def info_bits(lam: NDArray, sigma: float, tau: float) -> float:
    return float(0.5 * np.sum(np.log2(1.0 + snr_modes(lam, sigma, tau))))


def posterior_std(lam: NDArray, V: NDArray, sigma: float, tau: float) -> NDArray:
    """Per-pixel sqrt diag of :math:`(F + I/\\tau^2)^{-1}` (CRB with a Gaussian prior)."""
    w = 1.0 / (np.asarray(lam) / sigma**2 + 1.0 / tau**2)
    return np.sqrt((V * V) @ w)


def resolution_matrix(lam: NDArray, V: NDArray, sigma: float, tau: float) -> NDArray:
    """:math:`R = (F + I/\\tau^2)^{-1} F`; column ``j`` is the point-spread function of pixel ``j``."""
    x = snr_modes(lam, sigma, tau)
    return (V * (x / (1.0 + x))) @ V.T


def _half_width(line: NDArray, i: int) -> float:
    """Full width (samples) at half of ``line[i]`` around ``i``, linearly interpolated."""
    v = line[i]
    if not v > 0:
        return float("nan")
    h = 0.5 * v
    n = len(line)
    lo = float(i)
    for k in range(i - 1, -1, -1):
        if line[k] < h:
            lo = k + (h - line[k]) / (line[k + 1] - line[k])
            break
        lo = float(k) - 0.5
    hi = float(i)
    for k in range(i + 1, n):
        if line[k] < h:
            hi = k - (h - line[k]) / (line[k - 1] - line[k])
            break
        hi = float(k) + 0.5
    return max(hi - lo, 1.0)


def psf_widths(R: NDArray, grid: PixelGrid) -> tuple[NDArray, NDArray]:
    """FWHM of each pixel's PSF along rows (range from the wall) and cols (cross-range), in cells."""
    nr, nc = grid.shape
    wr = np.full(grid.n_pix, np.nan)
    wc = np.full(grid.n_pix, np.nan)
    for j in range(grid.n_pix):
        img = R[:, j].reshape(nr, nc)
        pr, pc = divmod(j, nc)
        wr[j] = _half_width(img[:, pc], pr) * grid.size
        wc[j] = _half_width(img[pr, :], pc) * grid.size
    return wr, wc


def psf_widths_polar(
    R: NDArray,
    grid: PixelGrid,
    centre: tuple[float, float] = (DEVICE_ROW, 45.0),
    reach: float = 40.0,
) -> tuple[NDArray, NDArray]:
    """FWHM of each PSF along the radial (range) and tangential (cross-range) directions from ``centre``, in cells.

    The PSF image is sampled every half cell by bilinear interpolation of
    the pixel values, so a single-pixel PSF has a width of one pixel
    (``grid.size`` cells), the floor of this measure.
    """
    from scipy.ndimage import map_coordinates

    nr, nc = grid.shape
    cen = grid.centres()
    t = np.arange(-reach, reach + 0.25, 0.5)
    i0 = int(np.argmin(np.abs(t)))
    off = grid.r0 + (grid.size - 1) / 2, grid.c0 + (grid.size - 1) / 2
    wr = np.full(grid.n_pix, np.nan)
    wt = np.full(grid.n_pix, np.nan)
    for j in range(grid.n_pix):
        img = R[:, j].reshape(nr, nc)
        y, x = cen[j]
        u = np.array([y - centre[0], x - centre[1]])
        u /= max(float(np.hypot(*u)), 1e-9)
        for out, d in ((wr, u), (wt, np.array([-u[1], u[0]]))):
            rows = (y + t * d[0] - off[0]) / grid.size
            cols = (x + t * d[1] - off[1]) / grid.size
            line = map_coordinates(img, [rows, cols], order=1, mode="constant", cval=0.0)
            out[j] = _half_width(line, i0) * 0.5
    return wr, wt


def psf_spread(
    R: NDArray, grid: PixelGrid, centre: tuple[float, float] = (DEVICE_ROW, 45.0)
) -> tuple[NDArray, NDArray]:
    """Energy-weighted RMS extent of each PSF along the radial and tangential directions (cells).

    :math:`\\sqrt{\\sum_i R_{ij}^2 ((\\mathbf x_i - \\mathbf x_j)\\cdot\\mathbf u)^2 /
    \\sum_i R_{ij}^2}` with :math:`\\mathbf u` the radial or tangential unit
    vector at pixel :math:`j`. Unlike a half-maximum width it counts arcs,
    sidelobes and ghost images, so it measures how far the information about
    pixel :math:`j` is smeared.
    """
    cen = grid.centres()
    u = cen - np.asarray(centre, dtype=np.float64)
    u /= np.maximum(np.hypot(u[:, 0], u[:, 1]), 1e-9)[:, None]
    t = np.stack([-u[:, 1], u[:, 0]], 1)
    d = cen[:, None, :] - cen[None, :, :]  # (i, j, 2): x_i - x_j
    R2 = R * R
    w = R2 / np.maximum(R2.sum(0), 1e-300)
    rad = np.sqrt((w * (d * u[None, :, :]).sum(-1) ** 2).sum(0))
    tan = np.sqrt((w * (d * t[None, :, :]).sum(-1) ** 2).sum(0))
    return rad, tan


def psf_concentration(R: NDArray, grid: PixelGrid, radius: float = 5.0) -> NDArray:
    """Fraction of each PSF's energy within ``radius`` cells of its pixel (1 = perfectly focused)."""
    cen = grid.centres()
    d2 = ((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
    R2 = R * R
    return (R2 * (d2 <= radius * radius)).sum(0) / np.maximum(R2.sum(0), 1e-300)


# ------------------------------------------------------------ emission algebra


def emission_gram(shots: Sequence[NDArray], nfft: int) -> NDArray[np.complex128]:
    """Per-frequency emission Gram :math:`M(f) = \\sum_k \\bar w_k w_k^\\top`.

    ``shots[k]`` is an ``(S, L)`` array of speaker drives for shot ``k``;
    returns ``(bins, S, S)``.
    """
    W = np.stack(
        [np.fft.rfft(np.asarray(s, dtype=np.float64), n=nfft, axis=1) for s in shots]
    )  # (K, S, bins)
    return np.einsum("ksf,ktf->fst", np.conj(W), W)


def drive_rows(kernels: NDArray, drives: NDArray, steps: int) -> NDArray:
    """Rows of one (shot, mic) trace: :math:`\\sum_s d_s * K_s`, truncated to ``steps``.

    ``kernels`` ``(S, P, T)`` are impulse-response sensitivities from each
    speaker, ``drives`` ``(S, L)``. Returns ``(P, steps)``.
    """
    S, P, T = kernels.shape
    n = T + drives.shape[1]
    Kf = np.fft.rfft(kernels, n=n, axis=2)
    Df = np.fft.rfft(drives, n=n, axis=1)
    return np.fft.irfft(np.einsum("spf,sf->pf", Kf, Df), n=n, axis=1)[:, :steps]


def virtual_elements(speakers: Sequence[Pos], mics: Sequence[Pos]) -> NDArray:
    """Far-field MIMO virtual (monostatic-equivalent) elements: midpoints of every speaker-mic pair."""
    return np.array([((s[0] + m[0]) / 2, (s[1] + m[1]) / 2) for s in speakers for m in mics])


def image_sources(
    pos: Pos, lo: float = WALL - 0.5, hi: float = GRID_N - WALL - 0.5
) -> list[tuple[float, float]]:
    """First-order image positions of ``pos`` across the four walls of the interior box."""
    r, c = pos
    return [(2 * lo - r, c), (2 * hi - r, c), (r, 2 * lo - c), (r, 2 * hi - c)]


__all__ = [
    "DT",
    "F0",
    "LISTEN",
    "Room",
    "PixelGrid",
    "Shot",
    "Config",
    "ricker_samples",
    "band_pulses",
    "chirp_code",
    "unit_energy",
    "energy",
    "laptop",
    "scheme_shots",
    "seq_shots",
    "standard_configs",
    "run_shot",
    "born_rows",
    "config_grams",
    "config_rows",
    "Grams",
    "fit_rigid_kernel",
    "reference_echo",
    "scattered_traces",
    "empty_traces",
    "eig_gram",
    "noise_sigma",
    "snr_modes",
    "dof",
    "dfs",
    "info_bits",
    "posterior_std",
    "resolution_matrix",
    "psf_widths",
    "psf_widths_polar",
    "psf_concentration",
    "psf_spread",
    "emission_gram",
    "drive_rows",
    "virtual_elements",
    "image_sources",
]
