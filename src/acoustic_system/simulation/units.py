"""SI units for scenes and datasets (plan Task 5.2).

The engine is dimension-agnostic: it only needs the wave speed ``c``, the
cell size ``dx`` and a Courant number. Grid units (``c = dx = 1``) are
convenient but hide the physical scale. The plan audit (A5) found that the
Phase 2 datasets matched no physical laptop-in-a-room scenario. This module
makes the scale explicit.

Conversions for cell size :math:`\\Delta x` [m], sound speed :math:`c` [m/s]
and Courant number :math:`\\sigma`:

* timestep :math:`\\Delta t = \\sigma\\,\\Delta x / c`;
* a grid-unit time :math:`t_g` is :math:`t_g\\,\\Delta x / c` seconds, and a
  grid-unit frequency :math:`f_g` (cycles per grid time unit) is
  :math:`f_g\\,c/\\Delta x` Hz;
* resolvable band: the grid carries wavelengths :math:`\\lambda \\ge 2\\Delta x`
  (spatial Nyquist, :math:`f \\le c/(2\\Delta x)`). With :math:`N`
  cells per wavelength for acceptable dispersion (about 8-10 for the
  second-order scheme), the useful band ends at :math:`c/(N\\Delta x)`.
"""

from __future__ import annotations

from dataclasses import dataclass

SPEED_OF_SOUND_AIR = 343.0  # m/s at 20 C


@dataclass(frozen=True)
class PhysicalScale:
    """Maps grid units to SI for a given cell size and sound speed."""

    dx: float  # metres per cell
    c: float = SPEED_OF_SOUND_AIR

    def seconds(self, t_grid: float) -> float:
        return t_grid * self.dx / self.c

    def hertz(self, f_grid: float) -> float:
        return f_grid * self.c / self.dx

    def grid_frequency(self, f_hz: float) -> float:
        return f_hz * self.dx / self.c

    def metres(self, cells: float) -> float:
        return cells * self.dx

    def cells(self, metres: float) -> float:
        return metres / self.dx

    def max_frequency(self, cells_per_wavelength: float = 8.0) -> float:
        """Highest usefully resolved frequency [Hz]."""
        return self.c / (cells_per_wavelength * self.dx)

    @staticmethod
    def for_band(
        f_max_hz: float, cells_per_wavelength: float = 8.0, c: float = SPEED_OF_SOUND_AIR
    ) -> "PhysicalScale":
        """Cell size that resolves up to ``f_max_hz`` with the given sampling."""
        return PhysicalScale(dx=c / (cells_per_wavelength * f_max_hz), c=c)


@dataclass(frozen=True)
class AcquisitionReport:
    """Physical reading of a sensing acquisition protocol."""

    dx: float
    room_m: float
    mic_baseline_m: float
    band_hz: tuple[float, float]
    wavelength_min_m: float
    duration_s: float
    notes: tuple[str, ...]

    def __str__(self) -> str:
        lines = [
            f"cell size          : {self.dx * 100:.2f} cm",
            f"room width         : {self.room_m:.2f} m",
            f"mic baseline       : {self.mic_baseline_m * 100:.1f} cm",
            f"source band        : {self.band_hz[0]:.0f}-{self.band_hz[1]:.0f} Hz",
            f"shortest wavelength: {self.wavelength_min_m * 100:.1f} cm",
            f"recording          : {self.duration_s * 1000:.1f} ms",
        ]
        return "\n".join(lines + [f"note: {n}" for n in self.notes])


def plausibility(
    grid: int,
    mic_spacing_cells: float,
    f_start_grid: float,
    f_end_grid: float,
    duration_steps: int,
    courant: float = 0.5,
    *,
    dx: float | None = None,
    mic_baseline_m: float | None = None,
    c: float = SPEED_OF_SOUND_AIR,
) -> AcquisitionReport:
    """Interpret a grid-unit acquisition protocol physically.

    Pin the scale either by the cell size ``dx`` or by the physical mic
    baseline (e.g. 0.2 m for a laptop), then check the result against a
    laptop-in-a-room scenario (room 3-8 m, baseline 10-30 cm, audio band
    below 20 kHz).
    """
    if (dx is None) == (mic_baseline_m is None):
        raise ValueError("give exactly one of dx or mic_baseline_m")
    if dx is None:
        assert mic_baseline_m is not None
        dx = mic_baseline_m / mic_spacing_cells
    sc = PhysicalScale(dx=dx, c=c)
    room = sc.metres(grid)
    base = sc.metres(mic_spacing_cells)
    band = (sc.hertz(f_start_grid), sc.hertz(f_end_grid))
    lam_min = c / band[1] if band[1] > 0 else float("inf")
    dur = sc.seconds(duration_steps * courant)
    notes = []
    if not 3.0 <= room <= 8.0:
        notes.append(f"room width {room:.2f} m is outside a typical 3-8 m room")
    if not 0.08 <= base <= 0.35:
        notes.append(f"mic baseline {base * 100:.0f} cm is outside a laptop's 8-35 cm")
    if band[1] > 20000:
        notes.append("band exceeds 20 kHz (inaudible; laptop speakers roll off far below)")
    if lam_min < 2 * dx:
        notes.append("band exceeds the spatial Nyquist limit of the grid")
    return AcquisitionReport(dx, room, base, band, lam_min, dur, tuple(notes))


__all__ = ["SPEED_OF_SOUND_AIR", "PhysicalScale", "AcquisitionReport", "plausibility"]
