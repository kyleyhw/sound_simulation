"""Room impulse-response analysis for real captures (plan 9.2-9.4, 9.7).

A NumPy port of the browser Lab's estimators (``web/src/lab/measure.ts``)
so exported captures can be scored offline and the two implementations
can be checked against each other.

* ``find_echoes``: direct sound = largest |h|; echoes = local maxima of |h|
  above ``threshold_db`` relative to the direct peak, at least
  ``min_separation_ms`` apart, strongest first, returned in time order.
  The one-way reflector distance is c * (t_echo - t_direct) / 2 (the
  speaker and mic are co-located on a laptop, a few cm apart).
* ``decay_metrics``: Schroeder backward integral
  E(t) = int_t^inf h^2, and T60 extrapolated by least squares from the
  -5..-25 dB (T20), -5..-35 dB (T30) and 0..-10 dB (EDT) ranges.
* ``sabine`` / ``eyring``: diffuse-field reverberation times of a shoebox,
  T = 24 ln(10) V / (c S a) with a = alpha (Sabine) or -ln(1 - alpha) (Eyring).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def b64_to_f32(s: str) -> NDArray[np.float32]:
    return np.frombuffer(base64.b64decode(s), dtype="<f4").copy()


def f32_to_b64(a: NDArray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()


@dataclass
class Echo:
    index: int
    delay_ms: float
    distance: float
    level_db: float


def find_echoes(
    h: NDArray,
    fs: float,
    c: float = 343.0,
    threshold_db: float = -24.0,
    min_separation_ms: float = 0.8,
    max_echoes: int = 8,
) -> tuple[int, list[Echo]]:
    env = np.abs(np.asarray(h, dtype=np.float64))
    direct = int(np.argmax(env))
    ref = env[direct] or 1.0
    sep = max(1, round(min_separation_ms / 1000 * fs))
    i = np.arange(direct + sep, len(env) - 1)
    is_peak = (env[i] >= env[i - 1]) & (env[i] > env[i + 1])
    lvl = 20 * np.log10(env[i] / ref + 1e-30)
    cand = i[is_peak & (lvl > threshold_db)]
    cand = cand[np.argsort(-env[cand], kind="stable")]
    picked: list[int] = []
    for j in cand:
        if all(abs(int(j) - p) >= sep for p in picked):
            picked.append(int(j))
        if len(picked) >= max_echoes:
            break
    picked.sort()
    out = []
    for j in picked:
        dt = (j - direct) / fs
        out.append(Echo(j, dt * 1000, dt * c / 2, float(20 * np.log10(env[j] / ref))))
    return direct, out


def decay_metrics(h: NDArray, fs: float, start: int = 0) -> dict[str, float | None]:
    x = np.asarray(h, dtype=np.float64)[start:]
    e = np.cumsum((x**2)[::-1])[::-1]
    if not e.size or e[0] <= 0:
        return {"t20": None, "t30": None, "edt": None}
    edc = 10 * np.log10(e / e[0] + 1e-30)

    def fit(hi: float, lo: float) -> float | None:
        above = np.nonzero(edc <= hi)[0]
        below = np.nonzero(edc <= lo)[0]
        if not above.size or not below.size or below[0] - above[0] < 4:
            return None
        i0, i1 = int(above[0]), int(below[0])
        t = np.arange(i0, i1 + 1) / fs
        slope = np.polyfit(t, edc[i0 : i1 + 1], 1)[0]
        return float(-60 / slope) if slope < 0 else None

    return {"t20": fit(-5, -25), "t30": fit(-5, -35), "edt": fit(0, -10)}


def _vs(lx: float, ly: float, lz: float) -> tuple[float, float]:
    return lx * ly * lz, 2 * (lx * ly + lx * lz + ly * lz)


def sabine(lx: float, ly: float, lz: float, alpha: float, c: float = 343.0) -> float:
    v, s = _vs(lx, ly, lz)
    return 24 * np.log(10) * v / (c * s * alpha)


def eyring(lx: float, ly: float, lz: float, alpha: float, c: float = 343.0) -> float:
    v, s = _vs(lx, ly, lz)
    return 24 * np.log(10) * v / (c * s * -np.log(1 - alpha))


def alpha_from_t60(lx: float, ly: float, lz: float, t60: float, c: float = 343.0) -> float:
    """Eyring inversion: the mean absorption that gives reverberation time t60."""
    v, s = _vs(lx, ly, lz)
    return float(1 - np.exp(-24 * np.log(10) * v / (c * s * t60)))
