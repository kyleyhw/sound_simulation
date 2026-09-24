"""Physics verification suite (plan Task 5.3, 5.4.3, 5.5.3).

Checks the FDTD engine against analytic results rather than a regression
snapshot, and writes figures + a JSON summary for the report:

1. Cavity eigenfrequencies (pressure-release and rigid boxes) vs the
   analytic modes and the scheme's discrete dispersion relation.
2. Grid-convergence order, from an exact standing-mode solution.
3. Discrete energy conservation in lossless boxes.
4. Numerical dispersion: measured mode frequencies against continuum and
   von Neumann predictions, along an axis and the diagonal.
5. Point-source response vs the analytic 2D and 3D Green's functions.
6. Reverberation time of absorbing rooms vs Sabine and Eyring.
7. Reflection of the absorbing edges (Mur, sponge, CPML) vs angle and frequency.

    uv run python scripts/verify_physics.py --out tests/reports/physics_artifacts
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from acoustic_system.simulation.physics import absorption_from_beta  # noqa: E402
from acoustic_system.simulation.setup import Driver  # noqa: E402
from acoustic_system.simulation.simulate import Simulate  # noqa: E402
from acoustic_system.simulation.waveforms import RickerWavelet  # noqa: E402


def peak_freq(x: np.ndarray, dt: float, f_lo: float, f_hi: float) -> float:
    """Spectral peak in [f_lo, f_hi] with parabolic interpolation (Hann, 8x pad)."""
    n = len(x)
    w = np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    spec = np.abs(np.fft.rfft(x * w, nfft))
    f = np.fft.rfftfreq(nfft, dt)
    sel = np.where((f >= f_lo) & (f <= f_hi))[0]
    k = sel[np.argmax(spec[sel])]
    a, b, c = np.log(spec[k - 1 : k + 2] + 1e-30)
    off = 0.5 * (a - c) / (a - 2 * b + c)
    return float(f[k] + off * (f[1] - f[0]))


def mode_shapes(n: int, ms: list[int], boundary: str) -> np.ndarray:
    """Exact 1D discrete eigenvectors of the engine's Laplacian, one row per m.

    Pressure-release (Dirichlet nodes 0 and n-1 held at 0): sin(pi m i/(n-1)).
    Rigid (cell-centred Neumann, ghost = mirror): cos(pi m (i + 1/2)/n).
    """
    i = np.arange(n)
    if boundary == "soft":
        return np.array([np.sin(np.pi * m * i / (n - 1)) for m in ms])
    return np.array([np.cos(np.pi * m * (i + 0.5) / n) for m in ms])


def modal_series(
    sim: Simulate, modes: list[tuple[int, int]], boundary: str, steps: int
) -> np.ndarray:
    """Run ``steps`` and return each mode's amplitude over time, (steps, len(modes)).

    Projecting the whole field onto an exact eigenvector isolates that mode,
    so near-degenerate modes (e.g. (3,1) and (2,2) in a 41x31 box) can no
    longer be confused by a spectral peak search, and every trace is a
    single sinusoid whose frequency is measured to ~1e-6.
    """
    nx, ny = sim.grid_shape
    mx = sorted({m for m, _ in modes})
    my = sorted({k for _, k in modes})
    bx = mode_shapes(nx, mx, boundary)
    by = mode_shapes(ny, my, boundary)
    ix = [mx.index(m) for m, _ in modes]
    iy = [my.index(k) for _, k in modes]
    out = np.empty((steps, len(modes)))
    for t in range(steps):
        sim.step()
        a = bx @ sim.p.astype(np.float64) @ by.T
        out[t] = a[ix, iy]
    return out


def discrete_freq(k: list[float], sigma: float, dt: float, dx: float) -> float:
    """von Neumann: sin^2(w dt/2) = sigma^2 sum sin^2(k_a dx/2)."""
    s = sigma**2 * sum(np.sin(ka * dx / 2) ** 2 for ka in k)
    return float(2 * np.arcsin(np.sqrt(s)) / dt / (2 * np.pi))


# ---------------------------------------------------------------------------
# 1. Cavity modes
# ---------------------------------------------------------------------------


def cavity_modes() -> dict:
    out = {}
    nx, ny = 41, 31
    for boundary in ("soft", "rigid"):
        sim = Simulate((nx, ny), boundary=boundary)
        sim.set_drivers([Driver((7, 5), RickerWavelet(1.0, 0.25, 6.0))])
        # Effective lengths: Dirichlet nodes 0..N-1 held at 0 -> L=(N-1)dx;
        # rigid cell-centred Neumann (ghost = mirror) -> L = N dx.
        lx, ly = (nx - 1, ny - 1) if boundary == "soft" else (nx, ny)
        modes = (
            [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)]
            if boundary == "rigid"
            else [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1)]
        )
        series = modal_series(sim, modes, boundary, 40000)
        rows = []
        for q, (m, n) in enumerate(modes):
            kx, ky = np.pi * m / lx, np.pi * n / ly
            f_cont = 0.5 * np.hypot(m / lx, n / ly)
            f_disc = discrete_freq([kx, ky], np.sqrt(sim._coeff), sim.timestep, 1.0)
            f_meas = peak_freq(series[:, q], sim.timestep, f_cont * 0.9, f_cont * 1.1)
            rows.append(
                {
                    "mode": [m, n],
                    "f_continuum": f_cont,
                    "f_discrete": f_disc,
                    "f_measured": f_meas,
                    "err_vs_continuum_pct": 100 * (f_meas / f_cont - 1),
                    "err_vs_discrete_pct": 100 * (f_meas / f_disc - 1),
                }
            )
        out[boundary] = rows
    return out


# ---------------------------------------------------------------------------
# 2. Convergence order (exact standing mode)
# ---------------------------------------------------------------------------


def convergence() -> dict:
    rows = []
    t_end = 1.0
    for n in (16, 32, 64, 128):
        dx = 1.0 / n
        sim = Simulate((n + 1, n + 1), gridstep=dx, wavespeed=1.0, courant=0.5)
        x = np.arange(n + 1) * dx
        mode = np.outer(np.sin(np.pi * x), np.sin(np.pi * x)).astype(np.float32)
        omega = np.pi * np.sqrt(2.0)
        dt = sim.timestep
        sim.p[...] = mode * np.float32(np.cos(omega * dt))  # p^1
        sim.p_prev[...] = mode  # p^0
        sim.time = dt
        steps = int(round((t_end - dt) / dt))
        for _ in range(steps):
            sim.step()
        exact = mode * np.cos(omega * sim.time)
        err = float(np.abs(sim.p - exact).max())
        rows.append({"n": n, "dx": dx, "max_error": err})
    for a, b in zip(rows, rows[1:]):
        b["order"] = float(np.log2(a["max_error"] / b["max_error"]))
    return {"rows": rows}


# ---------------------------------------------------------------------------
# 3. Energy
# ---------------------------------------------------------------------------


def energy(sim: Simulate, solid: np.ndarray | None = None) -> float:
    """Discrete leap-frog energy (conserved exactly by the lossless scheme).

    E = 1/2 sum ((p - p_prev)/dt)^2 + 1/2 sum_faces (D p)(D p_prev),
    over fluid cells and fluid-fluid faces. Rigid cells are outside the
    domain (their faces carry no flux), so pairs touching ``solid`` are
    dropped; pressure-release cells are part of the domain (p = 0 there),
    so their faces stay in.
    """
    p = sim.p.astype(np.float64)
    pp = sim.p_prev.astype(np.float64)
    fluid = np.ones(p.shape, bool) if solid is None else ~solid
    kin = ((((p - pp) / sim.timestep) ** 2) * fluid).sum()
    pot = 0.0
    for a in range(p.ndim):
        both = np.logical_and(
            np.take(fluid, range(1, p.shape[a]), axis=a),
            np.take(fluid, range(0, p.shape[a] - 1), axis=a),
        )
        pot += (np.diff(p, axis=a) * np.diff(pp, axis=a) * both).sum()
    return 0.5 * kin + 0.5 * pot


def energy_conservation() -> dict:
    out = {}
    for boundary in ("soft", "rigid"):
        sim = Simulate((128, 128), boundary=boundary)
        m = np.zeros((128, 128), np.uint8)
        m[40:60, 70:90] = 2 if boundary == "rigid" else 1
        sim.set_material_map(m)
        sim.set_drivers([Driver((64, 30), RickerWavelet(1.0, 0.1, 15.0))])
        for _ in range(100):
            sim.step()
        solid = m == 2
        e = [energy(sim, solid)]
        for _ in range(20):
            for _ in range(500):
                sim.step()
            e.append(energy(sim, solid))
        e = np.array(e)
        out[boundary] = {"steps": 10000, "max_rel_drift": float(np.abs(e / e[0] - 1).max())}
    return out


# ---------------------------------------------------------------------------
# 4. Dispersion (mode frequencies across the band)
# ---------------------------------------------------------------------------


def dispersion() -> dict:
    n = 64
    sim = Simulate((n, n), boundary="rigid")
    sim.set_drivers([Driver((3, 2), RickerWavelet(1.0, 0.45, 3.0))])
    groups = (
        ("axis", [(m, 0) for m in range(2, 50, 4)]),
        ("diagonal", [(m, m) for m in range(2, 36, 3)]),
    )
    all_modes = [md for _, g in groups for md in g]
    series = modal_series(sim, all_modes, "rigid", 30000)
    sig = np.sqrt(sim._coeff)
    rows = []
    for axis_name, modes in groups:
        for m, k2 in modes:
            x = series[:, all_modes.index((m, k2))]
            kx, ky = np.pi * m / n, np.pi * k2 / n
            f_cont = 0.5 * np.hypot(m / n, k2 / n)
            f_disc = discrete_freq([kx, ky], sig, sim.timestep, 1.0)
            f_meas = peak_freq(
                x, sim.timestep, min(f_cont, f_disc) * 0.9, max(f_cont, f_disc) * 1.1
            )
            lam = 1.0 / f_cont  # cells per wavelength (c = dx = 1)
            rows.append(
                {
                    "direction": axis_name,
                    "cells_per_wavelength": lam,
                    "phase_speed_measured": f_meas / f_cont,
                    "phase_speed_theory": f_disc / f_cont,
                }
            )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# 5. Green's functions
# ---------------------------------------------------------------------------


def greens() -> tuple[dict, plt.Figure]:
    out = {}
    wf = RickerWavelet(1.0, 0.05, 40.0)
    # 2D: p = (dx^2/dt^2) int w'(tau) F(t - tau) dtau, F = arccosh(t/r)/(2 pi)
    n = 400
    sim = Simulate((n, n))
    c0 = n // 2
    r = 40
    sim.set_drivers([Driver((c0, c0), wf)])
    tr = []
    for _ in range(int(260 / sim.timestep)):
        sim.step()
        tr.append(float(sim.p[c0, c0 + r]))
    num = np.array(tr)
    dt = sim.timestep
    t = (np.arange(len(num)) + 1) * dt  # p after step k is at time (k+1) dt
    tau = np.arange(0, t[-1], 0.01)
    wdot = np.gradient(np.array([wf(v) for v in tau]), tau)

    def F(s: np.ndarray) -> np.ndarray:
        return np.where(s > r, np.arccosh(np.maximum(s / r, 1.0)) / (2 * np.pi), 0.0)

    # Injection at t_n enters p^{n+1}: continuous source time is t_n.
    ana = np.array([np.trapezoid(wdot * F(tt - dt - tau), tau) for tt in t]) / dt**2
    corr = float(np.corrcoef(num, ana)[0, 1])
    amp = float(np.abs(num).max() / np.abs(ana).max())
    out["2d"] = {"r_cells": r, "correlation": corr, "amplitude_ratio": amp}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(t, num, label="FDTD")
    ax[0].plot(t, ana, "--", label="analytic")
    ax[0].set_title(f"2D point source, r = {r} cells")
    ax[0].set_xlabel("t")
    ax[0].legend()
    # 3D: p = dx^3/dt^2 * w(t - r) / (4 pi r)
    n3 = 96
    sim3 = Simulate((n3, n3, n3))
    c3 = n3 // 2
    r3 = 24
    sim3.set_drivers([Driver((c3, c3, c3), wf)])
    tr3 = []
    for _ in range(int(120 / sim3.timestep)):
        sim3.step()
        tr3.append(float(sim3.p[c3, c3, c3 + r3]))
    num3 = np.array(tr3)
    dt3 = sim3.timestep
    t3 = (np.arange(len(num3)) + 1) * dt3
    ana3 = np.array([wf(v - dt3 - r3) for v in t3]) / (4 * np.pi * r3) / dt3**2
    # Compare before the first wall reflection (path 2 * (n3 - c3) - r3 = 72
    # cells, peak at t = 40 + 72) reaches the probe.
    keep = t3 < 92
    num3, ana3, t3 = num3[keep], ana3[keep], t3[keep]
    corr3 = float(np.corrcoef(num3, ana3)[0, 1])
    amp3 = float(np.abs(num3).max() / np.abs(ana3).max())
    out["3d"] = {"r_cells": r3, "correlation": corr3, "amplitude_ratio": amp3}
    ax[1].plot(t3, num3, label="FDTD")
    ax[1].plot(t3, ana3, "--", label="analytic")
    ax[1].set_title(f"3D point source, r = {r3} cells")
    ax[1].set_xlabel("t")
    ax[1].legend()
    fig.tight_layout()
    return out, fig


# ---------------------------------------------------------------------------
# 6. Reverberation time vs Sabine / Eyring (2D)
# ---------------------------------------------------------------------------


def alpha_diffuse_2d(beta: float) -> float:
    """Random-incidence absorption of a locally reacting wall in 2D.

    Energy flux onto a wall from a 2D diffuse field is weighted by cos(theta)
    for theta in (-pi/2, pi/2): alpha_d = (1/2) int alpha(theta) cos(theta).
    """
    th = np.linspace(-np.pi / 2, np.pi / 2, 4001)
    r = (np.cos(th) - beta) / (np.cos(th) + beta)
    return float(0.5 * np.trapezoid((1 - r**2) * np.cos(th), th))


def t60_from_decay(x: np.ndarray, dt: float) -> float:
    """T60 from a Schroeder backward integral, fitted on -5..-25 dB (T20 x 3)."""
    e = np.cumsum((x.astype(np.float64) ** 2)[::-1])[::-1]
    if not e[0] > 0:
        raise ValueError("silent trace (probe inside a wall?)")
    db = 10 * np.log10(e / e[0] + 1e-30)
    i0 = int(np.argmax(db <= -5))
    i1 = int(np.argmax(db <= -25))
    t = np.arange(len(x)) * dt
    slope = np.polyfit(t[i0:i1], db[i0:i1], 1)[0]
    return float(-60.0 / slope)


def reverberation() -> dict:
    rows = []
    nx, ny = 160, 110
    rng = np.random.default_rng(0)
    for beta in (0.05, 0.1, 0.2):
        sim = Simulate((nx, ny), boundary="absorb", boundary_beta=beta)
        # Rigid scatterers make the field closer to diffuse.
        m = np.zeros((nx, ny), np.uint8)
        for _ in range(14):
            ci, cj = rng.integers(15, nx - 15), rng.integers(15, ny - 15)
            m[ci - 2 : ci + 3, cj - 2 : cj + 3] = 2
        probes = [(120, 80), (60, 85), (130, 25), (90, 55)]
        for q in probes + [(40, 30)]:  # keep scatterers off the source and mics
            m[q[0] - 3 : q[0] + 4, q[1] - 3 : q[1] + 4] = 0
        sim.set_material_map(m)
        sim.set_drivers([Driver((40, 30), RickerWavelet(5.0, 0.12, 10.0))])
        rec = []
        for _ in range(int(9000 / sim.timestep)):
            sim.step()
            rec.append([float(sim.p[q]) for q in probes])
        rec = np.array(rec)
        t60 = float(np.mean([t60_from_decay(rec[:, k], sim.timestep) for k in range(len(probes))]))
        area = nx * ny
        perim = 2 * (nx + ny)
        a = alpha_diffuse_2d(beta)
        # 2D: energy decays as exp(-c alpha L t / (pi S)); T60 = 6 ln10 pi S/(c L a)
        t_sab = 6 * np.log(10) * np.pi * area / (perim * a)
        t_eyr = 6 * np.log(10) * np.pi * area / (perim * -np.log(1 - a))
        rows.append(
            {
                "beta": beta,
                "alpha_normal": absorption_from_beta(beta),
                "alpha_diffuse_2d": a,
                "t60_fdtd": t60,
                "t60_sabine": float(t_sab),
                "t60_eyring": float(t_eyr),
            }
        )
    return {"rows": rows, "room_cells": [nx, ny]}


# ---------------------------------------------------------------------------
# 7. Absorbing-edge reflection vs angle and frequency
# ---------------------------------------------------------------------------


def edge_reflection() -> dict:
    """Image-source method: reflected = trace(domain) - trace(free field).

    The source sits d cells above the bottom edge of a wide domain; probes at
    the same depth, lateral offset x, see the edge's reflection arrive from the
    image source at incidence theta = atan(x / 2d). The reflected spectrum is
    divided by the free-field spectrum at the image distance sqrt(4d^2 + x^2),
    both from a reflection-free reference domain (walls too far to return
    within the window).
    """
    d = 40
    offsets = [0, 30, 60, 100, 150]
    wf = RickerWavelet(1.0, 0.08, 20.0)
    steps = 840
    rows, cols = 400, 800
    src = (rows - 1 - d, cols // 2)
    probes = [(src[0], src[1] + x) for x in offsets]
    n_ref = 1000
    c = n_ref // 2

    def run(shape, boundary, s, ps):
        sim = Simulate(shape, boundary=boundary, sponge_cells=24, cpml_cells=24)
        sim.set_drivers([Driver(s, wf)])
        out = []
        for _ in range(steps):
            sim.step()
            out.append([float(sim.p[q]) for q in ps])
        return np.array(out)

    ref = run((n_ref, n_ref), "soft", (c, c), [(c, c + x) for x in offsets])
    img = run((n_ref, n_ref), "soft", (c, c), [(c + 2 * d, c + x) for x in offsets])
    freqs = np.fft.rfftfreq(steps, 0.5)
    band = (freqs > 0.03) & (freqs < 0.15)
    win = np.hanning(steps)
    out: dict = {
        "angles_deg": [float(np.degrees(np.arctan2(x, 2 * d))) for x in offsets],
        "freqs": freqs[band].tolist(),
    }
    for boundary in ("soft", "mur", "sponge", "cpml"):
        tr = run((rows, cols), boundary, src, probes)
        refl = tr - ref
        rdb = []
        for k in range(len(offsets)):
            spec_r = np.abs(np.fft.rfft(refl[:, k] * win))
            spec_i = np.abs(np.fft.rfft(img[:, k] * win))
            rdb.append((20 * np.log10(spec_r[band] / (spec_i[band] + 1e-12) + 1e-12)).tolist())
        out[boundary] = rdb
        out[boundary + "_worst_db"] = [float(np.max(r)) for r in rdb]
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="tests/reports/physics_artifacts")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    timings: dict = {}

    def run(name, fn):
        if args.only and name not in args.only:
            return None
        t0 = time.perf_counter()
        r = fn()
        timings[name] = time.perf_counter() - t0
        print(f"[verify] {name}: {timings[name]:.1f} s", flush=True)
        return r

    r = run("modes", cavity_modes)
    if r:
        results["modes"] = r
    r = run("convergence", convergence)
    if r:
        results["convergence"] = r
    r = run("energy", energy_conservation)
    if r:
        results["energy"] = r
    r = run("dispersion", dispersion)
    if r:
        results["dispersion"] = r
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for name, mk in (("axis", "o"), ("diagonal", "s")):
            rows = [x for x in r["rows"] if x["direction"] == name]
            lam = [x["cells_per_wavelength"] for x in rows]
            ax.plot(lam, [x["phase_speed_measured"] for x in rows], mk, label=f"measured ({name})")
            ax.plot(
                lam, [x["phase_speed_theory"] for x in rows], "-", label=f"von Neumann ({name})"
            )
        ax.set_xscale("log")
        ax.set_xlabel("cells per wavelength")
        ax.set_ylabel("numerical / true phase speed")
        ax.axhline(1, color="k", lw=0.6)
        ax.legend(fontsize=8)
        ax.set_title("Numerical dispersion (2nd-order scheme, sigma = 0.5)")
        fig.tight_layout()
        fig.savefig(out / "dispersion.png", dpi=130)
    r = run("greens", greens)
    if r:
        results["greens"], fig = r
        fig.savefig(out / "greens.png", dpi=130)
    r = run("reverberation", reverberation)
    if r:
        results["reverberation"] = r
    r = run("edges", edge_reflection)
    if r:
        results["edges"] = r
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.4), sharey=True)
        for ax, name in zip(axs, ("mur", "sponge", "cpml")):  # soft ~ 0 dB is the sanity check
            for k, ang in enumerate(r["angles_deg"]):
                ax.plot(r["freqs"], r[name][k], label=f"{ang:.0f}°")
            ax.set_title(f"{name}: reflection vs frequency")
            ax.set_xlabel("frequency (cycles per time unit)")
            ax.axhline(-40, color="k", lw=0.6, ls=":")
        axs[0].set_ylabel("reflection (dB)")
        axs[2].legend(title="incidence", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "edges.png", dpi=130)
    results["timings_s"] = timings
    path = out / "verification.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(results)
    path.write_text(json.dumps(old, indent=2))
    print(f"[verify] wrote {path}")


if __name__ == "__main__":
    main()
