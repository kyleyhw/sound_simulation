"""Information limits of room sensing with a small device (study 2026-09-26).

Linearised (Born) sensitivity of every recorded trace to the occupancy of
each 2 x 2-cell pixel of the loop room, for the 8-element bar, the
two-speaker device and its emission schemes, synthetic apertures,
reflective walls and other pulse bandwidths; then the whitened SVD, degrees
of freedom, CRB and resolution maps (``acoustic_system.imaging.information``).

    NUMBA_NUM_THREADS=2 uv run python scripts/eval_information.py all

Stages (each caches under ``data/information/``, HDF5 so it stays out of git):

    calibrate  fit an occupied rigid pixel's response as a K_M + b K_D, and the
               reference echo that sets the noise level
    grams      J^T J per configuration
    exact      4 x 4-pixel check: the exact rigid-pixel Jacobian (one engine run
               per pixel and shot) against the Born one
    analyse    spectra, DOF vs SNR, CRB / resolution maps -> results.json
    figures    PNGs in tests/reports/information_2026_09_26_artifacts/
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from acoustic_system.imaging import information as inf  # noqa: E402
from acoustic_system.simulation.units import LAPTOP_ROOM  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "information"
LOG = ROOT / "data" / "logs" / "information.log"
ART = ROOT / "tests" / "reports" / "information_2026_09_26_artifacts"

TAU = 0.5  # prior sd of pixel occupancy (Bernoulli(0.5) variance): weak
SNRS = (40, 30, 20, 10)
SNR_CURVE = np.arange(-10, 62, 2)
MAP_CONFIGS = ("bar8", "seq", "wide_seq", "seq_k4", "seq_k8", "seq_rigid", "wide_rigid", "seq_f16")
GROUPS = {
    "a) array": ["bar8"],
    "b) two speakers, in turn": ["seq", "wide_seq"],
    "c) simultaneous, separable": [
        "band",
        "band_raw",
        "code",
        "chirp_seq",
        "code128",
        "chirp128_seq",
    ],
    "d) summed / steered": ["sum", "beams"],
    "e) synthetic aperture": ["seq_k2", "seq_k4", "seq_k8"],
    "f) reflective walls": ["seq_plaster", "seq_wood", "seq_rigid", "wide_rigid", "bar8_rigid"],
    "g) bandwidth": ["seq_f04", "seq_f12", "seq_f16"],
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def calib_path() -> pathlib.Path:
    return CACHE / "calibration.hdf5"


def load_calibration() -> dict:
    with h5py.File(calib_path(), "r") as f:
        return json.loads(str(f.attrs["json"]))


# ------------------------------------------------------------------ stages


def stage_calibrate(force: bool = False) -> None:
    if calib_path().exists() and not force:
        log("calibrate: cached")
        return
    cfg = inf.standard_configs()
    grid = inf.PixelGrid()
    rng = np.random.default_rng(0)
    pix = rng.choice(grid.n_pix, 24, replace=False)
    out: dict = {"pixels": [int(p) for p in pix], "fits": {}}
    for name, npx in (("seq", 24), ("wide_seq", 12), ("bar8", 12), ("seq_rigid", 24)):
        t = time.time()
        r = inf.fit_rigid_kernel(cfg[name], grid, pix[:npx])
        out["fits"][name] = r
        log(
            f"calibrate {name}: a={r['a']:.3f} b={r['b']:.3f} explained={r['explained']:.3f} "
            f"(monopole only {r['explained_mono_only']:.3f}) [{time.time() - t:.0f}s]"
        )
    out["a"] = out["fits"]["seq"]["a"]
    out["b"] = out["fits"]["seq"]["b"]
    echoes = {}
    for name in ("seq", "wide_seq", "bar8", "seq_rigid"):
        echoes[name] = inf.reference_echo(cfg[name])
        log(f"reference echo {name}: {echoes[name]:.4g}")
    out["echo"] = echoes
    out["echo_ref"] = echoes["seq"]
    CACHE.mkdir(parents=True, exist_ok=True)
    with h5py.File(calib_path(), "w") as f:
        f.attrs["json"] = json.dumps(out)


def gram_path(name: str, size: int = 2) -> pathlib.Path:
    return CACHE / f"gram_{name}_p{size}.hdf5"


def stage_grams(names: list[str] | None = None, force: bool = False) -> None:
    cal = load_calibration()
    cfg = inf.standard_configs()
    grid = inf.PixelGrid()
    for name in names or list(cfg):
        p = gram_path(name)
        if p.exists() and not force:
            continue
        t = time.time()
        g = inf.config_grams(cfg[name], grid)
        gram = g.combine(cal["a"], cal["b"])
        with h5py.File(p, "w") as f:
            f["gram"] = gram
            f["mono"] = g.mm
            f.attrs["traces"] = cfg[name].n_traces
            f.attrs["time_steps"] = cfg[name].time_steps
        log(
            f"gram {name}: {cfg[name].n_traces} traces, {len(cfg[name].shots)} shots [{time.time() - t:.0f}s]"
        )


def load_gram(name: str, size: int = 2, key: str = "gram") -> np.ndarray:
    with h5py.File(gram_path(name, size), "r") as f:
        return np.asarray(f[key])


EXACT_CONFIGS = ("seq", "wide_seq", "bar8")


def stage_exact(force: bool = False) -> None:
    """Exact rigid-pixel Jacobian at 4 x 4 pixels vs the Born one (same pixels)."""
    cfg = inf.standard_configs()
    grid = inf.PixelGrid(size=4)
    for name in EXACT_CONFIGS:
        p = CACHE / f"exact_{name}_p4.hdf5"
        if p.exists() and not force:
            continue
        c = cfg[name]
        t = time.time()
        e = inf.empty_traces(c)
        J = np.stack([inf.scattered_traces(c, grid.mask(j), e) for j in range(grid.n_pix)])
        km, kd = inf.config_rows(c, grid, range(grid.n_pix))
        with h5py.File(p, "w") as f:
            f["exact"] = J
            f["km"] = km
            f["kd"] = kd
        log(f"exact {name}: {grid.n_pix} pixels [{time.time() - t:.0f}s]")


PROBES = {
    "near (row 70)": (70, 45),
    "mid (row 45)": (45, 45),
    "far (row 20)": (20, 45),
    "off-axis (45, 20)": (45, 20),
}


def _shrink(lam: np.ndarray, sig: float) -> np.ndarray:
    x = inf.snr_modes(lam, sig, TAU)
    return x / (1.0 + x)


def stage_analyse() -> dict:
    cal = load_calibration()
    cfg = inf.standard_configs()
    grid = inf.PixelGrid()
    echo = cal["echo_ref"]
    res: dict = {
        "tau": TAU,
        "echo_ref": echo,
        "calibration": {k: cal[k] for k in ("a", "b", "echo")},
        "fits": {
            k: {kk: v[kk] for kk in ("a", "b", "explained", "explained_mono_only")}
            for k, v in cal["fits"].items()
        },
        "grid": {
            "r0": grid.r0,
            "r1": grid.r1,
            "c0": grid.c0,
            "c1": grid.c1,
            "size": grid.size,
            "n_pix": grid.n_pix,
        },
        "groups": GROUPS,
        "configs": {},
    }
    maps: dict = {}
    for name, c in cfg.items():
        lam, V = inf.eig_gram(load_gram(name))
        row: dict = {
            "label": c.label,
            "room": c.room.kind,
            "shots": len(c.shots),
            "traces": c.n_traces,
            "time_steps": c.time_steps,
            "rank_1e-6": int(np.sum(lam > 1e-6 * lam[0])),
            "sv": np.sqrt(lam[:400]).tolist(),
            "dof_curve": [inf.dof(lam, inf.noise_sigma(s, echo), TAU) for s in SNR_CURVE],
            "dfs_curve": [inf.dfs(lam, inf.noise_sigma(s, echo), TAU) for s in SNR_CURVE],
            "resolved_curve": [
                float(np.mean((V * V) @ _shrink(lam, inf.noise_sigma(s, echo)) >= 0.5))
                for s in SNR_CURVE
            ],
            "snr": {},
        }
        e_tot = sum(inf.energy(d) for sh in c.shots for _, d in sh.drives)
        row["emitted_energy_rel_seq"] = e_tot / (2 * inf.RICKER_ENERGY)
        row["time_rel_seq"] = c.time_steps / (2 * inf.LISTEN)
        for s in SNRS:
            sig = inf.noise_sigma(s, echo)
            sd = inf.posterior_std(lam, V, sig, TAU)
            R = inf.resolution_matrix(lam, V, sig, TAU)
            dR = np.diag(R)
            wr, wt = inf.psf_widths_polar(R, grid)
            conc = inf.psf_concentration(R, grid)
            sr, st = inf.psf_spread(R, grid)
            probes = {}
            for pn, (r, cc) in PROBES.items():
                j = grid.index(r, cc)
                probes[pn] = {
                    "diagR": float(dR[j]),
                    "fwhm_range": float(wr[j]),
                    "fwhm_cross": float(wt[j]),
                    "concentration": float(conc[j]),
                    "rms_range": float(sr[j]),
                    "rms_cross": float(st[j]),
                    "std": float(sd[j]),
                }
            # Same measurement time as the in-turn pair (repeat and average): SNR + 10 log10(T_seq / T).
            sig_t = inf.noise_sigma(s + 10 * np.log10(1 / row["time_rel_seq"]), echo)
            # Same total emitted energy as the in-turn pair.
            sig_e = inf.noise_sigma(s - 10 * np.log10(row["emitted_energy_rel_seq"]), echo)
            row["snr"][str(s)] = {
                "dof": inf.dof(lam, sig, TAU),
                "dfs": inf.dfs(lam, sig, TAU),
                "bits": inf.info_bits(lam, sig, TAU),
                "dof_equal_time": inf.dof(lam, sig_t, TAU),
                "dof_equal_energy": inf.dof(lam, sig_e, TAU),
                "resolved_frac": float(np.mean(dR >= 0.5)),
                "mapped_frac": float(np.mean(sd < TAU / 2)),
                "median_std": float(np.median(sd)),
                "median_range_cells": float(np.nanmedian(wr)),
                "median_cross_cells": float(np.nanmedian(wt)),
                "median_concentration": float(np.median(conc)),
                "median_rms_range": float(np.median(sr)),
                "median_rms_cross": float(np.median(st)),
                "probes": probes,
            }
            if name in MAP_CONFIGS and s in (20, 30):
                maps[(name, s)] = (sd, dR, st, R, conc)
        res["configs"][name] = row
        r20, r30 = row["snr"]["20"], row["snr"]["30"]
        log(
            f"analyse {name:13s} DOF20={r20['dof']:4d} DOF30={r30['dof']:4d} DFS30={r30['dfs']:.1f} "
            f"resolved30={r30['resolved_frac']:.2f} cross30={r30['median_cross_cells']:.1f} "
            f"rms_x30={r30['median_rms_cross']:.1f} rms_r30={r30['median_rms_range']:.1f} "
            f"conc30={r30['median_concentration']:.2f}"
        )
    # Emission algebra: cross-talk of the codes, and Gram identities.
    alg = {}
    for sim, ref in (
        ("code", "chirp_seq"),
        ("code128", "chirp128_seq"),
        ("band", "seq"),
        ("sum", "seq"),
        ("beams", "seq"),
    ):
        Gs, Gr = load_gram(sim), load_gram(ref)
        alg[f"{sim}_vs_{ref}"] = {
            "rel_frobenius_diff": float(np.linalg.norm(Gs - Gr) / np.linalg.norm(Gr)),
            "trace_ratio": float(np.trace(Gs) / np.trace(Gr)),
        }
    res["emission_algebra"] = alg
    # Reflective walls / bandwidth / motion vs the same device at matched scattered energy
    # (trace of J^T J): separates the geometric (virtual-baseline) gain from the energy gain.
    em = {}
    for a, b in (
        ("seq", "seq_rigid"),
        ("seq", "seq_plaster"),
        ("seq", "seq_wood"),
        ("bar8", "bar8_rigid"),
        ("seq", "seq_f16"),
        ("seq", "seq_k4"),
    ):
        Ga, Gb = load_gram(a), load_gram(b)
        shift = float(10 * np.log10(np.trace(Gb) / np.trace(Ga)))
        la, Va = inf.eig_gram(Ga)
        for s in (20, 30):
            sig = inf.noise_sigma(s + shift, echo)
            R = inf.resolution_matrix(la, Va, sig, TAU)
            em[f"{a}_at_energy_of_{b}_{s}dB"] = {
                "shift_db": shift,
                "dof": inf.dof(la, sig, TAU),
                "resolved_frac": float(np.mean(np.diag(R) >= 0.5)),
                "median_concentration": float(np.median(inf.psf_concentration(R, grid))),
            }
    res["energy_matched"] = em
    res["exact_check"] = exact_summary(echo)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "results.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(
        CACHE / "maps.npz",
        **{
            f"{n}_{s}_{k}": v[i]
            for (n, s), v in maps.items()
            for i, k in enumerate(("std", "diagR", "cross", "R", "conc"))
        },
    )
    log(f"analyse: wrote {ART / 'results.json'}")
    return res


def exact_summary(echo: float) -> dict:
    out = {}
    for name in EXACT_CONFIGS:
        p = CACHE / f"exact_{name}_p4.hdf5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            J, km, kd = np.asarray(f["exact"]), np.asarray(f["km"]), np.asarray(f["kd"])
        A = np.stack([km.ravel(), kd.ravel()], 1)
        (a, b), *_ = np.linalg.lstsq(A, J.ravel(), rcond=None)
        Jb = a * km + b * kd
        cal = load_calibration()
        Jc = cal["a"] * km + cal["b"] * kd  # the 2 x 2 calibration applied to 4 x 4 pixels
        d: dict = {
            "a4": float(a),
            "b4": float(b),
            "explained": float(1 - np.sum((J - Jb) ** 2) / np.sum(J**2)),
        }
        for tag, M in (("exact", J), ("born_fit4", Jb), ("born_cal2", Jc)):
            lam, _ = inf.eig_gram(M @ M.T)
            d[tag] = {str(s): inf.dof(lam, inf.noise_sigma(s, echo), TAU) for s in SNRS}
            d[tag + "_sv"] = np.sqrt(lam).tolist()
        out[name] = d
    return out


# ---------------------------------------------------------------- figures


def _room_box(ax, lw: float = 1.0) -> None:
    ax.add_patch(
        plt.Rectangle(
            (inf.WALL - 0.5, inf.WALL - 0.5),
            inf.GRID_N - 2 * inf.WALL,
            inf.GRID_N - 2 * inf.WALL,
            fill=False,
            lw=lw,
            color="k",
        )
    )


def stage_figures() -> None:
    res = json.loads((ART / "results.json").read_text())
    cf = res["configs"]
    echo = res["echo_ref"]
    grid = inf.PixelGrid()
    ext = (grid.c0 - 0.5, grid.c1 - 0.5, grid.r1 - 0.5, grid.r0 - 0.5)
    mz = np.load(CACHE / "maps.npz")
    ART.mkdir(parents=True, exist_ok=True)

    # 1. singular-value spectra, normalised by the noise at 0 dB, with the prior thresholds per SNR.
    panels = [
        ("array vs two speakers", ["bar8", "seq", "wide_seq"]),
        ("emission schemes (narrow)", ["seq", "sum", "band", "code", "chirp_seq", "beams"]),
        ("synthetic aperture", ["seq", "seq_k2", "seq_k4", "seq_k8", "bar8"]),
        (
            "walls and bandwidth",
            ["seq", "seq_plaster", "seq_wood", "seq_rigid", "wide_rigid", "seq_f04", "seq_f16"],
        ),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True)
    for ax, (title, names) in zip(axs, panels):
        for n in names:
            sv = np.array(cf[n]["sv"]) * TAU / echo  # sqrt(x) at SNR 0 dB
            ax.semilogy(np.arange(1, len(sv) + 1), sv, label=n, lw=1.4)
        for s in SNRS:
            ax.axhline(10 ** (-s / 20), color="0.5", ls=":", lw=0.8)
            ax.text(395, 10 ** (-s / 20) * 1.15, f"{s} dB", ha="right", fontsize=7, color="0.4")
        ax.set_xlabel("mode index i")
        ax.set_title(title)
        ax.set_ylim(1e-4, 10)
        ax.set_xlim(0, 400)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel(r"$\sqrt{\lambda_i}\,\tau/A_{ref}$ (mode amplitude / noise at 0 dB)")
    fig.suptitle(
        "Singular values of the Born Jacobian; a mode is recoverable at SNR s where it lies above the s dB line"
    )
    fig.tight_layout()
    fig.savefig(ART / "spectra.png", dpi=110)
    plt.close(fig)

    # 2. DOF and resolved fraction vs SNR, for a curated set.
    key = [
        ("bar8", "k", "-"),
        ("bar8_rigid", "k", ":"),
        ("seq", "C0", "-"),
        ("wide_seq", "C0", "--"),
        ("sum", "C1", "-"),
        ("band", "C1", "--"),
        ("code", "C1", ":"),
        ("chirp_seq", "C1", "-."),
        ("seq_k2", "C2", ":"),
        ("seq_k4", "C2", "-"),
        ("seq_k8", "C2", "--"),
        ("seq_rigid", "C3", "-"),
        ("seq_wood", "C3", "--"),
        ("seq_f16", "C4", "-"),
        ("seq_f04", "C4", ":"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for n, col, ls in key:
        axs[0].plot(SNR_CURVE, cf[n]["dof_curve"], color=col, ls=ls, label=n, lw=1.5)
        axs[1].plot(SNR_CURVE, cf[n]["resolved_curve"], color=col, ls=ls, label=n, lw=1.5)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("recoverable DOF (modes above the prior)")
    axs[1].set_ylabel("fraction of pixels with diag R >= 0.5")
    for ax in axs:
        ax.set_xlabel("SNR (dB, per-sample noise vs the reference echo peak)")
        ax.grid(alpha=0.3)
        for s_ in (20, 30):
            ax.axvline(s_, color="0.6", lw=0.7)
    axs[0].axhline(grid.n_pix, color="0.5", lw=0.8)
    axs[0].text(-9, grid.n_pix * 1.05, "number of pixels", fontsize=7, color="0.4")
    axs[1].legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(ART / "dof_vs_snr.png", dpi=110)
    plt.close(fig)

    # 3. Maps: posterior std, diag R and cross-range PSF width, at 20 and 30 dB.
    for s in (20, 30):
        names = [n for n in MAP_CONFIGS if f"{n}_{s}_std" in mz]
        fig, axs = plt.subplots(
            3, len(names), figsize=(2.3 * len(names), 7.0), layout="constrained"
        )
        for k, n in enumerate(names):
            for r, (key, cmap, vmin, vmax, lab) in enumerate(
                (
                    ("std", "magma_r", 0, TAU, "posterior std"),
                    ("diagR", "viridis", 0, 1, "diag R"),
                    ("cross", "cividis", 0, 30, "tangential RMS PSF extent (cells)"),
                )
            ):
                img = mz[f"{n}_{s}_{key}"].reshape(grid.shape)
                ax = axs[r, k]
                im = ax.imshow(img, extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
                _elements(ax, n)
                ax.set_xlim(8, 92)
                ax.set_ylim(92, 8)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(n, fontsize=9)
                if k == len(names) - 1:
                    fig.colorbar(im, ax=axs[r, :].tolist(), shrink=0.8, label=lab)
        fig.suptitle(
            f"Per-pixel CRB (with prior sd {TAU}), resolvability diag R, tangential (cross-range) RMS extent of the PSF in cells; SNR {s} dB"
        )
        fig.savefig(ART / f"maps_{s}dB.png", dpi=100)
        plt.close(fig)

    # 4. PSFs at probe points, 30 dB.
    names = ["bar8", "seq", "wide_seq", "seq_k4", "seq_rigid"]
    fig, axs = plt.subplots(len(PROBES), len(names), figsize=(2.6 * len(names), 2.5 * len(PROBES)))
    for k, n in enumerate(names):
        R = mz[f"{n}_30_R"]
        for r, (pn, (pr, pc)) in enumerate(PROBES.items()):
            j = grid.index(pr, pc)
            img = R[:, j].reshape(grid.shape)
            ax = axs[r, k]
            v = max(abs(img).max(), 1e-12)
            ax.imshow(img, extent=ext, cmap="RdBu_r", vmin=-v, vmax=v)
            ax.plot(pc, pr, "k+", ms=8)
            _elements(ax, n)
            ax.set_xlim(8, 92)
            ax.set_ylim(92, 8)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(n, fontsize=9)
            if k == 0:
                ax.set_ylabel(pn, fontsize=8)
            ax.text(10, 88, f"R_jj={img.flat[j]:.2f}", fontsize=7)
    fig.suptitle("Point-spread functions (columns of R) at 30 dB")
    fig.tight_layout()
    fig.savefig(ART / "psf_30dB.png", dpi=100)
    plt.close(fig)

    # 5. Schematic: real and virtual elements.
    fig, axs = plt.subplots(1, 4, figsize=(17, 5.2), width_ratios=[1, 1, 1, 1.4])
    cases = [
        ("8-element bar", inf.BAR, inf.BAR, []),
        ("2 speakers (narrow / wide)", *inf.laptop(45), []),
        (
            "K = 4 placements",
            sum((inf.laptop(c)[0] for c in inf.PLACEMENTS[4]), []),
            sum((inf.laptop(c)[1] for c in inf.PLACEMENTS[4]), []),
            [],
        ),
        (
            "rigid walls: first-order image speakers",
            *inf.laptop(45),
            [im for s in inf.laptop(45)[0] for im in inf.image_sources(s)],
        ),
    ]
    for ax, (ttl, sp, mi, imgs) in zip(axs, cases):
        _room_box(ax)
        ax.add_patch(
            plt.Rectangle(
                (grid.c0 - 0.5, grid.r0 - 0.5),
                grid.c1 - grid.c0,
                grid.r1 - grid.r0,
                fill=True,
                alpha=0.08,
                color="C2",
            )
        )
        sp_a = np.array(sp)
        mi_a = np.array(mi)
        if ttl.startswith("K"):
            virt = np.concatenate([inf.virtual_elements(*inf.laptop(c)) for c in inf.PLACEMENTS[4]])
        else:
            virt = inf.virtual_elements(sp, mi)
        ax.plot(sp_a[:, 1], sp_a[:, 0], "rv", ms=7, label="speaker")
        ax.plot(mi_a[:, 1], mi_a[:, 0], "b^", ms=6, label="mic")
        ax.plot(virt[:, 1], virt[:, 0] + 3, "k.", ms=5, label="virtual element (pair midpoint)")
        if ttl.startswith("2"):
            ws, wm = inf.laptop(45, 28)
            ax.plot(
                np.array(ws)[:, 1],
                np.array(ws)[:, 0],
                "rv",
                mfc="none",
                ms=9,
                label="wide speakers",
            )
            wv = inf.virtual_elements(ws, wm)
            ax.plot(
                wv[:, 1], wv[:, 0] + 6, "o", color="0.4", mfc="none", ms=4, label="wide virtual"
            )
        if imgs:
            ia = np.array(imgs)
            ax.plot(ia[:, 1], ia[:, 0], "v", color="orange", ms=8, label="image speaker")
            for m in mi:
                for im in imgs:
                    ax.plot([im[1], m[1]], [im[0], m[0]], color="orange", lw=0.3, alpha=0.6)
        if imgs:
            ax.set_xlim(-45, 145)
            ax.set_ylim(105, -80)
        else:
            ax.set_xlim(0, 100)
            ax.set_ylim(100, 0)
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.set_xlabel("column (cells, 2.5 cm)")
    axs[0].set_ylabel("row (cells)")
    fig.suptitle("Real and virtual element positions (shaded: pixels analysed; box: room interior)")
    fig.tight_layout()
    fig.savefig(ART / "schematic.png", dpi=110)
    plt.close(fig)

    # 6. Exact vs Born spectra (4 x 4 pixels).
    ex = res.get("exact_check", {})
    if ex:
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for k, (n, d) in enumerate(ex.items()):
            for tag, ls in (("exact", "-"), ("born_fit4", "--"), ("born_cal2", ":")):
                sv = np.array(d[tag + "_sv"]) * TAU / echo
                ax.semilogy(np.arange(1, len(sv) + 1), sv, color=f"C{k}", ls=ls, label=f"{n} {tag}")
        for s in SNRS:
            ax.axhline(10 ** (-s / 20), color="0.5", ls=":", lw=0.8)
        ax.set_ylim(1e-4, 30)
        ax.set_xlabel("mode index (4 x 4-cell pixels, 304)")
        ax.set_ylabel(r"$\sqrt{\lambda_i}\,\tau/A_{ref}$")
        ax.set_title("Exact rigid-pixel Jacobian vs Born (4 x 4 pixels)")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ART / "exact_vs_born.png", dpi=110)
        plt.close(fig)
    log("figures written")


def _elements(ax, name: str) -> None:
    cfg = inf.standard_configs()[name]
    sp, mi = cfg.positions()
    ax.plot([p[1] for p in sp], [p[0] for p in sp], "rv", ms=3)
    ax.plot([p[1] for p in mi], [p[0] for p in mi], "c^", ms=2.5)


def physical_scale() -> dict:
    lr = LAPTOP_ROOM
    return {
        "dx_m": lr.dx,
        "f0_hz": lr.scale.hertz(inf.F0),
        "ricker_band_hz": [lr.scale.hertz(0.02), lr.scale.hertz(0.2)],
        "wavelength_f0_cells": 1 / inf.F0,
        "narrow_spacing_m": 12 * lr.dx,
        "wide_spacing_m": 28 * lr.dx,
        "room_interior_m": (inf.GRID_N - 2 * inf.WALL) * lr.dx,
        "record_ms": 1e3 * lr.scale.seconds(inf.LISTEN * inf.DT),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["calibrate", "grams", "exact", "analyse", "figures", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--configs", nargs="*")
    a = ap.parse_args()
    if a.stage in ("calibrate", "all"):
        stage_calibrate(a.force)
    if a.stage in ("grams", "all"):
        stage_grams(a.configs, a.force)
    if a.stage in ("exact", "all"):
        stage_exact(a.force)
    if a.stage in ("analyse", "all"):
        res = stage_analyse()
        res["physical"] = physical_scale()
        (ART / "results.json").write_text(json.dumps(res, indent=1))
    if a.stage in ("figures", "all"):
        stage_figures()


if __name__ == "__main__":
    main()
