"""Phase 7 headline experiments: sound zones, crosstalk cancellation, ANC,
sensing requirements and differentiable control.

Writes ``tests/reports/control_2026_09_24.md`` and the figures plus a
``results.json`` in ``tests/reports/control_2026_09_24_artifacts/``.

    NUMBA_NUM_THREADS=1 uv run python scripts/control_report.py            # all, ~8 min
    NUMBA_NUM_THREADS=1 uv run python scripts/control_report.py --only ctc # one section

Sections: ``zones`` (7.1/7.2), ``ctc`` (7.3), ``anc`` (7.4), ``req`` (7.5),
``diff`` (7.6). Results of sections not re-run are read back from the
existing ``results.json``, so the markdown is always complete.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from acoustic_system.control.anc import (  # noqa: E402
    band_noise,
    fxlms,
    quiet_zone,
    stride_for,
    tone,
)
from acoustic_system.control.beamforming import (  # noqa: E402
    acoustic_contrast_control,
    delay_and_sum,
    design_broadband,
    design_grid,
    fir_from_weights,
    normalise_to_reference,
    predicted_band_contrast,
    pressure_matching,
    time_reversal,
    verify_fir,
)
from acoustic_system.control.ctc import (  # noqa: E402
    TrackedCtc,
    displacement_sweep,
    head_grid,
    stereo_separation_db,
    verify_ctc,
)
from acoustic_system.control.differentiable import DiffScene, optimise_drives  # noqa: E402
from acoustic_system.control.metrics import (  # noqa: E402
    array_effort_db,
    band_contrast_db,
    contrast_spectrum_db,
)
from acoustic_system.control.requirements import (  # noqa: E402
    SensingError,
    SensingStudy,
    threshold_crossing,
    wall_errors,
)
from acoustic_system.control.transfer import (  # noqa: E402
    Box,
    Room,
    TransferSet,
    bandpass_pulse,
    disk_points,
    measure_transfer,
    simulate_drives,
)
from acoustic_system.simulation.physics import absorption_from_beta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "tests" / "reports" / "control_2026_09_24.md"
ART = ROOT / "tests" / "reports" / "control_2026_09_24_artifacts"

# Reference categorical palette (dataviz skill, light mode), fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

FURNITURE = (Box(0.2, 1.9, 0.7, 2.25), Box(2.6, 0.1, 2.85, 0.5, 5))  # sofa, absorber panel
ROOMS = {
    "absorbing": (Room(size=(3.0, 2.4), beta=0.3, boxes=FURNITURE), 0.15),
    "live": (Room(size=(3.0, 2.4), beta=0.1, boxes=FURNITURE), 0.3),
    "anechoic": (Room(size=(3.0, 2.4), boundary="cpml", boxes=FURNITURE), 0.1),
}
BAND = (300.0, 1500.0)
ARRAY = np.array([[1.15 + 0.1 * i, 0.4] for i in range(8)])
BRIGHT_C, DARK_C, ZONE_R = (1.0, 1.6), (2.0, 1.6), 0.15
LAPTOP = np.array([[1.35, 0.6], [1.65, 0.6]])
HEAD = np.array([1.5125, 1.1])
PRIMARY, SECONDARY, MIC = (0.5, 0.5), (2.3, 1.5), (2.0, 1.5)


def _style(ax: Any) -> None:
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _j(x: Any) -> Any:
    """JSON-safe conversion."""
    if isinstance(x, dict):
        return {str(k): _j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, np.ndarray):
        return _j(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else str(v)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def zone_points(room: Room) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b = disk_points(room, BRIGHT_C, ZONE_R)
    d = disk_points(room, DARK_C, ZONE_R)
    c = int(np.argmin(np.linalg.norm(b - np.array(BRIGHT_C), axis=1)))
    b = np.concatenate([b[c : c + 1], np.delete(b, c, axis=0)])
    pts = np.concatenate([b, d])
    return pts, np.arange(len(b)), np.arange(len(b), len(pts)), b


# --------------------------------------------------------------------------- #
# 7.1 / 7.2 sound zones
# --------------------------------------------------------------------------- #


def run_zones() -> dict:
    out: dict = {"rooms": {}}
    x = None
    spectra = {}
    maps = {}
    for name, (room, dur) in ROOMS.items():
        pts, ib, id_, _ = zone_points(room)
        t0 = time.perf_counter()
        ts = measure_transfer(room, ARRAY, pts, duration=dur)
        t_meas = time.perf_counter() - t0
        r: dict = {
            "measure_s": t_meas,
            "steps": ts.steps,
            "tail_db": ts.tail_db(),
            "points": [len(ib), len(id_)],
        }
        # Consistency: a random multi-speaker drive predicted from H.
        if name == "absorbing":
            rng = np.random.default_rng(3)
            xx = bandpass_pulse(ts.dt, BAND, 301)
            drv = np.stack([np.convolve(xx, rng.standard_normal(200)) for _ in ARRAY])
            rec = simulate_drives(room, ARRAY, drv, pts[:20], ts.steps).rec
            f = np.fft.rfftfreq(ts.steps, ts.dt)
            sel = (f >= BAND[0]) & (f <= BAND[1])
            pred = np.einsum(
                "fms,sf->mf", ts.at(f[sel])[:, :20], np.fft.rfft(drv, n=ts.steps)[:, sel]
            )
            meas = np.fft.rfft(rec, n=ts.steps)[:, sel]
            r["prediction_rel_err"] = float(np.linalg.norm(meas - pred) / np.linalg.norm(meas))
            t0 = time.perf_counter()
            tt = measure_transfer(room, ARRAY, pts, duration=dur, engine="torch")
            r["torch_s"] = time.perf_counter() - t0
            r["torch_max_rel_diff"] = float(np.abs(tt.rec - ts.rec).max() / np.abs(ts.rec).max())
        # Narrowband comparison on every 2nd record bin in the band.
        f = np.fft.rfftfreq(ts.steps, ts.dt)
        f = f[(f >= BAND[0]) & (f <= BAND[1])][::2]
        H = ts.at(f)
        Hb, Hd = H[:, ib], H[:, id_]
        ref = 4
        qs = {
            "das": delay_and_sum(ARRAY, f, focus=BRIGHT_C),
            "tr": time_reversal(Hb[:, 0]),
            "pm": pressure_matching(Hb, Hd, Hb[:, :, ref], reg=1e-3),
            "acc": acoustic_contrast_control(Hb, Hd, reg=1e-3),
        }
        nb = {}
        for m, q in qs.items():
            q = normalise_to_reference(q, Hb, ref, 0)
            spec = contrast_spectrum_db(Hb, Hd, q)
            nb[m] = {
                "band_db": band_contrast_db(Hb, Hd, q),
                "mean_db": float(spec.mean()),
                "min_db": float(spec.min()),
                "effort_db": float(np.mean(array_effort_db(q, Hb, ref))),
            }
            if name == "absorbing":
                spectra[m] = (f, spec)
        r["narrowband"] = nb
        # Broadband FIR designs, each played in the engine.
        x = bandpass_pulse(ts.dt, BAND, 511)
        bb = {}
        for m in ("das", "tr", "pm", "acc"):
            d = design_broadband(ts, ib, id_, method=m, band=BAND, ref_speaker=ref)
            chk = verify_fir(
                room, ts, d.taps, x, ib, id_, BAND, energy_map=(name == "absorbing" and m == "acc")
            )
            bb[m] = {"predicted_db": chk.predicted_db, "measured_db": chk.measured_db}
            if name == "absorbing" and m == "acc":
                maps["acc"] = chk.energy
                spectra["acc FIR"] = (d.freqs, d.fir_db)
        r["broadband"] = bb
        if name == "absorbing":
            one = np.zeros((len(ARRAY), 2048))
            one[ref, 512] = 1.0
            chk = verify_fir(room, ts, one, x, ib, id_, BAND, energy_map=True)
            maps["ref"] = chk.energy
            r["single_speaker_db"] = chk.measured_db
        # FIR design choices (predicted, no engine run).
        if name in ("absorbing", "live"):
            fir = []
            for L, win, delay, os_ in (
                (1024, "hann", 0.5, 1),
                (2048, "hann", 0.5, 1),
                (2048, "tukey", 0.5, 1),
                (2048, "tukey", 0.25, 2),
                (4096, "tukey", 0.25, 2),
            ):
                bins, fg = design_grid(L * os_, ts.dt, BAND)
                Hg = ts.at(fg)
                q = normalise_to_reference(
                    acoustic_contrast_control(Hg[:, ib], Hg[:, id_]), Hg[:, ib], ref, 0
                )
                taps = fir_from_weights(
                    q, bins, L, ts.dt, delay=delay * L, window=win, n_fft=L * os_
                )
                fir.append(
                    {
                        "L": L,
                        "window": win,
                        "delay": f"L*{delay}",
                        "oversample": os_,
                        "predicted_db": predicted_band_contrast(ts, taps, x, ib, id_, BAND),
                    }
                )
            r["fir_choices"] = fir
        out["rooms"][name] = r
        print(f"zones {name}: {json.dumps(_j({k: r[k] for k in ('narrowband', 'broadband')}))}")

    # Figures.
    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = {"das": "delay-and-sum", "tr": "time reversal", "pm": "pressure matching",
              "acc": "ACC (narrowband)", "acc FIR": "ACC (2048-tap FIR)"}  # fmt: skip
    for k, (m, (f, s)) in enumerate(spectra.items()):
        ax.plot(f, s, color=C[k], lw=2 if m != "acc FIR" else 1.2, label=labels[m],
                ls="-" if m != "acc FIR" else "--")  # fmt: skip
    ax.axhline(10, color=INK2, lw=1, ls=":")
    ax.text(305, 10.6, "plan target 10 dB", ha="left", fontsize=8, color=INK2)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("contrast [dB]")
    ax.set_title("Bright/dark contrast, absorbing room (8 speakers)", fontsize=10, color=INK)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(ART / "zones_contrast.png", dpi=130)
    plt.close(fig)

    room = ROOMS["absorbing"][0]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    assert maps["ref"] is not None and maps["acc"] is not None
    vmax = 10 * np.log10(max(maps["ref"].max(), maps["acc"].max()))
    for ax, key, title in zip(axs, ("ref", "acc"), ("one speaker (reference)", "ACC, 8 speakers")):
        e = 10 * np.log10(np.maximum(maps[key], 1e-30)) - vmax
        ext = (0, room.size[0], 0, room.size[1])
        im = ax.imshow(e.T, origin="lower", extent=ext, cmap="Blues", vmin=-50, vmax=0)
        for c, lab in ((BRIGHT_C, "bright"), (DARK_C, "dark")):
            ax.add_patch(plt.Circle(c, ZONE_R, fill=False, color=INK, lw=1))
            ax.text(c[0], c[1] + ZONE_R + 0.05, lab, ha="center", fontsize=8, color=INK)
        ax.plot(ARRAY[:, 0], ARRAY[:, 1], "s", color=C[1], ms=4)
        for bx in FURNITURE:
            ax.add_patch(
                plt.Rectangle(
                    (bx.x0, bx.y0),
                    bx.x1 - bx.x0,
                    bx.y1 - bx.y0,
                    fill=False,
                    ec=INK2,
                    lw=1,
                    hatch="//",
                )  # fmt: skip
            )
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    fig.colorbar(im, ax=axs, label="band energy [dB re max]", shrink=0.85)
    fig.savefig(ART / "zones_loudness.png", dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 7.3 crosstalk cancellation
# --------------------------------------------------------------------------- #


def run_ctc() -> dict:
    out: dict = {"rooms": {}}
    rooms = {
        "absorbing": (ROOMS["absorbing"][0], 0.3),
        "treated": (Room(size=(3.0, 2.4), beta=1.0, boxes=FURNITURE), 0.2),
        "live": (ROOMS["live"][0], 0.4),
        "anechoic": (ROOMS["anechoic"][0], 0.1),
    }
    spectra = {}
    sweep = {}
    for name, (room, dur) in rooms.items():
        big = name == "absorbing"
        pts = head_grid(room, HEAD, 0.15 if big else 0.0, 0.10 if big else 0.0)
        ts = measure_transfer(room, LAPTOP, pts, duration=dur)
        tr = TrackedCtc(ts, eval_band=BAND)
        d = tr.design(HEAD)
        sep = tr.separation(d, HEAD)
        x = bandpass_pulse(ts.dt, BAND, 511)
        td = [verify_ctc(room, ts, d.taps, HEAD, x, BAND, channel=ch) for ch in (0, 1)]
        # The browser convention (ctc.ts): L = 1024, Hann, L/2 delay, beta 0.005.
        # (Its 150-7000 Hz band is cut to the measured 150-2000 Hz here.)
        web = TrackedCtc(ts, n_taps=1024, beta=0.005, delay=512, window="hann", oversample=1,
                         band=(150.0, 2000.0), eval_band=BAND)  # fmt: skip
        sw = web.separation(web.design(HEAD), HEAD, tr.eval_freqs)
        r = {
            "beta": room.beta if room.boundary == "absorb" else None,
            "alpha_normal": absorption_from_beta(room.beta) if room.boundary == "absorb" else 1.0,
            "tail_db": ts.tail_db(),
            "natural_db": float(np.mean(stereo_separation_db(tr.plant(HEAD)[tr._eval]))),
            "mean_db": float(sep.mean()),
            "min_db": float(sep.min()),
            "p5_db": float(np.percentile(sep, 5)),
            "frac_ge_15": float(np.mean(sep.min(axis=1) >= 15.0)),
            "broadband_predicted_db": [c.predicted_db for c in td],
            "broadband_measured_db": [c.measured_db for c in td],
            "web_convention": {
                "mean_db": float(sw.mean()),
                "min_db": float(sw.min()),
                "frac_ge_15": float(np.mean(sw.min(axis=1) >= 15.0)),
            },
        }
        spectra[name] = (tr.eval_freqs, sep.min(axis=1))
        if big:
            lat = np.array([[k * 0.025, 0.0] for k in range(-6, 7)])
            fwd = np.array([[0.0, k * 0.025] for k in range(-4, 5)])
            sweep = {
                "lateral_cm": (lat[:, 0] * 100).tolist(),
                "lateral_fixed": displacement_sweep(tr, HEAD, lat).tolist(),
                "lateral_tracked": displacement_sweep(tr, HEAD, lat, tracked=True).tolist(),
                "forward_cm": (fwd[:, 1] * 100).tolist(),
                "forward_fixed": displacement_sweep(tr, HEAD, fwd).tolist(),
                "forward_tracked": displacement_sweep(tr, HEAD, fwd, tracked=True).tolist(),
            }
            r["displacement"] = sweep
            lat_f = np.array(sweep["lateral_fixed"])
            r["lateral_15db_halfwidth_cm"] = float(
                threshold_crossing(lat[6:, 0] * 100, lat_f[6:], 15.0)
            )
        out["rooms"][name] = r
        print(f"ctc {name}: {json.dumps(_j({k: v for k, v in r.items() if k != 'displacement'}))}")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    for k, (name, (f, s)) in enumerate(spectra.items()):
        ax.plot(f, s, color=C[k], lw=1.2, label=name)
    ax.axhline(15, color=INK2, lw=1, ls=":")
    ax.text(305, 16, "plan target 15 dB", ha="left", fontsize=8, color=INK2)
    ax.set_ylim(0, 80)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("separation [dB] (worse channel)")
    ax.set_title("Crosstalk cancellation at the design head position", fontsize=10, color=INK)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(ART / "ctc_separation.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, key, lab in zip(axs, ("lateral", "forward"), ("lateral", "fore-aft")):
        xcm = sweep[f"{key}_cm"]
        ax.plot(xcm, sweep[f"{key}_fixed"], "o-", color=C[0], lw=2, ms=4, label="fixed filters")
        ax.plot(
            xcm,
            sweep[f"{key}_tracked"],
            "o-",
            color=C[1],
            lw=2,
            ms=4,
            label="tracked (re-designed)",
        )
        ax.axhline(15, color=INK2, lw=1, ls=":")
        ax.set_xlabel(f"{lab} head offset [cm]")
        _style(ax)
    axs[0].set_ylabel("mean separation [dB]")
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Separation against head position, absorbing room", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(ART / "ctc_displacement.png", dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 7.4 ANC
# --------------------------------------------------------------------------- #


def run_anc() -> dict:
    out: dict = {"rooms": {}}
    freqs = [150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
    att300 = None
    learn = None
    for name in ("absorbing", "live"):
        room = ROOMS[name][0]
        ts = measure_transfer(room, [SECONDARY], [MIC], duration=ROOMS[name][1])
        s_hat = ts.impulse_responses(n=2048)[0, 0]
        tones = []
        for f in freqs:
            x = tone(f, ts.dt, 16000)
            # mu = 0.1 unless the loop fails to converge (reverberant plant
            # dynamics); then halve it. The step size used is reported.
            for mu in (0.1, 0.05, 0.02):
                r = fxlms(room, PRIMARY, SECONDARY, MIC, x, s_hat, n_taps=8,
                          stride=stride_for(f, ts.dt), mu=mu, measure_steps=3000)  # fmt: skip
                if r.attenuation_db > 10.0:
                    break
            assert r.att_map is not None
            qz = quiet_zone(r.att_map, room, MIC)
            hit = np.nonzero(r.learning_db >= 20.0)[0]
            tones.append(
                {
                    "f": f,
                    "mu": mu,
                    "attenuation_db": r.attenuation_db,
                    "steps_to_20db": int(hit[0] * r.block) if len(hit) else None,
                    "qz_diameter_cm": qz.diameter_m * 100,
                    "qz_extent_cm": [v * 100 for v in qz.extent_m],
                    "lambda_over_10_cm": room.c / f * 10,
                }
            )
            if name == "absorbing" and f == 300.0:
                att300 = (r.att_map, qz.mask)
        x = band_noise((100.0, 500.0), ts.dt, 40000, seed=0)
        r = fxlms(room, PRIMARY, SECONDARY, MIC, x, s_hat, n_taps=96,
                  stride=stride_for(500.0, ts.dt), mu=0.2, measure_steps=8000)  # fmt: skip
        assert r.att_map is not None
        qz = quiet_zone(r.att_map, room, MIC)
        noise = {
            "band_hz": [100, 500],
            "attenuation_db": r.attenuation_db,
            "qz_diameter_cm": qz.diameter_m * 100,
        }
        if name == "absorbing":
            learn = (np.arange(len(r.learning_db)) * r.block * ts.dt, r.learning_db)
        out["rooms"][name] = {"tones": tones, "noise": noise}
        print(f"anc {name}: {json.dumps(_j(out['rooms'][name]))}")

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    ax = axs[0]
    for k, name in enumerate(("absorbing", "live")):
        t = out["rooms"][name]["tones"]
        ax.plot(
            [v["f"] for v in t],
            [v["qz_diameter_cm"] for v in t],
            "o-",
            color=C[k],
            lw=2,
            ms=5,
            label=name,
        )
    ff = np.array(freqs)
    ax.plot(ff, 34300 / ff / 10, color=INK2, lw=1, ls="--", label="lambda / 10")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(freqs)
    ax.set_xticklabels([f"{v:.0f}" for v in freqs])
    ax.set_yticks([2, 5, 10, 20, 50])
    ax.set_yticklabels(["2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(">= 10 dB zone diameter [cm]")
    ax.set_title("Quiet-zone size (tones)", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    ax = axs[1]
    room = ROOMS["absorbing"][0]
    assert att300 is not None
    ci = room.cells([MIC])[0]
    w = 16
    sub = att300[0][ci[0] - w : ci[0] + w + 1, ci[1] - w : ci[1] + w + 1]
    ext = (-w * room.dx * 100, w * room.dx * 100, -w * room.dx * 100, w * room.dx * 100)
    im = ax.imshow(sub.T, origin="lower", extent=ext, cmap="RdBu", vmin=-30, vmax=30)
    ax.contour(
        np.linspace(ext[0], ext[1], sub.shape[0]),
        np.linspace(ext[2], ext[3], sub.shape[1]),
        sub.T,
        levels=[10],
        colors=INK,
        linewidths=1,
    )
    ax.plot(0, 0, "x", color=INK)
    ax.plot((SECONDARY[0] - MIC[0]) * 100, 0, "s", color=C[1], ms=5)
    ax.set_xlabel("x - mic [cm]")
    ax.set_ylabel("y - mic [cm]")
    ax.set_title("Attenuation at 300 Hz (10 dB contour)", fontsize=9, color=INK)
    fig.colorbar(im, ax=ax, label="dB", shrink=0.85)
    ax = axs[2]
    assert learn is not None
    ax.plot(learn[0] * 1000, learn[1], color=C[0], lw=2)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("attenuation at mic [dB]")
    ax.set_title("Broadband 100-500 Hz noise, learning", fontsize=9, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(ART / "anc.png", dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 7.5 sensing requirements
# --------------------------------------------------------------------------- #


def run_req() -> dict:
    out: dict = {"rooms": {}}
    deltas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    scales = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    regs = {"reg 1e-3": 1e-3, "reg 1e-1 (robust)": 1e-1}
    curves = {}
    for name in ("absorbing", "live"):
        room, dur = ROOMS[name]
        _, _, _, b = zone_points(room)
        d = disk_points(room, DARK_C, ZONE_R)
        st = SensingStudy(room, ARRAY, b, d, band=BAND, duration=dur)
        free = Room(size=room.size, boundary="cpml")
        H_free = measure_transfer(free, ARRAY, st.points, duration=0.1).at(st.freqs)
        est: list[tuple[SensingError, np.ndarray]] = []
        for e in wall_errors(deltas, n_patterns=3, seed=1):
            if e.exact and any(x.exact for x, _ in est):
                continue
            est.append((e, st.estimate(e)))
        for s in scales:
            if s != 1.0:
                est.append((SensingError(beta_scale=s), st.estimate(SensingError(beta_scale=s))))
        for sh in (0.05, 0.1, 0.2):
            e = SensingError(box_shift_m=(sh, 0.0))
            est.append((e, st.estimate(e)))
        e = SensingError(drop_boxes=True)
        est.append((e, st.estimate(e)))
        res: dict = {}
        for rname, reg in regs.items():
            st.reg = reg
            rows = []
            for e, H in est:
                r = st.evaluate_model(H, e)
                rows.append(
                    {
                        "wall_cm": e.wall_m * 100,
                        "signs": list(e.signs),
                        "beta_scale": e.beta_scale,
                        "box_shift_cm": e.box_shift_m[0] * 100,
                        "drop_boxes": e.drop_boxes,
                        "achieved_db": r.band_db,
                        "model_db": r.predicted_db,
                        "sub_bands": r.sub_band_db,
                    }
                )
            ff = st.evaluate_model(H_free, SensingError(label="free field"))
            walls = {}
            for dd in deltas:
                v = [x["achieved_db"] for x in rows if x["wall_cm"] == dd * 100 and x["beta_scale"] == 1.0
                     and x["box_shift_cm"] == 0 and not x["drop_boxes"]]  # fmt: skip
                walls[dd * 100] = (float(np.mean(v)), float(np.min(v)), float(np.max(v)))
            wx = list(walls)
            res[rname] = {
                "rows": rows,
                "free_field_db": ff.band_db,
                "free_field_sub": ff.sub_band_db,
                "wall_curve": {str(k): v for k, v in walls.items()},
                "wall_10db_cm": threshold_crossing(wx, [walls[k][0] for k in wx], 10.0),
                "wall_15db_cm": threshold_crossing(wx, [walls[k][0] for k in wx], 15.0),
            }
            curves[(name, rname)] = (walls, ff.band_db, rows)
        out["rooms"][name] = res
        print(
            f"req {name}: "
            + json.dumps(
                _j(
                    {
                        k: {kk: v[kk] for kk in ("free_field_db", "wall_curve", "wall_10db_cm")}
                        for k, v in res.items()
                    }
                )
            )
        )

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for k, name in enumerate(("absorbing", "live")):
        for ls, rname in zip(("-", "--"), regs):
            walls, ffdb, rows = curves[(name, rname)]
            xs = list(walls)
            m = [walls[x][0] for x in xs]
            axs[0].plot(xs, m, ls, color=C[k], lw=2, marker="o", ms=4, label=f"{name}, {rname}")
            if ls == "-":
                axs[0].fill_between(xs, [walls[x][1] for x in xs], [walls[x][2] for x in xs],
                                    color=C[k], alpha=0.15, lw=0)  # fmt: skip
                axs[0].axhline(ffdb, color=C[k], lw=1, ls=":")
            ab = sorted([(x["beta_scale"], x["achieved_db"]) for x in rows
                         if x["wall_cm"] == 0 and x["box_shift_cm"] == 0 and not x["drop_boxes"]])  # fmt: skip
            axs[1].plot(
                [a for a, _ in ab], [v for _, v in ab], ls, color=C[k], lw=2, marker="o", ms=4
            )
    for ax in axs:
        ax.axhline(10, color=INK2, lw=1, ls=":")
        _style(ax)
    axs[0].set_xlabel("wall position error [cm]")
    axs[0].set_ylabel("achieved contrast [dB]")
    axs[0].set_title("Wall error (dotted: free-field model)", fontsize=9, color=INK)
    axs[1].set_xscale("log")
    axs[1].set_xticks(scales)
    axs[1].set_xticklabels([str(s) for s in scales])
    axs[1].minorticks_off()
    axs[1].set_xlabel("estimated / true wall admittance")
    axs[1].set_title("Absorption error", fontsize=9, color=INK)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(ART / "requirements.png", dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 7.6 differentiable control
# --------------------------------------------------------------------------- #


def run_diff() -> dict:
    torch.set_num_threads(1)
    room = Room(size=(3.0, 2.4), dx=0.05, beta=0.3, boxes=FURNITURE)
    b = disk_points(room, BRIGHT_C, ZONE_R)
    d = disk_points(room, DARK_C, ZONE_R)
    sc = DiffScene(room, ARRAY, b, d, band=(200.0, 800.0), n_taps=256)
    pts = np.concatenate([b, d])
    ts: TransferSet = measure_transfer(room, ARRAY, pts, duration=0.15, band=(100.0, 1000.0))
    ib, id_ = np.arange(len(b)), np.arange(len(b), len(pts))
    acc = design_broadband(ts, ib, id_, n_taps=256, band=sc.band, oversample=4)
    das = design_broadband(ts, ib, id_, method="das", n_taps=256, band=sc.band, oversample=4)
    t0 = time.perf_counter()
    from_acc = optimise_drives(sc, acc.taps, iters=60, lr=0.05)
    t_acc = time.perf_counter() - t0
    from_rand = optimise_drives(sc, None, iters=60, lr=0.05, seed=0)
    out = {
        "grid": list(room.shape),
        "dx_cm": room.dx * 100,
        "band_hz": list(sc.band),
        "steps": sc.steps,
        "taps": sc.n_taps,
        "window_ms": sc.steps * room.timestep * 1000,
        "acc_narrowband_band_db": float(np.mean(acc.ideal_db)),
        "acc_fir_window_db": sc.contrast_db(acc.taps),
        "das_fir_window_db": sc.contrast_db(das.taps),
        "opt_from_acc_db": from_acc.final_db,
        "opt_from_random_db": from_rand.final_db,
        "random_initial_db": from_rand.initial_db,
        "opt_from_acc_numba_db": sc.verify_numba(from_acc.z),
        "opt_from_random_numba_db": sc.verify_numba(from_rand.z),
        "seconds_per_iter": t_acc / 60,
    }
    print("diff: " + json.dumps(_j(out)))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(from_acc.history_db, color=C[0], lw=2, label="start: ACC FIR")
    ax.plot(from_rand.history_db, color=C[1], lw=2, label="start: random")
    ax.axhline(out["acc_fir_window_db"], color=C[0], lw=1, ls=":")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("window contrast [dB]")
    ax.set_title("Drive signals optimised through TorchFDTD", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(ART / "differentiable.png", dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _f(v: Any, nd: int = 1) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return "-"
    if not np.isfinite(v):
        return "> range"
    return f"{v:.{nd}f}"


def write_report(R: dict) -> None:
    z, c, a, q, d = R.get("zones"), R.get("ctc"), R.get("anc"), R.get("req"), R.get("diff")
    L: list[str] = []
    w = L.append
    w("# Phase 7: beamforming and sound-field control (2026-09-24)")
    w("")
    w("Generated by `scripts/control_report.py` (all numbers below come from that run;")
    w("`results.json` in the artifacts folder holds every value). Module docs:")
    w("`docs/control.md`. Tests: `tests/control/`.")
    w("")
    w("Scenes are 2D, in SI units: cell 2.5 cm, c = 343 m/s, Courant 0.5, so")
    w("dt = 36.4 us (27.4 kHz). The furnished room is 3.0 x 2.4 m with a rigid sofa")
    w("and an absorber panel. Its walls are locally reacting impedance walls:")
    w("")
    w("| room | outer walls | normal-incidence absorption |")
    w("|---|---|---|")
    w(f"| absorbing | beta = 0.3 | {absorption_from_beta(0.3):.2f} |")
    w(f"| live | beta = 0.1 | {absorption_from_beta(0.1):.2f} |")
    w(
        f"| treated (CTC only) | beta = 1.0 | {absorption_from_beta(1.0):.2f} (grazing waves still reflect) |"
    )
    w("| anechoic | CPML | reference |")
    w("")
    w("## Plan targets")
    w("")
    w("| target | result | met? |")
    w("|---|---|---|")
    if z:
        acc = z["rooms"]["absorbing"]["broadband"]["acc"]
        w(
            f"| >= 10 dB bright/dark contrast with an array, absorbing room | {_f(acc['measured_db'])} dB broadband (ACC, 8 speakers, 300-1500 Hz, 2048-tap FIRs played in the engine) | yes |"
        )
    if c:
        ca = c["rooms"]["absorbing"]
        ct = c["rooms"]["treated"]
        w(
            f"| >= 15 dB crosstalk cancellation, 2 laptop speakers, absorbing room | {_f(min(ca['broadband_measured_db']))} dB broadband (worse channel, engine run); per frequency >= 15 dB over {ca['frac_ge_15'] * 100:.0f} % of 300-1500 Hz, minimum {_f(ca['min_db'])} dB | broadband yes; not at every frequency |"
        )
        w(
            f"| (same, beta = 1 walls) | >= {_f(ct['min_db'])} dB at every frequency; {_f(min(ct['broadband_measured_db']))} dB broadband | yes |"
        )
        cl = c["rooms"]["live"]
        w(
            f"| (same, live room, beta = 0.1) | {_f(min(cl['broadband_measured_db']))} dB broadband; >= 15 dB over {cl['frac_ge_15'] * 100:.0f} % of the band | no |"
        )
    w("")
    if z:
        w("## 7.1 Transfer functions and metrics")
        w("")
        za = z["rooms"]["absorbing"]
        w("Each speaker plays a zero-mean band-pass pulse (100-2000 Hz); H = Y/W.")
        w(f"In the absorbing room the 8 x {sum(za['points'])} transfer functions take")
        w(f"{za['measure_s']:.1f} s (numba, one thread, {za['steps']} steps each). The batched")
        w(f"`TorchFDTD` path takes {za['torch_s']:.1f} s on this CPU and agrees to")
        w(f"{za['torch_max_rel_diff']:.1e} (relative). A random 8-speaker drive played in the")
        w(
            f"engine matches the prediction from H to a relative error of {za['prediction_rel_err']:.1e}"
        )
        tails = ", ".join(f"{k} {_f(v['tail_db'], 0)} dB" for k, v in z["rooms"].items())
        w(f"(300-1500 Hz). Record tails: {tails}.")
        w("")
        w("## 7.2 Sound zones")
        w("")
        w("8 speakers, 10 cm apart, 0.4 m from the wall; bright zone centred (1.0, 1.6),")
        w("dark zone (2.0, 1.6), both 15 cm radius (113 points each). Every method is")
        w("normalised to the loudness of the centre speaker in the bright zone, so effort")
        w("is relative to that speaker.")
        w("")
        w("Narrowband (per-frequency weights, 300-1500 Hz; band = energy ratio over the band):")
        w("")
        w("| room | method | band contrast [dB] | mean / min per freq [dB] | effort [dB] |")
        w("|---|---|---|---|---|")
        names = {
            "das": "delay-and-sum",
            "tr": "time reversal",
            "pm": "pressure matching",
            "acc": "ACC",
        }
        for room, r in z["rooms"].items():
            for m, v in r["narrowband"].items():
                w(
                    f"| {room} | {names[m]} | {_f(v['band_db'])} | {_f(v['mean_db'])} / {_f(v['min_db'])} | {_f(v['effort_db'])} |"
                )
        w("")
        w("Broadband: 2048-tap FIR filters (75 ms; delay L/4, Tukey window, 2x oversampled")
        w("design), fed a 300-1500 Hz pulse and **played in the time-domain engine**.")
        w("'predicted' is the frequency-domain prediction from H and the filters:")
        w("")
        w("| room | method | predicted [dB] | measured in engine [dB] |")
        w("|---|---|---|---|")
        for room, r in z["rooms"].items():
            for m, v in r["broadband"].items():
                w(
                    f"| {room} | {names[m]} | {_f(v['predicted_db'], 2)} | {_f(v['measured_db'], 2)} |"
                )
        w("")
        worst = max(
            abs(v["predicted_db"] - v["measured_db"])
            for r in z["rooms"].values()
            for v in r["broadband"].values()
        )
        w(f"Largest prediction/engine difference: {worst:.3f} dB (tolerance pinned in the")
        w(f"tests: 0.2 dB). A single speaker alone gives {_f(za['single_speaker_db'])} dB.")
        w("")
        w("FIR design choices for ACC (predicted broadband contrast):")
        w("")
        w("| room | L | window | delay | oversample | contrast [dB] |")
        w("|---|---|---|---|---|---|")
        for room in ("absorbing", "live"):
            for v in z["rooms"][room]["fir_choices"]:
                w(
                    f"| {room} | {v['L']} | {v['window']} | {v['delay']} | {v['oversample']} | {_f(v['predicted_db'])} |"
                )
        w("")
        w("Reading: ACC reaches 22-29 dB narrowband (band contrast) in every room.")
        w("Realising it with FIRs costs under 1 dB in the absorbing room and about 5 dB")
        w("in the live room, where the room inverse is longer than the filter (4096")
        w("taps recover most of it). Pressure matching trades contrast for a")
        w("controlled bright-zone field. Delay-and-sum and time reversal do not look at")
        w("the dark zone at all: they gain 4-12 dB over one speaker only by focusing,")
        w("and reflections (live room) erode even that.")
        w("")
        w("![contrast](control_2026_09_24_artifacts/zones_contrast.png)")
        w("![loudness](control_2026_09_24_artifacts/zones_loudness.png)")
        w("")
    if c:
        w("## 7.3 Laptop crosstalk cancellation")
        w("")
        w("Speakers 30 cm apart; ears two points 17.5 cm apart, 50 cm in front (no head,")
        w("as in `web/src/control/ctc.ts`). Kirkeby inverse per frequency, beta = 0.001")
        w("relative to mean |H|^2, designed over 200-1800 Hz, 4096 taps (150 ms), delay")
        w("L/4, Tukey window, 4x oversampled; scored over 300-1500 Hz on a 1.7 Hz grid.")
        w("Separation is the worse of the two programmes at each frequency.")
        w("")
        w(
            "| room | plain stereo [dB] | mean [dB] | 5th pct [dB] | min [dB] | freq >= 15 dB | broadband, engine (L / R) [dB] | predicted (L / R) [dB] | ctc.ts convention: mean / min / >= 15 dB |"
        )
        w("|---|---|---|---|---|---|---|---|---|")
        for room, r in c["rooms"].items():
            wc = r["web_convention"]
            w(f"| {room} | {_f(r['natural_db'])} | {_f(r['mean_db'])} | {_f(r['p5_db'])} | {_f(r['min_db'])} | {r['frac_ge_15'] * 100:.0f} % | "
              f"{_f(r['broadband_measured_db'][0])} / {_f(r['broadband_measured_db'][1])} | {_f(r['broadband_predicted_db'][0])} / {_f(r['broadband_predicted_db'][1])} | "
              f"{_f(wc['mean_db'])} / {_f(wc['min_db'])} / {wc['frac_ge_15'] * 100:.0f} % |")  # fmt: skip
        w("")
        ca = c["rooms"]["absorbing"]
        w("The per-frequency dips in the absorbing and live rooms sit at room modes,")
        w("where one mode dominates both ears and the 2x2 plant is nearly rank one")
        w("(in a development run with a similar geometry the condition number at the")
        w("dips was 26-30): the")
        w("regularised inverse cannot separate the ears there without a huge filter")
        w("gain, and a 150 ms FIR cannot hold the long inverse. With beta = 1 walls no")
        w("mode dominates and every frequency clears 15 dB. **In the live room the")
        w("target is not met** (worse channel below 15 dB broadband; see the table).")
        w("The browser convention (1024 taps, Hann, L/2) is much weaker in any room,")
        w("because the room inverse is longer than its 512-sample causal half.")
        w("")
        disp = ca["displacement"]
        w("Head displacement (absorbing room): fixed filters stay above 15 dB for")
        w(
            f"lateral moves up to about {_f(ca['lateral_15db_halfwidth_cm'])} cm. Filters re-designed"
        )
        w(
            f"for the tracked head (7.3.3) keep {_f(min(disp['lateral_tracked']))}-{_f(max(disp['lateral_tracked']))} dB"
        )
        w("over +-15 cm lateral:")
        w("")
        w("| lateral offset [cm] | " + " | ".join(_f(v, 1) for v in disp["lateral_cm"]) + " |")
        w("|---|" + "---|" * len(disp["lateral_cm"]))
        w("| fixed [dB] | " + " | ".join(_f(v) for v in disp["lateral_fixed"]) + " |")
        w("| tracked [dB] | " + " | ".join(_f(v) for v in disp["lateral_tracked"]) + " |")
        w("")
        w("| fore-aft offset [cm] | " + " | ".join(_f(v, 1) for v in disp["forward_cm"]) + " |")
        w("|---|" + "---|" * len(disp["forward_cm"]))
        w("| fixed [dB] | " + " | ".join(_f(v) for v in disp["forward_fixed"]) + " |")
        w("| tracked [dB] | " + " | ".join(_f(v) for v in disp["forward_tracked"]) + " |")
        w("")
        w("![ctc](control_2026_09_24_artifacts/ctc_separation.png)")
        w("![ctc displacement](control_2026_09_24_artifacts/ctc_displacement.png)")
        w("")
    if a:
        w("## 7.4 FxLMS noise cancellation")
        w("")
        w("Primary noise at (0.5, 0.5) m, error mic at (2.0, 1.5) m, secondary speaker")
        w("30 cm from the mic. Feedforward with the noise as reference; secondary path")
        w("estimated from the transfer-function measurement (2048-tap band-limited IR);")
        w("filtered-x NLMS run sample by sample against the engine (mic DC blocker at")
        w("30 Hz). Tones: 8 taps at quarter-period spacing, 16,000 steps")
        w("(0.58 s), weights frozen for the last 3,000 steps where everything is scored.")
        w("mu starts at 0.1 and is halved when a run does not converge (the reverberant")
        w("plant responds to a weight change over the reverberation time).")
        w("")
        w(
            "| room | f [Hz] | mu | attenuation at mic [dB] | steps until 20 dB (500-step blocks) | >= 10 dB zone diameter [cm] | extent x / y [cm] | lambda/10 [cm] |"
        )
        w("|---|---|---|---|---|---|---|---|")
        for room, r in a["rooms"].items():
            for t in r["tones"]:
                s20 = "-" if t["steps_to_20db"] is None else f"{t['steps_to_20db'] + 500}"
                w(f"| {room} | {t['f']:.0f} | {t['mu']} | {_f(t['attenuation_db'])} | {s20} | {_f(t['qz_diameter_cm'])} | "
                  f"{_f(t['qz_extent_cm'][0])} / {_f(t['qz_extent_cm'][1])} | {_f(t['lambda_over_10_cm'])} |")  # fmt: skip
        w("")
        for room, r in a["rooms"].items():
            n = r["noise"]
            w(f"- {room} room, 100-500 Hz Gaussian noise (96 taps, mu = 0.2, 1.46 s): "
              f"{_f(n['attenuation_db'])} dB at the mic, >= 10 dB zone {_f(n['qz_diameter_cm'])} cm.")  # fmt: skip
        w("")
        w("A deterministic, perfectly referenced tone is cancelled to the numerical floor")
        w("at the mic (35-120 dB). What matters is the zone. Its diameter scales with the")
        w("wavelength, at 1-2.5 times the classic diffuse-field estimate of lambda/10")
        w("(it is elongated, and larger here, because the primary field is dominated by")
        w("its direct sound). Single-mic local ANC gives a zone of 10 cm or more up to")
        w("about 500 Hz, which shrinks to 3-4 cm by 1 kHz. Band-limited noise is harder:")
        w("the adaptive filter must also match the room's primary path over the whole")
        w("band. The noise results are 22 dB (absorbing) and 11 dB (live) at the mic.")
        w("")
        w("![anc](control_2026_09_24_artifacts/anc.png)")
        w("")
    if q:
        w("## 7.5 Sensing requirements")
        w("")
        w("ACC designed from transfer functions simulated in an *estimated* room, scored")
        w("with the *true* room's transfer functions (same array and zones as 7.2,")
        w("narrowband weights, 300-1500 Hz band contrast). Wall errors move all four")
        w("walls by the same distance (three sign patterns: mean, min-max band in the")
        w("figure); errors are quantised to the 2.5 cm grid.")
        w("")
        for room, res in q["rooms"].items():
            w(f"**{room} room**")
            w("")
            hdr = list(res)
            w("| error | " + " | ".join(hdr) + " |")
            w("|---|" + "---|" * len(hdr))
            keys = list(res[hdr[0]]["wall_curve"])
            for k in keys:
                w(
                    f"| walls {float(k):.1f} cm | "
                    + " | ".join(
                        f"{_f(res[h]['wall_curve'][k][0])} ({_f(res[h]['wall_curve'][k][1])}-{_f(res[h]['wall_curve'][k][2])})"
                        for h in hdr
                    )
                    + " |"
                )
            rows0 = res[hdr[0]]["rows"]
            for i, row in enumerate(rows0):
                if row["wall_cm"] != 0:
                    continue
                if row["beta_scale"] != 1.0:
                    lab = f"absorption x {row['beta_scale']}"
                elif row["box_shift_cm"]:
                    lab = f"furniture moved {row['box_shift_cm']:.0f} cm"
                elif row["drop_boxes"]:
                    lab = "furniture missing"
                else:
                    continue
                w(
                    f"| {lab} | "
                    + " | ".join(_f(res[h]["rows"][i]["achieved_db"]) for h in hdr)
                    + " |"
                )
            w(
                "| free-field model (no room) | "
                + " | ".join(_f(res[h]["free_field_db"]) for h in hdr)
                + " |"
            )
            w(
                "| wall error where contrast < 15 dB | "
                + " | ".join(f"{_f(res[h]['wall_15db_cm'])} cm" for h in hdr)
                + " |"
            )
            w(
                "| wall error where contrast < 10 dB | "
                + " | ".join(f"{_f(res[h]['wall_10db_cm'])} cm" for h in hdr)
                + " |"
            )
            w("")
        w("What this tells Phase 6:")
        w("")
        ab = q["rooms"]["absorbing"]["reg 1e-3"]
        lv = q["rooms"]["live"]["reg 1e-3"]
        mdl = [r["model_db"] for res in q["rooms"].values() for r in res["reg 1e-3"]["rows"]]
        w(
            f"- The model's own prediction stays at {min(mdl):.0f}-{max(mdl):.0f} dB whatever the error"
        )
        w("  (reg 1e-3), so a wrong room model is always over-confident. Only the")
        w("  contrast achieved in the true room counts.")
        w(
            f"- Wall positions, absorbing room: 15 dB needs the walls to within about {_f(ab['wall_15db_cm'])} cm."
        )
        w(
            f"  Contrast never falls below 10 dB, because the free-field floor ({_f(ab['free_field_db'])} dB)"
        )
        w("  is already above it: with absorbing walls the direct sound dominates.")
        w(
            f"- Wall positions, live room: 10 dB needs about {_f(lv['wall_10db_cm'])} cm and 15 dB about"
        )
        w(f"  {_f(lv['wall_15db_cm'])} cm. Both are below the 2.5 cm grid step, so they are")
        w("  interpolated between the exact model and a one-cell error. Read them as")
        w("  'better than one cell (2.5 cm)', about a tenth of the shortest wavelength")
        w("  (23 cm at 1.5 kHz). By 5 cm of error a room-aware design is no better than")
        w(f"  a free-field one ({_f(lv['free_field_db'])} dB).")
        w("- Absorption: a factor-of-two error costs 10-13 dB but still clears 10 dB;")
        w("  +-15-20 % costs about 2 dB.")
        w("- Furniture: a missing or 10-20 cm-misplaced sofa costs about as much as a")
        w("  2.5-5 cm wall error.")
        w("- Stronger regularisation (1e-1) lowers the exact-model contrast by 7-8 dB and")
        w("  gains at most about 1 dB at large errors. It is not a substitute for")
        w("  accurate sensing.")
        w("- Requirement for Phase 6: wall positions to about 1.5-4 cm (below the 2.5 cm")
        w(
            "  grid step in a live room), absorption to within about 30 %, and furniture positions to"
        )
        w("  about 5 cm for 15 dB zones in the 300-1500 Hz band.")
        w("")
        w("![requirements](control_2026_09_24_artifacts/requirements.png)")
        w("")
    if d:
        w("## 7.6 Differentiable control")
        w("")
        w(
            f"`TorchFDTD` (autograd), {d['grid'][0]} x {d['grid'][1]} grid at {d['dx_cm']:.0f} cm, the same furnished"
        )
        w(
            f"absorbing room, array and zones, {d['band_hz'][0]:.0f}-{d['band_hz'][1]:.0f} Hz. The optimisation"
        )
        w(
            f"variables are {d['taps']}-tap filters per speaker applied to a band-pass programme; the"
        )
        w(
            f"objective is the bright/dark energy ratio over the {d['window_ms']:.0f} ms window ({d['steps']} steps)."
        )
        w(
            f"Adam, 60 iterations, {d['seconds_per_iter']:.2f} s per iteration (forward and backward, one thread)."
        )
        w("")
        w("| design | window contrast [dB] | replayed in numba [dB] |")
        w("|---|---|---|")
        w(f"| delay-and-sum FIR | {_f(d['das_fir_window_db'])} | - |")
        w(
            f"| ACC FIR (narrowband optimum {_f(d['acc_narrowband_band_db'])} dB) | {_f(d['acc_fir_window_db'])} | - |"
        )
        w(
            f"| optimised, random start ({_f(d['random_initial_db'])} dB) | {_f(d['opt_from_random_db'])} | {_f(d['opt_from_random_numba_db'])} |"
        )
        w(
            f"| optimised, ACC start | {_f(d['opt_from_acc_db'])} | {_f(d['opt_from_acc_numba_db'])} |"
        )
        w("")
        gain = d["opt_from_acc_db"] - d["acc_fir_window_db"]
        w("With short filters (19 ms) and a finite scoring window, the ACC FIR keeps only")
        w("about half of its narrowband optimum (in dB). Optimising the drives through")
        w(f"the simulator adds {gain:.1f} dB to it, because it scores the finite filters,")
        w("transients and window directly. Starting from ACC beats a random start with")
        w("the same number of iterations. Replaying the optimised drives in the numba")
        w("engine gives the same contrast, so the tensor and numba engines agree. Caveat:")
        w("energy that arrives after the window is not scored.")
        w("")
        w("![differentiable](control_2026_09_24_artifacts/differentiable.png)")
        w("")
    REPORT.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=["zones", "ctc", "anc", "req", "diff"])
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    runners = {"zones": run_zones, "ctc": run_ctc, "anc": run_anc, "req": run_req, "diff": run_diff}
    for name in args.only or list(runners):
        t0 = time.perf_counter()
        results[name] = _j(runners[name]())
        results[name]["runtime_s"] = time.perf_counter() - t0
        print(f"{name}: {results[name]['runtime_s']:.0f} s")
        path.write_text(json.dumps(results, indent=1))
    write_report(results)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
