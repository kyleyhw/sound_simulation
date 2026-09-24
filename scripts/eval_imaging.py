"""Score the no-ML physics imagers against the no-audio prior (plan Task 6.2).

Protocol (nothing is tuned on held-out rooms):

1. **Priors.** The no-audio baseline for the filled-mask target is the
   per-pixel training prior :math:`\\hat\\pi(x)` over training rooms
   ``[n_fit, 10000)``; for the illuminated-boundary target it is the mean
   illuminated-boundary mask over ``n_prior_ill`` of those rooms (with their
   own poses).
2. **Fit rooms.** Training rooms ``[0, n_fit)`` (K = 4 poses each) are
   imaged. For every method the smoothing width, and for carving the onset
   threshold, are chosen by logistic log-loss on these rooms. The fusion
   :math:`\\operatorname{logit} q = \\operatorname{logit}\\hat\\pi + \\sum_i a_i
   z_i + b`, with :math:`z_i` the per-room standardised image, is fitted by
   logistic regression on these rooms, and the IoU threshold :math:`\\tau`
   maximises their mean IoU.
3. **Held-out rooms.** The first ``n_heldout`` rooms of the held-out
   archive are imaged with their first K = 4 poses (the training pose
   count; K = 8 is reported as a secondary run with the K = 4 fit). Per-room
   IoU at the training :math:`\\tau`, average precision, boundary F (1-cell
   tolerance) and the information gain over the prior are computed, and
   every method is compared to the prior by the paired per-room difference
   (mean, SE, :math:`z = \\bar\\Delta/\\mathrm{SE}`).
4. **FWI** (proof of concept) runs on ``n_fwi_train`` training rooms (for
   its threshold and fusion) and ``n_fwi_heldout`` held-out rooms.
5. **Noise.** The fused methods are re-fitted and re-scored with white noise
   added to every recording (``--noise-db``), a robustness check the
   noise-free archives cannot provide.

Also writes the CRLB design chart (6.4) and the room-parameter validation
(6.5). Results go to ``<out-dir>/results.json`` and figures to
``<out-dir>/*.png``.

Usage::

    NUMBA_NUM_THREADS=1 uv run python scripts/eval_imaging.py \\
        --out-dir tests/reports/imaging_2026_09_24_artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time

import h5py
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from acoustic_system.imaging import crlb, room_params  # noqa: E402
from acoustic_system.imaging.image_source import carve_free_space  # noqa: E402
from acoustic_system.imaging.ir import TikhonovDeconvolver, envelope, matched_filter  # noqa: E402
from acoustic_system.imaging.pipeline import (  # noqa: E402
    ArchiveRoom,
    compute_room_images,
    device_mask,
)
from acoustic_system.imaging.targets import illuminated_boundary  # noqa: E402
from acoustic_system.learning.metrics import (  # noqa: E402
    average_precision,
    boundary_f,
    info_gain_bits,
    iou,
    mean_se,
)

DATA = _REPO_ROOT / "data" / "training_data"
IMAGE_KEYS = ("backprojection", "ellipses", "carving", "time_reversal")
SIGMAS = (0.0, 1.0, 2.0, 3.0)
CARVE_VARIANTS = tuple((k, th) for k in ("raw", "mf") for th in (1e-3, 1e-2, 0.05, 0.2, 0.4))
EPS_P = 1e-4


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract(
    path: pathlib.Path,
    rooms: range,
    n_poses: int,
    noise_db: float | None,
    cache: pathlib.Path,
) -> dict[str, np.ndarray]:
    """Images, residuals and targets for ``rooms`` (cached as ``.npz``)."""
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    out: dict[str, list] = {
        k: []
        for k in IMAGE_KEYS + ("residual", "mask", "illum", "devices", "sources", "mics", "drive")
    }
    rng = np.random.default_rng(12345)
    dec = None
    t0 = time.perf_counter()
    with h5py.File(path, "r") as hf:
        for n, r in enumerate(rooms):
            room = ArchiveRoom.from_group(hf[f"sample_{r:04d}"], n_poses)
            if dec is None:
                dec = TikhonovDeconvolver(room.drive, room.recordings.shape[-1], lam=1e-2)
            im = compute_room_images(room, dec, noise_db=noise_db, rng=rng)
            for k in IMAGE_KEYS:
                out[k].append(getattr(im, k).astype(np.float32))
            out["residual"].append(im.residual.astype(np.float32))
            out["mask"].append(room.mask)
            out["illum"].append(illuminated_boundary(room.mask, room.sources, room.mics))
            out["devices"].append(device_mask(room))
            out["sources"].append(room.sources)
            out["mics"].append(room.mics)
            out["drive"].append(room.drive)
            if (n + 1) % 50 == 0:
                print(
                    f"  {path.name} K={n_poses} noise={noise_db}: {n + 1}/{len(rooms)} "
                    f"({time.perf_counter() - t0:.0f} s)",
                    flush=True,
                )
    arr = {k: np.stack(v) for k, v in out.items()}
    np.savez_compressed(cache, allow_pickle=False, **arr)
    return arr


def carving_images(
    f: dict[str, np.ndarray], threshold: float, kind: str = "raw", dt: float = 0.5
) -> np.ndarray:
    """Carving counts recomputed from the stored residuals.

    ``kind="raw"`` detects the onset on the raw scattered residual (sharp in
    noise-free data); ``kind="mf"`` on the envelope of its matched-filter
    output, which gains the chirp's time-bandwidth product against noise.
    """
    grid = f["mask"].shape[1:]
    res = f["residual"].astype(np.float64)
    if kind == "mf":
        res = envelope(matched_filter(res, f["drive"][0]))
    return np.stack(
        [
            carve_free_space(grid, s, m, r, dt, rel_threshold=threshold)
            for s, m, r in zip(f["sources"], f["mics"], res)
        ]
    )


# ---------------------------------------------------------------------------
# Fusion and scoring
# ---------------------------------------------------------------------------


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS_P, 1 - EPS_P)
    return np.log(p / (1 - p))


def zscore(x: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Per-room smoothing (Gaussian ``sigma`` cells) then standardisation."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    if sigma > 0:
        x = ndimage.gaussian_filter(x, (0, sigma, sigma))
    flat = x.reshape(len(x), -1)
    flat = (flat - flat.mean(1, keepdims=True)) / (flat.std(1, keepdims=True) + 1e-12)
    return flat.reshape(x.shape)


def fit_logistic(
    offset: np.ndarray, feats: list[np.ndarray], y: np.ndarray
) -> tuple[np.ndarray, float]:
    """Weights ``(a_1..a_n, b)`` of logit q = offset + sum a_i z_i + b, and the mean log-loss."""
    X = np.stack([f.ravel() for f in feats], 1) if feats else np.zeros((offset.size, 0))
    o = offset.ravel()
    t = y.ravel().astype(np.float64)

    def nll(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = o + X @ w[:-1] + w[-1]
        q = 1.0 / (1.0 + np.exp(-z))
        loss = float(np.mean(np.logaddexp(0.0, z) - t * z))
        g = q - t
        return loss, np.concatenate([X.T @ g, [g.sum()]]) / t.size

    res = minimize(nll, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B")
    return res.x, float(res.fun)


def predict(offset: np.ndarray, feats: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    z = offset + sum(wi * f for wi, f in zip(w[:-1], feats)) + w[-1]
    return 1.0 / (1.0 + np.exp(-z))


def tune_tau(prob: np.ndarray, truth: np.ndarray) -> float:
    """IoU-optimal global threshold on training rooms."""
    qs = np.unique(np.quantile(prob, np.linspace(0.5, 0.999, 120)))
    best, best_tau = -1.0, 0.5
    for tau in qs:
        v = float(np.mean([iou(p >= tau, t) for p, t in zip(prob, truth)]))
        if v > best:
            best, best_tau = v, float(tau)
    return best_tau


def score(
    prob: np.ndarray, truth: np.ndarray, tau: float, prior: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-room IoU, AP, boundary F and information gain (bits)."""
    return {
        "iou": np.array([iou(p >= tau, t) for p, t in zip(prob, truth)]),
        "ap": np.array([average_precision(p, t) for p, t in zip(prob, truth)]),
        "bf": np.array([boundary_f(p >= tau, t)[2] for p, t in zip(prob, truth)]),
        "ig": np.array([info_gain_bits(p, t, prior) for p, t in zip(prob, truth)]),
    }


def summarise(scores: dict[str, np.ndarray], base: dict[str, np.ndarray] | None) -> dict:
    out = {}
    for k, v in scores.items():
        m, se = mean_se(v)
        out[k] = {"mean": m, "se": se}
        if base is not None:
            d = v - base[k]
            dm, dse = mean_se(d)
            out[k].update(
                {"delta": dm, "delta_se": dse, "z": dm / dse if dse > 0 else float("nan")}
            )
    return out


class Evaluator:
    """Fits every method on training features and scores held-out features."""

    def __init__(self, train: dict, prior: np.ndarray, target: str):
        self.train = train
        self.prior = prior
        self.target = target
        self.y = train[target]
        self.offset = np.broadcast_to(logit(prior), self.y.shape)
        self.choices: dict[str, dict] = {}
        self.carve_cache: dict[tuple, np.ndarray] = {}

    def _carve(self, f: dict, variant: tuple[str, float]) -> np.ndarray:
        key = (id(f),) + tuple(variant)
        if key not in self.carve_cache:
            self.carve_cache[key] = carving_images(f, variant[1], variant[0])
        return self.carve_cache[key]

    def _feat(self, f: dict, name: str, sigma: float, variant) -> np.ndarray:
        if name == "carving":
            # More covering first-arrival ellipses means more likely free.
            return zscore(-self._carve(f, variant or ("raw", 1e-3)), sigma)
        return zscore(f[name], sigma)

    def choose(self, name: str) -> dict:
        """Smoothing (and carving threshold) by training log-loss."""
        best: dict | None = None
        best_loss = float("inf")
        for th in CARVE_VARIANTS if name == "carving" else (None,):
            for s in SIGMAS:
                w, loss = fit_logistic(self.offset, [self._feat(self.train, name, s, th)], self.y)
                if loss < best_loss:
                    best_loss = loss
                    best = {"sigma": s, "threshold": th, "w": w, "loss": loss}
        assert best is not None
        self.choices[name] = best
        return best

    def fit_all(self) -> dict:
        for name in IMAGE_KEYS:
            if name not in self.choices:
                self.choose(name)
        feats = [
            self._feat(self.train, n, self.choices[n]["sigma"], self.choices[n]["threshold"])
            for n in IMAGE_KEYS
        ]
        w, loss = fit_logistic(self.offset, feats, self.y)
        self.choices["all"] = {"w": w, "loss": loss}
        return self.choices["all"]

    def probs(self, f: dict, method: str) -> np.ndarray:
        off = np.broadcast_to(logit(self.prior), f[self.target].shape)
        if method == "prior":
            return np.broadcast_to(self.prior, f[self.target].shape).copy()
        if method == "prior+devices":
            p = np.broadcast_to(self.prior, f[self.target].shape).copy()
            p[f["devices"]] = EPS_P
            return p
        if method == "all":
            c = self.choices
            feats = [self._feat(f, n, c[n]["sigma"], c[n]["threshold"]) for n in IMAGE_KEYS]
            return predict(off, feats, c["all"]["w"])
        c = self.choices[method]
        return predict(off, [self._feat(f, method, c["sigma"], c["threshold"])], c["w"])

    def standalone(self, f: dict, method: str) -> np.ndarray:
        """The image alone, signed so that larger means occupied (for AP)."""
        c = self.choices[method]
        z = self._feat(f, method, c["sigma"], c["threshold"])
        return z * np.sign(c["w"][0])


def evaluate_target(
    train: dict, held: dict, prior: np.ndarray, target: str, methods: list[str]
) -> dict:
    ev = Evaluator(train, prior, target)
    for m in methods:
        if m in IMAGE_KEYS:
            ev.choose(m)
    if "all" in methods:
        ev.fit_all()
    results: dict = {"methods": {}}
    base = None
    for m in ["prior"] + [x for x in methods if x != "prior"]:
        tau = tune_tau(ev.probs(train, m), train[target])
        sc = score(ev.probs(held, m), held[target], tau, prior)
        if m == "prior":
            base = sc
        entry = summarise(sc, None if m == "prior" else base)
        entry["tau"] = tau
        if m in ev.choices:
            ch = ev.choices[m]
            entry["fit"] = {
                "w": [float(x) for x in ch["w"]],
                "sigma": ch.get("sigma"),
                "threshold": ch.get("threshold"),
                "train_logloss": ch["loss"],
            }
        if m in IMAGE_KEYS:
            ap = np.array(
                [average_precision(p, t) for p, t in zip(ev.standalone(held, m), held[target])]
            )
            entry["standalone_ap"] = dict(zip(("mean", "se"), mean_se(ap)))
        results["methods"][m] = entry
        results.setdefault("per_room", {})[m] = {k: v.tolist() for k, v in sc.items()}
    results["n_rooms"] = int(len(held[target]))
    results["_evaluator"] = ev
    return results


# ---------------------------------------------------------------------------
# FWI proof of concept
# ---------------------------------------------------------------------------


def run_fwi(path: pathlib.Path, rooms: range, n_poses: int, cache: pathlib.Path) -> dict:
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    import torch

    from acoustic_system.imaging.fwi import invert_room

    torch.set_num_threads(1)
    occ, masks, ill, hist = [], [], [], []
    t0 = time.perf_counter()
    with h5py.File(path, "r") as hf:
        for n, r in enumerate(rooms):
            room = ArchiveRoom.from_group(hf[f"sample_{r:04d}"], n_poses)
            res = invert_room(
                room.mask.shape,
                room.sources,
                room.mics,
                room.drive,
                np.transpose(room.recordings, (0, 2, 1)),
            )
            occ.append(res.occupancy)
            masks.append(room.mask)
            ill.append(illuminated_boundary(room.mask, room.sources, room.mics))
            hist.append(res.history)
            print(
                f"  FWI {path.name} room {r}: misfit {res.history[0]:.3f} -> {res.history[-1]:.3f} "
                f"({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
    out = {
        "occ": np.stack(occ),
        "mask": np.stack(masks),
        "illum": np.stack(ill),
        "history": np.array(hist),
    }
    np.savez_compressed(cache, allow_pickle=False, **out)
    return out


def evaluate_fwi(
    fwi_tr: dict,
    fwi_ho: dict,
    prior: np.ndarray,
    target: str,
    held_all: dict | None,
    all_probs: np.ndarray | None,
) -> dict:
    """FWI alone and fused with the prior, scored against the prior on the FWI rooms."""
    y_tr = fwi_tr[target]
    off_tr = np.broadcast_to(logit(prior), y_tr.shape)
    feat_tr = [zscore(logit(fwi_tr["occ"]))]
    w, loss = fit_logistic(off_tr, feat_tr, y_tr)
    off_ho = np.broadcast_to(logit(prior), fwi_ho[target].shape)
    feat_ho = [zscore(logit(fwi_ho["occ"]))]
    out: dict = {"n_train": int(len(y_tr)), "n_heldout": int(len(fwi_ho[target])), "methods": {}}
    prior_tr = np.broadcast_to(prior, y_tr.shape)
    base = score(
        np.broadcast_to(prior, fwi_ho[target].shape),
        fwi_ho[target],
        tune_tau(prior_tr, y_tr),
        prior,
    )
    out["methods"]["prior"] = summarise(base, None)
    alone_tau = tune_tau(fwi_tr["occ"], y_tr)
    sc = score(np.clip(fwi_ho["occ"], EPS_P, 1 - EPS_P), fwi_ho[target], alone_tau, prior)
    out["methods"]["fwi"] = summarise(sc, base) | {"tau": alone_tau}
    fused_tr = predict(off_tr, feat_tr, w)
    tau = tune_tau(fused_tr, y_tr)
    sc = score(predict(off_ho, feat_ho, w), fwi_ho[target], tau, prior)
    out["methods"]["prior+fwi"] = summarise(sc, base) | {"tau": tau, "w": [float(x) for x in w]}
    if all_probs is not None and held_all is not None:
        out["methods"]["all physics (same rooms)"] = summarise(
            score(all_probs, fwi_ho[target], held_all["tau"], prior), base
        )
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_examples(
    held: dict, ev_fill: Evaluator, ev_ill: Evaluator, path: pathlib.Path, rooms=(0, 1, 2, 3)
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_all = ev_fill.probs(held, "all")
    p_ill = ev_ill.probs(held, "all")
    cols = [
        ("mask", held["mask"]),
        ("illuminated boundary", held["illum"]),
        ("back-projection", held["backprojection"]),
        (
            "carving (count)",
            carving_images(
                {k: held[k][list(rooms)] for k in ("mask", "sources", "mics", "residual", "drive")},
                ev_fill.choices["carving"]["threshold"][1],
                ev_fill.choices["carving"]["threshold"][0],
            ),
        ),
        ("time reversal", held["time_reversal"]),
        ("fused, filled target", p_all),
        ("fused, boundary target", p_ill),
    ]
    fig, ax = plt.subplots(len(rooms), len(cols), figsize=(2.1 * len(cols), 2.2 * len(rooms)))
    for r_i, r in enumerate(rooms):
        for c_i, (title, arr) in enumerate(cols):
            a = ax[r_i, c_i]
            img = arr[r_i] if title.startswith("carving") else arr[r]
            a.imshow(img, cmap="magma" if c_i >= 2 else "gray_r")
            s, m = held["sources"][r], held["mics"][r]
            a.plot(s[:, 1], s[:, 0], "c*", ms=5)
            a.plot(m[:, :, 1].ravel(), m[:, :, 0].ravel(), "g.", ms=3)
            a.set_xticks([])
            a.set_yticks([])
            if r_i == 0:
                a.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_fwi(fwi_ho: dict, path: pathlib.Path, n: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, len(fwi_ho["occ"]))
    fig, ax = plt.subplots(3, n, figsize=(2.3 * n, 7), squeeze=False)
    for i in range(n):
        ax[0, i].imshow(fwi_ho["mask"][i], cmap="gray_r")
        ax[1, i].imshow(fwi_ho["occ"][i], cmap="magma", vmin=0, vmax=1)
        ax[2, i].plot(fwi_ho["history"][i])
        ax[2, i].set_yscale("log")
        for a in ax[:2, i]:
            a.set_xticks([])
            a.set_yticks([])
    ax[0, 0].set_ylabel("truth")
    ax[1, 0].set_ylabel("FWI occupancy")
    ax[2, 0].set_ylabel("normalised misfit")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_deltas(res: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (tname, key) in zip(
        ax, (("filled mask", "fill_k4"), ("illuminated boundary", "illum_k4"))
    ):
        meth = [m for m in res[key]["methods"] if m != "prior"]
        for j, metric in enumerate(("iou", "ap")):
            d = [res[key]["methods"][m][metric]["delta"] for m in meth]
            e = [res[key]["methods"][m][metric]["delta_se"] for m in meth]
            a.barh(np.arange(len(meth)) + 0.4 * j, d, 0.4, xerr=e, label=f"Δ{metric.upper()}")
        a.set_yticks(np.arange(len(meth)) + 0.2)
        a.set_yticklabels(meth, fontsize=8)
        a.axvline(0, color="k", lw=0.8)
        a.set_title(f"paired Δ vs no-audio prior ({tname})")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def room_param_validation(path: pathlib.Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    shape = (160, 110)
    nx, ny = shape
    area, perim = nx * ny, 2 * (nx + ny)
    drive = room_params.dc_free_chirp(400, 0.5, 0.02, 0.25)
    mics = [(120, 80), (60, 85), (130, 25)]
    for beta in (0.05, 0.1, 0.2, 0.4):
        rec, dt = room_params.simulate_absorbing_room(shape, beta, (40, 27), mics, drive, 18000)
        est = [
            room_params.estimate_room_params(r, drive, dt, area, perim, band=(0.04, 0.2))
            for r in rec
        ]
        ad = room_params.alpha_diffuse_2d(beta)
        rows.append(
            {
                "beta": beta,
                "alpha_diffuse_2d": ad,
                "t60_est": float(np.mean([e.t60 for e in est])),
                "t60_est_sd": float(np.std([e.t60 for e in est])),
                "t30_est": float(np.mean([e.t30 for e in est])),
                "t60_sabine": room_params.t60_sabine_2d(ad, area, perim),
                "t60_eyring": room_params.t60_eyring_2d(ad, area, perim),
                "alpha_eyring_est": float(np.mean([e.alpha_eyring for e in est])),
                "alpha_sabine_est": float(np.mean([e.alpha_sabine for e in est])),
                "drr_db": [float(e.drr_db) for e in est],
            }
        )
        from acoustic_system.imaging.ir import wiener_deconvolve

        h = room_params.bandpass(wiener_deconvolve(rec[0], drive, 1e-4, 1 << 16), dt, (0.04, 0.2))
        nd = int(np.argmax(np.abs(h)))
        edc = room_params.schroeder_edc(h[nd:])
        ax.plot(np.arange(edc.size) * dt, edc, label=f"β={beta}")
    ax.set_ylim(-60, 2)
    ax.set_xlabel("t [grid units]")
    ax.set_ylabel("EDC [dB]")
    ax.set_title("Schroeder decay, 160×110 absorbing room")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--train-archive", type=pathlib.Path, default=DATA / "active_sensing_v2_train_10kx4.hdf5"
    )
    ap.add_argument(
        "--heldout-archive",
        type=pathlib.Path,
        default=DATA / "active_sensing_v2_heldout_500x8.hdf5",
    )
    ap.add_argument("--n-fit", type=int, default=500)
    ap.add_argument("--n-prior-ill", type=int, default=3000)
    ap.add_argument("--n-heldout", type=int, default=500)
    ap.add_argument("--n-fwi-train", type=int, default=12)
    ap.add_argument("--n-fwi-heldout", type=int, default=40)
    ap.add_argument("--noise-db", type=float, default=20.0)
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_REPO_ROOT / "tests/reports/imaging_2026_09_24_artifacts",
    )
    ap.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path(tempfile.gettempdir()) / "acoustic_imaging_cache",
        help="Feature cache (outside the repo by default).",
    )
    ap.add_argument("--skip-fwi", action="store_true")
    args = ap.parse_args()
    if os.environ.get("NUMBA_NUM_THREADS") != "1":
        print("note: set NUMBA_NUM_THREADS=1 on a shared machine", file=sys.stderr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    results: dict = {"config": {k: str(v) for k, v in vars(args).items()}}

    # 1. Priors from training rooms outside the fit set.
    with h5py.File(args.train_archive, "r") as hf:
        n_train = len([k for k in hf.keys() if k.startswith("sample_")])
        prior_rooms = range(args.n_fit, n_train)
        masks = np.stack([hf[f"sample_{r:04d}"]["obstacles"][()] for r in prior_rooms]).astype(bool)
        prior_fill = masks.mean(0)
        ill_path = args.cache_dir / f"prior_illum_{args.n_fit}_{args.n_prior_ill}.npy"
        if ill_path.exists():
            prior_ill = np.load(ill_path)
        else:
            acc = np.zeros(masks.shape[1:])
            for r in range(args.n_fit, args.n_fit + args.n_prior_ill):
                g = hf[f"sample_{r:04d}"]
                acc += illuminated_boundary(
                    g["obstacles"][()].astype(bool),
                    g.attrs["driver_positions"],
                    g.attrs["sensor_positions"],
                )
            prior_ill = acc / args.n_prior_ill
            np.save(ill_path, prior_ill)
    results["prior"] = {
        "n_rooms_fill": len(prior_rooms),
        "mean_fill": float(prior_fill.mean()),
        "n_rooms_illum": args.n_prior_ill,
        "mean_illum": float(prior_ill.mean()),
    }
    print(f"priors ready ({time.perf_counter() - t_start:.0f} s)", flush=True)

    # 2-3. Images for fit and held-out rooms.
    cd = args.cache_dir
    train = extract(
        args.train_archive, range(args.n_fit), 4, None, cd / f"train_{args.n_fit}_k4.npz"
    )
    held = extract(
        args.heldout_archive, range(args.n_heldout), 4, None, cd / f"held_{args.n_heldout}_k4.npz"
    )
    held8 = extract(
        args.heldout_archive, range(args.n_heldout), 8, None, cd / f"held_{args.n_heldout}_k8.npz"
    )
    methods = ["prior+devices", "ellipses", "carving", "backprojection", "time_reversal", "all"]
    r_fill = evaluate_target(train, held, prior_fill, "mask", methods)
    r_ill = evaluate_target(train, held, prior_ill, "illum", methods)
    ev_fill, ev_ill = r_fill["_evaluator"], r_ill["_evaluator"]
    results["fill_k4"] = strip(r_fill)
    results["illum_k4"] = strip(r_ill)
    # K = 8 held-out, same K = 4 fit (filled target only: its prior does not depend on K).
    k8 = {"methods": {}}
    base = None
    for m in ["prior", "backprojection", "carving", "all"]:
        tau = results["fill_k4"]["methods"][m]["tau"]
        sc = score(ev_fill.probs(held8, m), held8["mask"], tau, prior_fill)
        if m == "prior":
            base = sc
        k8["methods"][m] = summarise(sc, None if m == "prior" else base)
    results["fill_k8"] = k8
    print(f"noise-free evaluation done ({time.perf_counter() - t_start:.0f} s)", flush=True)

    # 5. Noise robustness (fused methods, re-fitted on noisy training rooms).
    tr_n = extract(
        args.train_archive,
        range(args.n_fit),
        4,
        args.noise_db,
        cd / f"train_{args.n_fit}_k4_n{args.noise_db:g}.npz",
    )
    ho_n = extract(
        args.heldout_archive,
        range(args.n_heldout),
        4,
        args.noise_db,
        cd / f"held_{args.n_heldout}_k4_n{args.noise_db:g}.npz",
    )
    rn = evaluate_target(tr_n, ho_n, prior_fill, "mask", ["backprojection", "carving", "all"])
    rn_ill = evaluate_target(tr_n, ho_n, prior_ill, "illum", ["backprojection", "carving", "all"])
    results["fill_k4_noise"] = strip(rn)
    results["illum_k4_noise"] = strip(rn_ill)
    for key in ("fill_k4", "illum_k4", "fill_k4_noise", "illum_k4_noise"):
        results[key].pop("per_room", None)
    print(f"noise evaluation done ({time.perf_counter() - t_start:.0f} s)", flush=True)

    # Figures.
    fig_examples(held, ev_fill, ev_ill, args.out_dir / "examples.png")
    fig_deltas(results, args.out_dir / "paired_deltas.png")

    # 4. FWI proof of concept.
    if not args.skip_fwi:
        fwi_tr = run_fwi(
            args.train_archive, range(args.n_fwi_train), 4, cd / f"fwi_train_{args.n_fwi_train}.npz"
        )
        fwi_ho = run_fwi(
            args.heldout_archive,
            range(args.n_fwi_heldout),
            4,
            cd / f"fwi_held_{args.n_fwi_heldout}.npz",
        )
        sub = {k: v[: args.n_fwi_heldout] for k, v in held.items()}
        results["fwi"] = {
            "fill": evaluate_fwi(
                fwi_tr,
                fwi_ho,
                prior_fill,
                "mask",
                {"tau": results["fill_k4"]["methods"]["all"]["tau"]},
                ev_fill.probs(sub, "all"),
            ),
            "illum": evaluate_fwi(
                fwi_tr,
                fwi_ho,
                prior_ill,
                "illum",
                {"tau": results["illum_k4"]["methods"]["all"]["tau"]},
                ev_ill.probs(sub, "all"),
            ),
            "misfit_final_mean": float(np.mean(fwi_ho["history"][:, -1])),
        }
        fig_fwi(fwi_ho, args.out_dir / "fwi.png")

    # 6.4 CRLB and 6.5 room parameters.
    crlb.design_chart(str(args.out_dir / "crlb_design_chart.png"))
    results["crlb"] = {db: crlb.design_table(snr_db=db) for db in (10.0, 20.0, 30.0)}
    results["room_params"] = room_param_validation(args.out_dir / "room_params_edc.png")
    results["runtime_s"] = time.perf_counter() - t_start
    with open(args.out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"wrote {args.out_dir / 'results.json'} ({results['runtime_s']:.0f} s)")
    for key in ("fill_k4", "illum_k4", "fill_k8", "fill_k4_noise", "illum_k4_noise"):
        print(f"== {key}")
        for m, e in results[key]["methods"].items():
            s = f"  {m:16s} IoU {e['iou']['mean']:.4f} AP {e['ap']['mean']:.4f} BF {e['bf']['mean']:.4f}"
            if "delta" in e["iou"]:
                s += (
                    f" | dIoU {e['iou']['delta']:+.4f}±{e['iou']['delta_se']:.4f} (z {e['iou']['z']:.1f})"
                    f" dAP {e['ap']['delta']:+.4f}±{e['ap']['delta_se']:.4f} (z {e['ap']['z']:.1f})"
                )
            print(s)


if __name__ == "__main__":
    main()
