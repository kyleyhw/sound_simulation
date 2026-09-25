"""Generative uncertainty model of whole obstacle masks (plan 6.3.4).

A masked (absorbing-state) discrete diffusion over the 64x64 filled mask,
conditioned on the four aligned physics images plus the prior logits
(K = 4), warm-started from the 6.3.1 U-Net
(``acoustic_system.imaging.generative``). It is scored against the U-Net's
calibrated marginals and against independent Bernoulli sampling from them.

Protocol (nothing is tuned on held-out rooms), as in
``eval_imaging_models.py``:

* Training rooms ``[500, 4000)`` (K = 4) train the model; the U-Net it
  starts from was trained on the same rooms.
* Training rooms ``[0, 500)`` are validation rooms: the number of decoding
  steps (on rooms 0-99) and every IoU threshold are chosen there.
* All 500 held-out rooms, first 4 poses, are scored once.

Stages (``--stages``, cached under ``<cache-dir>/generative``):

- ``train``: fine-tune the masked-diffusion U-Net;
- ``ablate``: decoding steps T on validation rooms 0-99;
- ``sample``: N samples per room for the validation and held-out rooms;
- ``score``: every metric, ``<out-dir>/results.json``, and the figures.

Usage (about an hour on two shared cores)::

    NUMBA_NUM_THREADS=1 uv run python scripts/eval_generative.py \\
        --out-dir tests/reports/imaging_generative_2026_09_25_artifacts

Needs the per-pose image caches and the ``unet_s0`` model/logits written by
``scripts/eval_imaging_models.py`` in ``--cache-dir``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from acoustic_system.imaging import generative as G  # noqa: E402
from acoustic_system.imaging import models as M  # noqa: E402
from acoustic_system.learning.metrics import mean_se  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_imaging_models", _REPO_ROOT / "scripts/eval_imaging_models.py"
)
assert _spec is not None and _spec.loader is not None
EM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EM)
EI = EM.EI

K = 4
N_GRID = 64
LOG = print


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class Data:
    """Conditioning inputs, masks, priors and the U-Net's cached held-out/val logits."""

    def __init__(self, cache: pathlib.Path, n_train_rooms: int, n_heldout: int):
        self.cache = cache
        fill, illum = EM.priors(cache, n_train_rooms)
        self.prior_fill = fill
        self.prior_logit = np.stack([EI.logit(fill), EI.logit(illum)]).astype(np.float32)
        val = EM.load_npz([EM.train_chunk_path(cache, 0)], ("images", "mask"))
        held = EM.load_npz([EM.held_path(cache, n_heldout)], ("images", "mask", "sources", "mics"))
        self.Xv = EM.aligned_inputs(val["images"], K)
        self.Xh = EM.aligned_inputs(held["images"], K)
        self.val_mask = val["mask"].astype(bool)
        self.held_mask = held["mask"].astype(bool)
        self.held_sources = held["sources"][:, :K]
        self.held_mics = held["mics"][:, :K]
        lg = cache / "logits"
        self.unet_val = EM.sigmoid(np.load(lg / "unet_s0_val.npy")[:, 0].astype(np.float64))
        self.unet_held = EM.sigmoid(np.load(lg / "unet_s0_held_k4.npy")[:, 0].astype(np.float64))
        self.n_train_rooms = n_train_rooms

    def train(self) -> tuple[np.ndarray, np.ndarray]:
        starts = range(EM.CHUNK, self.n_train_rooms, EM.CHUNK)
        tr = EM.load_npz([EM.train_chunk_path(self.cache, s) for s in starts], ("images", "mask"))
        X = EM.aligned_inputs(tr["images"], K)
        Y = tr["mask"].astype(np.float32)[:, None]
        LOG(f"loaded {len(X)} training rooms")
        return X, Y

    def prior_batch(self, b: int, t: int = 0) -> torch.Tensor:
        pl = M.d4_grid(self.prior_logit, t)
        return torch.from_numpy(np.broadcast_to(pl, (b,) + pl.shape).copy())


# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------


def model_path(gdir: pathlib.Path, name: str) -> pathlib.Path:
    return gdir / f"{name}.pt"


def build_model(args: argparse.Namespace) -> G.MaskedDiffusionUNet:
    return G.MaskedDiffusionUNet(4, args.width)


def stage_train(args: argparse.Namespace, data: Data, gdir: pathlib.Path) -> G.MaskedDiffusionUNet:
    path = model_path(gdir, args.name)
    torch.manual_seed(args.seed)
    model = build_model(args)
    if path.exists():
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    if args.warm_start:
        state = torch.load(args.cache_dir / "models" / f"{args.warm_start}.pt")
        G.warm_start_from_aligned(model, state)
        LOG(f"warm start from {args.warm_start}")
    X, Y = data.train()

    def batches(ep: int) -> Iterator[G.GenBatch]:
        rng = np.random.default_rng(args.seed * 1000 + ep)
        idx_all = rng.permutation(len(X))
        for s in range(0, len(X), args.batch):
            idx = idx_all[s : s + args.batch]
            t = int(rng.integers(8))
            yield (
                torch.from_numpy(M.d4_grid(X[idx], t)),
                data.prior_batch(len(idx), t),
                torch.from_numpy(M.d4_grid(Y[idx], t)),
            )

    LOG(f"training {args.name} ({M.n_params(model)} parameters, {args.epochs} epochs)")
    t0 = time.perf_counter()
    hist = G.fit_masked(model, batches, args.epochs, lr=args.lr, seed=args.seed, log=LOG)
    torch.save(model.state_dict(), path)
    meta = {
        "history": hist,
        "train_s": time.perf_counter() - t0,
        "params": M.n_params(model),
        "warm_start": args.warm_start,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    (gdir / f"{args.name}.json").write_text(json.dumps(meta))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def draw(
    model: G.MaskedDiffusionUNet,
    data: Data,
    X: np.ndarray,
    n: int,
    steps: int,
    seed: int,
    bias: float = 0.0,
    rooms_per_call: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """``(R, n, H, W)`` bool samples and the ``(R, H, W)`` Rao-Blackwellised mean."""
    gen = torch.Generator().manual_seed(seed)
    S, RB = [], []
    t0 = time.perf_counter()
    for s in range(0, len(X), rooms_per_call):
        x = torch.from_numpy(np.ascontiguousarray(X[s : s + rooms_per_call]))
        smp, rp = G.sample(model, x, data.prior_batch(len(x)), n, steps, gen, logit_bias=bias)
        S.append(smp)
        RB.append(rp.mean(1))
        if (s // rooms_per_call) % 25 == 0:
            LOG(f"    rooms {s + len(x)}/{len(X)}: {time.perf_counter() - t0:.0f} s")
    return np.concatenate(S), np.concatenate(RB)


def samples_cached(
    gdir: pathlib.Path,
    key: str,
    model: G.MaskedDiffusionUNet,
    data: Data,
    X: np.ndarray,
    n: int,
    steps: int,
    seed: int,
    bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    if bias != 0.0:
        key += f"_b{bias:g}"
    path = gdir / f"{key}.npz"
    if path.exists():
        with np.load(path) as z:
            shape = tuple(z["shape"])
            s = np.unpackbits(z["bits"])[: int(np.prod(shape))].reshape(shape).astype(bool)
            return s, z["rb"], float(z["seconds"])
    LOG(f"  sampling {key}: {len(X)} rooms x {n} samples, T = {steps}, bias {bias:g}")
    t0 = time.perf_counter()
    s, rb = draw(model, data, X, n, steps, seed, bias)
    sec = time.perf_counter() - t0
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        bits=np.packbits(s.ravel()),
        shape=np.array(s.shape),
        rb=rb.astype(np.float32),
        seconds=sec,
    )
    tmp.rename(path)
    return s, rb, sec


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def per_room_set_scores(samples: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    fns = {
        "energy": G.energy_score,
        "jaccard": G.jaccard_kernel_score,
        "variogram": G.variogram_score,
        "crps": G.pixel_crps,
    }
    out = {k: np.array([f(s, t) for s, t in zip(samples, truth)]) for k, f in fns.items()}
    out["pairwise_iou"] = np.array([G.mean_pairwise_iou(s) for s in samples])
    for n in (1, 8, 32):
        if samples.shape[1] >= n:
            out[f"best_of_{n}"] = np.array(
                [G.best_of_n_iou(s, t, n) for s, t in zip(samples, truth)]
            )
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    m, se = mean_se(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return {"delta": m, "se": se, "z": m / se if se > 0 else float("nan")}


def summary(sc: dict[str, np.ndarray]) -> dict:
    return {k: dict(zip(("mean", "se"), mean_se(v))) for k, v in sc.items()}


def binary_entropy(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


@torch.no_grad()
def step0_marginals(model: G.MaskedDiffusionUNet, data: Data) -> tuple[np.ndarray, np.ndarray]:
    """The model's own marginal with nothing visible (validation, held-out)."""
    model.eval()
    out = []
    for X in (data.Xv, data.Xh):
        parts = []
        for s in range(0, len(X), 50):
            x = torch.from_numpy(np.ascontiguousarray(X[s : s + 50]))
            z = torch.zeros(len(x), 1, N_GRID, N_GRID)
            logit = model(x, data.prior_batch(len(x)), z, z)
            parts.append(torch.sigmoid(logit[:, 0].double()).numpy())
        out.append(np.concatenate(parts))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Stage: ablate (validation rooms only)
# ---------------------------------------------------------------------------


def stage_ablate(
    args: argparse.Namespace, data: Data, model: G.MaskedDiffusionUNet, gdir: pathlib.Path
) -> dict:
    rooms = slice(0, args.ablate_rooms)
    X = data.Xv[rooms]
    truth = data.val_mask[rooms]
    out: dict = {}
    for T in args.ablate_steps:
        s, rb, sec = samples_cached(
            gdir, f"val_ablate_T{T}_n{args.ablate_n}", model, data, X, args.ablate_n, T, 100 + T
        )
        sc = per_room_set_scores(s, truth)
        out[str(T)] = {
            "seconds": sec,
            "sample_area": float(s.sum((2, 3)).mean()),
            "scores": summary(sc),
            "mean_abs_rb_minus_unet": float(np.abs(rb - data.unet_val[rooms]).mean()),
            "per_room_energy": sc["energy"].tolist(),
        }
        LOG(
            f"  T={T:3d}: energy {sc['energy'].mean():.3f} variogram {sc['variogram'].mean():.2f}"
            f" pairwise IoU {sc['pairwise_iou'].mean():.3f} ({sec:.0f} s)"
        )
    energies = {T: out[str(T)]["scores"]["energy"]["mean"] for T in args.ablate_steps}
    T = int(min(energies, key=lambda T: energies[T]))
    out["chosen_T"] = T
    out["true_area"] = float(truth.sum((1, 2)).mean())
    LOG(f"  chosen T = {T} (lowest validation energy score); true area {out['true_area']:.1f}")
    # Sampling logit bias at the chosen T (parallel decoding over-fills).
    out["bias"] = {}
    for b in args.ablate_bias:
        s, rb, sec = samples_cached(
            gdir, f"val_ablate_T{T}_n{args.ablate_n}", model, data, X, args.ablate_n, T, 100 + T, b
        )
        sc = per_room_set_scores(s, truth)
        out["bias"][f"{b:g}"] = {
            "seconds": sec,
            "sample_area": float(s.sum((2, 3)).mean()),
            "scores": summary(sc),
        }
        LOG(
            f"  bias {b:+.2f}: energy {sc['energy'].mean():.3f} variogram"
            f" {sc['variogram'].mean():.2f} area {s.sum((2, 3)).mean():.1f} ({sec:.0f} s)"
        )
    eb = {b: out["bias"][f"{b:g}"]["scores"]["energy"]["mean"] for b in args.ablate_bias}
    out["chosen_bias"] = float(min(eb, key=lambda b: eb[b]))
    LOG(f"  chosen bias = {out['chosen_bias']:g} (lowest validation energy score)")
    return out


# ---------------------------------------------------------------------------
# Stage: score
# ---------------------------------------------------------------------------


def stage_score(
    args: argparse.Namespace,
    data: Data,
    model: G.MaskedDiffusionUNet,
    gdir: pathlib.Path,
    steps: int,
    bias: float = 0.0,
) -> dict:
    n = args.n_samples
    res: dict = {"steps": steps, "n_samples": n, "logit_bias": bias}
    sv, rbv, sec_v = samples_cached(
        gdir, f"val_T{steps}_n{n}", model, data, data.Xv, n, steps, args.seed + 11, bias
    )
    sh, rbh, sec_h = samples_cached(
        gdir, f"held_T{steps}_n{n}", model, data, data.Xh, n, steps, args.seed + 12, bias
    )
    q0v, q0h = step0_marginals(model, data)
    res["sampling_seconds"] = {"val": sec_v, "held": sec_h}
    truth = data.held_mask
    prior = data.prior_fill

    # --- Mean maps: IoU at a validation threshold, AP, BF, info gain, calibration.
    maps_val = {
        "unet": data.unet_val,
        "gen_mean_rb": rbv.astype(np.float64),
        "gen_mean_empirical": sv.mean(1),
        "gen_step0_marginal": q0v,
    }
    maps_held = {
        "unet": data.unet_held,
        "gen_mean_rb": rbh.astype(np.float64),
        "gen_mean_empirical": sh.mean(1),
        "gen_step0_marginal": q0h,
    }
    per_room: dict[str, dict[str, np.ndarray]] = {}
    res["mean_maps"] = {}
    for name in maps_val:
        tau = EI.tune_tau(maps_val[name], data.val_mask)
        sc = EI.score(maps_held[name], truth, tau, prior)
        per_room[name] = sc
        mp, fy, w, ece = M.reliability(maps_held[name], truth)
        entry = {
            "tau": tau,
            **summary(sc),
            "ece": ece,
            "reliability": {"pred": mp.tolist(), "freq": fy.tolist(), "weight": w.tolist()},
        }
        if name != "unet":
            entry["vs_unet"] = {k: paired(sc[k], per_room["unet"][k]) for k in sc}
        res["mean_maps"][name] = entry
        LOG(
            f"  {name:20s} IoU {sc['iou'].mean():.4f} AP {np.nanmean(sc['ap']):.4f}"
            f" IG {sc['ig'].mean():.1f} ECE {ece:.4f} (tau {tau:.3f})"
        )
    prior_sc = EI.score(
        np.broadcast_to(prior, truth.shape),
        truth,
        EI.tune_tau(np.broadcast_to(prior, data.val_mask.shape), data.val_mask),
        prior,
    )
    res["mean_maps"]["prior"] = summary(prior_sc)
    res["rb_vs_unet_mean_abs_diff"] = float(np.abs(rbh - data.unet_held).mean())
    res["rb_vs_empirical_mean_abs_diff"] = float(np.abs(rbh - sh.mean(1)).mean())

    # --- Sample sets: generative, decorrelated generative, independent U-Net Bernoulli.
    rng = np.random.default_rng(args.seed + 13)
    sets = {
        "generative": sh,
        "generative decorrelated": np.stack([G.decorrelate(s, rng) for s in sh]),
        "U-Net independent Bernoulli": np.stack(
            [G.independent_bernoulli(q, n, rng) for q in data.unet_held]
        ),
    }
    tau_u = res["mean_maps"]["unet"]["tau"]
    point = (data.unet_held >= tau_u)[:, None]
    set_scores = {name: per_room_set_scores(s, truth) for name, s in sets.items()}
    set_scores["U-Net thresholded (point)"] = {
        "energy": np.array([G.energy_score(p, t) for p, t in zip(point, truth)]),
        "jaccard": np.array([G.jaccard_kernel_score(p, t) for p, t in zip(point, truth)]),
        "variogram": np.array([G.variogram_score(p, t) for p, t in zip(point, truth)]),
        "crps": np.array([G.pixel_crps(p, t) for p, t in zip(point, truth)]),
    }
    res["sets"] = {}
    for name, sc in set_scores.items():
        entry = summary(sc)
        if name != "generative":
            entry["generative_minus_this"] = {
                k: paired(set_scores["generative"][k], sc[k]) for k in sc
            }
        res["sets"][name] = entry
        LOG(
            f"  {name:28s} ES {sc['energy'].mean():.3f} JS {sc['jaccard'].mean():.4f}"
            f" VS {sc['variogram'].mean():.1f} CRPS {sc['crps'].mean():.2f}"
            + (f" best-of-32 {sc['best_of_32'].mean():.4f}" if "best_of_32" in sc else "")
        )

    # --- Diversity against error, per room.
    gen = set_scores["generative"]
    err_mean = 1.0 - per_room["gen_mean_rb"]["iou"]
    err_unet = 1.0 - per_room["unet"]["iou"]
    err_sample = 1.0 - gen["best_of_1"]
    diversity = 1.0 - gen["pairwise_iou"]
    unet_entropy = binary_entropy(data.unet_held).sum((1, 2))
    unet_ent_norm = unet_entropy / np.maximum(data.unet_held.sum((1, 2)), 1.0)
    ind_div = 1.0 - set_scores["U-Net independent Bernoulli"]["pairwise_iou"]

    def rho(a: np.ndarray, b: np.ndarray) -> dict:
        r, p = spearmanr(a, b)
        return {"spearman": float(r), "p": float(p)}

    res["diversity"] = {
        "generative_diversity_vs_mean_map_error": rho(diversity, err_mean),
        "generative_diversity_vs_sample_error": rho(diversity, err_sample),
        "generative_diversity_vs_unet_error": rho(diversity, err_unet),
        "unet_total_entropy_vs_unet_error": rho(unet_entropy, err_unet),
        "unet_entropy_per_predicted_cell_vs_unet_error": rho(unet_ent_norm, err_unet),
        "independent_diversity_vs_unet_error": rho(ind_div, err_unet),
        "generative_mean_pairwise_iou": summary({"x": gen["pairwise_iou"]})["x"],
        "generative_mean_pairwise_iou_quartiles": np.quantile(
            gen["pairwise_iou"], [0.25, 0.5, 0.75]
        ).tolist(),
    }
    for key, v in res["diversity"].items():
        LOG(f"  {key}: {v}")

    # Diversity terciles: mean-map IoU per tercile of generative diversity.
    q = np.quantile(diversity, [1 / 3, 2 / 3])
    bins = np.digitize(diversity, q)
    res["diversity_terciles"] = [
        {
            "diversity_range": [
                float(diversity[bins == b].min()),
                float(diversity[bins == b].max()),
            ],
            "mean_map_iou": float(per_room["gen_mean_rb"]["iou"][bins == b].mean()),
            "unet_iou": float(per_room["unet"]["iou"][bins == b].mean()),
            "n": int((bins == b).sum()),
        }
        for b in range(3)
    ]

    # Empty-sample and area statistics.
    area_true = truth.sum((1, 2))
    area_samp = sh.sum((2, 3))
    res["area"] = {
        "true_mean": float(area_true.mean()),
        "sample_mean": float(area_samp.mean()),
        "unet_expected": float(data.unet_held.sum((1, 2)).mean()),
        "rb_expected": float(rbh.sum((1, 2)).mean()),
        "fraction_empty_samples": float((area_samp == 0).mean()),
        "coverage_of_true_area_in_sample_range": float(
            np.mean((area_true >= area_samp.min(1)) & (area_true <= area_samp.max(1)))
        ),
    }
    res["_per_room"] = {
        "gen_mean_iou": per_room["gen_mean_rb"]["iou"],
        "unet_iou": per_room["unet"]["iou"],
        "diversity": diversity,
        "best_of_1": gen["best_of_1"],
        "best_of_32": gen["best_of_32"],
    }
    res["_samples"] = sh
    res["_rb"] = rbh
    res["_indep"] = sets["U-Net independent Bernoulli"]
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_examples(data: Data, res: dict, path: pathlib.Path, rooms: tuple[int, ...]) -> None:
    plt = _plt()
    sh, rb, ind = res["_samples"], res["_rb"], res["_indep"]
    cols = ["truth", "U-Net p", "sample mean"] + [f"sample {i + 1}" for i in range(4)]
    cols.append("indep. Bernoulli")
    fig, ax = plt.subplots(len(rooms), len(cols), figsize=(1.75 * len(cols), 1.9 * len(rooms)))
    pr = res["_per_room"]
    for i, r in enumerate(rooms):
        panels = [
            (data.held_mask[r], "gray_r"),
            (data.unet_held[r], "magma"),
            (rb[r], "magma"),
        ]
        panels += [(sh[r, j], "gray_r") for j in range(4)]
        panels.append((ind[r, 0], "gray_r"))
        for j, (arr, cmap) in enumerate(panels):
            a = ax[i, j]
            a.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            s, m = data.held_sources[r], data.held_mics[r]
            a.plot(s[:, 1], s[:, 0], "c*", ms=3)
            a.plot(m[:, :, 1].ravel(), m[:, :, 0].ravel(), "g.", ms=1.5)
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(cols[j], fontsize=8)
        ax[i, 0].set_ylabel(
            f"room {r}\nIoU {pr['unet_iou'][r]:.2f}\ndiv {pr['diversity'][r]:.2f}", fontsize=7
        )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_diversity(res: dict, path: pathlib.Path) -> None:
    plt = _plt()
    pr = res["_per_room"]
    d = res["diversity"]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.5))
    ax[0].scatter(pr["diversity"], 1 - pr["gen_mean_iou"], s=5, alpha=0.5)
    ax[0].set_xlabel("sample diversity (1 - mean pairwise IoU)")
    ax[0].set_ylabel("error of the mean map (1 - IoU)")
    r = d["generative_diversity_vs_mean_map_error"]["spearman"]
    ax[0].set_title(f"held-out rooms, Spearman {r:.2f}", fontsize=9)
    ax[0].grid(alpha=0.3)
    ax[1].scatter(pr["best_of_1"], pr["best_of_32"], s=5, alpha=0.5, label="best of 32")
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[1].set_xlabel("mean single-sample IoU")
    ax[1].set_ylabel("oracle best-of-32 IoU")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_reliability(variants: dict[str, dict], path: pathlib.Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    first = next(iter(variants.values()))
    curves = [("U-Net", first["mean_maps"]["unet"])]
    curves.append(("generative, nothing visible", first["mean_maps"]["gen_step0_marginal"]))
    for tag, res in variants.items():
        curves.append((f"sample mean, {tag}", res["mean_maps"]["gen_mean_rb"]))
    for lab, e in curves:
        c = e["reliability"]
        ax.plot(c["pred"], c["freq"], "o-", ms=3, label=f"{lab} (ECE {e['ece']:.4f})")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("reliability, filled mask (K=4, held-out)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_ablation(abl: dict, path: pathlib.Path) -> None:
    plt = _plt()
    Ts = sorted(int(t) for t in abl if t.isdigit())
    fig, ax = plt.subplots(1, 4, figsize=(12.4, 3.0))
    for a, key, lab in zip(
        ax,
        ("energy", "variogram", "pairwise_iou"),
        ("energy score", "variogram score", "mean pairwise IoU"),
    ):
        y = [abl[str(t)]["scores"][key]["mean"] for t in Ts]
        e = [abl[str(t)]["scores"][key]["se"] for t in Ts]
        a.errorbar(Ts, y, yerr=e, marker="o", ms=3, capsize=2)
        a.set_xscale("log", base=2)
        a.set_xticks(Ts)
        a.set_xticklabels([str(t) for t in Ts])
        a.set_xlabel("decoding steps T")
        a.set_title(lab, fontsize=9)
        a.grid(alpha=0.3)
    bs = sorted(float(b) for b in abl.get("bias", {}))
    if bs:
        a = ax[3]
        es = [abl["bias"][f"{b:g}"]["scores"]["energy"]["mean"] for b in bs]
        area = [abl["bias"][f"{b:g}"]["sample_area"] for b in bs]
        a.plot(bs, es, "o-", ms=3, label="energy score")
        a.set_xlabel(f"sampling logit bias (T = {abl['chosen_T']})")
        a.set_ylabel("energy score")
        a2 = a.twinx()
        a2.plot(bs, area, "s--", ms=3, color="C1", label="mean sample area")
        a2.axhline(abl["true_area"], color="C1", ls=":", lw=0.8)
        a2.set_ylabel("cells (dotted: true)")
        a.set_title("logit bias", fontsize=9)
        a.grid(alpha=0.3)
    fig.suptitle("validation rooms 0-99", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating | np.integer):
        return x.item()
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path(tempfile.gettempdir()) / "acoustic_imaging_models_cache",
    )
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_REPO_ROOT / "tests/reports/imaging_generative_2026_09_25_artifacts",
    )
    ap.add_argument("--stages", default="train,ablate,sample,score")
    ap.add_argument("--name", default="maskdiff_s0")
    ap.add_argument("--warm-start", default="unet_s0", help="U-Net to start from ('' = scratch)")
    ap.add_argument("--n-train-rooms", type=int, default=4000)
    ap.add_argument("--n-heldout", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--steps", type=int, default=0, help="decoding steps (0 = ablation choice)")
    ap.add_argument("--ablate-steps", default="4,8,16,32")
    ap.add_argument("--ablate-rooms", type=int, default=100)
    ap.add_argument("--ablate-n", type=int, default=16)
    ap.add_argument("--ablate-bias", default="0,-0.5,-1,-1.5")
    ap.add_argument("--examples", default="0,1,2,3,5,8")
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    if os.environ.get("NUMBA_NUM_THREADS") != "1":
        print("note: set NUMBA_NUM_THREADS=1 on a shared machine", file=sys.stderr)
    torch.set_num_threads(args.torch_threads)
    args.ablate_steps = tuple(int(x) for x in args.ablate_steps.split(","))
    args.ablate_bias = tuple(float(x) for x in args.ablate_bias.split(","))
    stages = tuple(args.stages.split(","))
    gdir = args.cache_dir / "generative"
    gdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data = Data(args.cache_dir, args.n_train_rooms, args.n_heldout)
    model = stage_train(args, data, gdir)
    results: dict = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "model": json.loads((gdir / f"{args.name}.json").read_text()),
    }
    steps = args.steps
    biases = [0.0]
    if "ablate" in stages or steps == 0:
        results["ablation"] = stage_ablate(args, data, model, gdir)
        steps = steps or results["ablation"]["chosen_T"]
        if results["ablation"]["chosen_bias"] != 0.0:
            biases.append(results["ablation"]["chosen_bias"])
    if "sample" in stages or "score" in stages:
        variants = {}
        for b in biases:
            LOG(f"scoring logit bias {b:g}")
            variants[f"bias {b:g}"] = stage_score(args, data, model, gdir, steps, b)
        results["held"] = variants
        if "score" in stages:
            out = args.out_dir
            out.mkdir(parents=True, exist_ok=True)
            rooms = tuple(int(r) for r in args.examples.split(","))
            main_res = variants[f"bias {biases[-1]:g}"]
            fig_examples(data, main_res, out / "examples.png", rooms)
            if len(biases) > 1:
                fig_examples(data, variants["bias 0"], out / "examples_bias0.png", rooms)
            fig_diversity(main_res, out / "diversity_vs_error.png")
            fig_reliability(variants, out / "reliability.png")
            if "ablation" in results:
                fig_ablation(results["ablation"], out / "steps_ablation.png")
            results["runtime_s"] = time.perf_counter() - t0
            (out / "results.json").write_text(json.dumps(_jsonable(results), indent=1))
            LOG(f"wrote {out / 'results.json'}")
    LOG(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
