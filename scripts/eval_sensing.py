"""Full re-scoring of sensing checkpoints against no-audio baselines (plan 3.4.6/3.4.7).

For every checkpoint and every K in ``--poses`` this scores the held-out
archive with:

* **IoU @ tau** at the checkpoint's stored, validation-selected operating
  point (calibrated Bayes fusion), and IoU @ 0.5 for continuity with
  the pre-2.3 reports;
* **oracle IoU**: the best threshold chosen on the held-out set itself.
  It is labelled as an oracle and is an upper bound, never a result;
* **AP**, per-room average precision of the fused map (threshold-free,
  invariant to recalibration);
* **info gain**, log-likelihood gain in bits per room of the fused
  probabilities over (a) the per-pixel training prior map and (b) the
  scalar training prior (see ``learning/metrics.py``);
* **boundary F** @ tau, with a 1-cell tolerance;
* for joint/skip models, the same numbers for the model's *native*
  joint forward over the first K poses.

The no-audio baselines are scored on the same rooms:

* the per-pixel training prior map, thresholded at a tau chosen on
  training rooms (not held-out);
* predict-all.

The reported significance is the per-room paired difference to the prior
map, as mean ± SE (z = mean / SE).

Usage::

    python scripts/eval_sensing.py \\
        --checkpoint checkpoints/skip_v2/best_iou.pt \\
        --train-archive data/training_data/active_sensing_v2_train_10kx4.hdf5 \\
        --heldout data/training_data/active_sensing_v2_heldout_500x8.hdf5 \\
        --out tests/reports/rescore_artifacts/skip_v2.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import h5py
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from acoustic_system.learning.calibration import load_calibration  # noqa: E402
from acoustic_system.learning.metrics import (  # noqa: E402
    average_precision,
    boundary_f,
    info_gain_bits,
    iou,
    mean_se,
    paired_diff_se,
)
from acoustic_system.learning.model import build_model  # noqa: E402

TAU_GRID = np.concatenate([np.arange(0.01, 0.2, 0.01), np.arange(0.2, 0.96, 0.05)])


def load_masks(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        return np.stack([np.asarray(f[k]["obstacles"], dtype=bool) for k in keys])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def prior_baseline(train_masks: np.ndarray, rng_seed: int = 0) -> tuple[np.ndarray, float]:
    """Per-pixel prior map from 90 % of training rooms; tau chosen on the rest."""
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(len(train_masks))
    n_sel = max(1, len(perm) // 10)
    fit, sel = train_masks[perm[n_sel:]], train_masks[perm[:n_sel]]
    prior_map = fit.mean(axis=0)
    best_tau, best = 0.5, -1.0
    for tau in np.linspace(0.0, float(prior_map.max()), 80)[1:]:
        pred = prior_map >= tau
        v = float(np.mean([iou(pred, m) for m in sel]))
        if v > best:
            best, best_tau = v, float(tau)
    return prior_map, best_tau


def score_rooms(
    probs: list[np.ndarray], truths: np.ndarray, tau: float, prior_map: np.ndarray, pi: float
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {
        "iou_tau": [],
        "iou_05": [],
        "ap": [],
        "gain_map": [],
        "gain_scalar": [],
        "bf": [],
    }
    for p, t in zip(probs, truths):
        out["iou_tau"].append(iou(p > tau, t))
        out["iou_05"].append(iou(p > 0.5, t))
        out["ap"].append(average_precision(p, t))
        out["gain_map"].append(info_gain_bits(p, t, prior_map))
        out["gain_scalar"].append(info_gain_bits(p, t, pi))
        out["bf"].append(boundary_f(p > tau, t)[2])
    return out


def oracle_iou(probs: list[np.ndarray], truths: np.ndarray) -> tuple[float, float]:
    best, best_tau = -1.0, 0.5
    for tau in TAU_GRID:
        v = float(np.mean([iou(p > tau, t) for p, t in zip(probs, truths)]))
        if v > best:
            best, best_tau = v, float(tau)
    return best, best_tau


def summarise(scores: dict[str, list[float]], base_iou: list[float]) -> dict[str, float]:
    row: dict[str, float] = {}
    for k, v in scores.items():
        m, se = mean_se(v)
        row[k] = m
        row[k + "_se"] = se
    d, dse = paired_diff_se(scores["iou_tau"], base_iou)
    row["d_iou_vs_prior"] = d
    row["d_iou_vs_prior_se"] = dse
    row["z_vs_prior"] = d / dse if dse and dse > 0 else float("nan")
    g, gse = mean_se(scores["gain_map"])
    row["z_gain"] = g / gse if gse and gse > 0 else float("nan")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True, nargs="+")
    ap.add_argument("--train-archive", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--poses", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-rooms", type=int, default=None)
    ap.add_argument("--out", required=True, help="JSON output path")
    args = ap.parse_args()
    t0 = time.perf_counter()

    train_masks = load_masks(args.train_archive)
    prior_map, prior_tau = prior_baseline(train_masks)
    pi = float(train_masks.mean())
    truths = load_masks(args.heldout)
    if args.max_rooms:
        truths = truths[: args.max_rooms]
    n_rooms = len(truths)

    base_iou = [iou(prior_map >= prior_tau, t) for t in truths]
    base = {
        "prior_map_iou": mean_se(base_iou),
        "prior_map_tau": prior_tau,
        "prior_map_ap": mean_se([average_precision(prior_map, t) for t in truths]),
        "prior_map_bf": mean_se([boundary_f(prior_map >= prior_tau, t)[2] for t in truths]),
        "predict_all_iou": mean_se([iou(np.ones_like(t), t) for t in truths]),
        "scalar_prior": pi,
        "n_rooms": n_rooms,
    }
    print(
        f"[rescore] baselines on {n_rooms} held-out rooms: prior map IoU "
        f"{base['prior_map_iou'][0]:.4f}±{base['prior_map_iou'][1]:.4f} (tau {prior_tau:.3f} "
        f"chosen on train rooms), AP {base['prior_map_ap'][0]:.4f}, predict-all "
        f"{base['predict_all_iou'][0]:.4f}",
        flush=True,
    )

    results: dict[str, object] = {"baselines": base, "models": {}}
    for ck_path in args.checkpoint:
        ckpt = torch.load(ck_path, map_location="cpu", weights_only=False)
        mtype = str(ckpt.get("model_type", "dual"))
        model = build_model(mtype, n_mics=int(ckpt.get("n_mics", 2)))
        model.load_state_dict(ckpt["model"])
        model.eval()
        calib = load_calibration(ck_path)
        cal_t = float(calib["temperature"]) if calib else 1.0
        cal_b = float(calib["bias"]) if calib else 0.0
        tau = float(calib.get("threshold", 0.5)) if calib else 0.5
        pi_model = float(
            calib["prior"] if calib else ckpt.get("train_prior", pi)
        )  # the prior the fusion rule uses
        prior_logit = float(np.log(pi_model / (1 - pi_model)))
        native = mtype in ("joint", "skip")

        fused: dict[int, list[np.ndarray]] = {k: [] for k in args.poses}
        nat: dict[int, list[np.ndarray]] = {k: [] for k in args.poses}
        with h5py.File(args.heldout, "r") as f, torch.no_grad():
            keys = sorted(k for k in f.keys() if k.startswith("sample_"))[:n_rooms]
            for key in keys:
                grp = f[key]
                sensor = np.asarray(grp["sensor"], dtype=np.float32)  # (K, T, M)
                source = torch.from_numpy(np.asarray(grp["source"], dtype=np.float32))
                sens_t = torch.from_numpy(sensor.transpose(0, 2, 1).copy())  # (K, M, T)
                src_t = source[None, None].repeat(sens_t.shape[0], 1, 1)
                logits = model(sens_t, src_t)[:, 0].numpy().astype(np.float64)
                logits = logits / cal_t + cal_b
                for k in args.poses:
                    fused[k].append(sigmoid(logits[:k].sum(0) - (k - 1) * prior_logit))
                    if native:
                        nl = model(sens_t[None, :k], source[None, None])[0, 0].numpy()
                        nat[k].append(sigmoid(nl.astype(np.float64) / cal_t + cal_b))

        model_rows: dict[str, object] = {"model_type": mtype, "tau": tau, "calibrated": bool(calib)}
        for k in args.poses:
            row = summarise(score_rooms(fused[k], truths, tau, prior_map, pi), base_iou)
            row["oracle_iou"], row["oracle_tau"] = oracle_iou(fused[k], truths)
            model_rows[f"bayes_K{k}"] = row
            if native:
                nrow = summarise(score_rooms(nat[k], truths, tau, prior_map, pi), base_iou)
                model_rows[f"native_K{k}"] = nrow
            print(
                f"[rescore] {pathlib.Path(ck_path).parent.name} K={k}: IoU@tau {row['iou_tau']:.4f} "
                f"(prior map {base['prior_map_iou'][0]:.4f}; diff {row['d_iou_vs_prior']:+.4f}"
                f"±{row['d_iou_vs_prior_se']:.4f}, z={row['z_vs_prior']:.1f}) AP {row['ap']:.4f} "
                f"gain {row['gain_map']:+.1f} bits (z={row['z_gain']:.1f}) BF {row['bf']:.3f} "
                f"oracle {row['oracle_iou']:.4f}",
                flush=True,
            )
        results["models"][str(ck_path)] = model_rows  # type: ignore[index]

    results["runtime_s"] = time.perf_counter() - t0
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"[rescore] wrote {out} ({results['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
