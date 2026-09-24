"""Fit per-pose logit calibration for a sensing checkpoint (Task 2.3e).

Reproduces the training run's validation split (same permutation seed
and fraction, read from the checkpoint), collects the model's
*per-pose* logits and the true masks on validation rooms only, fits
Platt-style $(T, b)$ (see ``learning/calibration.py`` for the
mathematics and rationale), and writes ``calibration.json`` next to
the checkpoint, where ``eval_multipose.py`` and ``learning/sensing.py``
auto-load it.

Fitting on the validation split matters: training rooms are partly
memorised, so their logits are anomalously confident and would fit a
temperature that under-corrects on unseen rooms. Held-out archives
must not be used either — they are the measurement instrument.

Usage::

    python scripts/fit_calibration.py \\
        --checkpoint checkpoints/skip_v2/best_iou.pt \\
        --dataset data/training_data/active_sensing_v2_train_10kx4.hdf5 \\
        [--max-rooms 300]

``--max-rooms`` caps the fitted subset (default 300 rooms ~ 5M pixels
at K=4): two scalar parameters saturate long before that, and the cap
keeps the fit under a minute on CPU.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from acoustic_system.learning.calibration import (  # noqa: E402
    fit_temperature_bias,
    save_calibration,
)
from acoustic_system.learning.model import build_model  # noqa: E402
from acoustic_system.learning.train import room_level_split  # noqa: E402


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return torch.nn.functional.interpolate(t, size=(size, size), mode="nearest")[0, 0].numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="The TRAINING archive the model was fit on.")
    p.add_argument("--max-rooms", type=int, default=300)
    args = p.parse_args()
    t0 = time.perf_counter()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(str(ckpt.get("model_type", "dual")), n_mics=int(ckpt.get("n_mics", 2)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    target_size = int(ckpt["args"].get("target_size", 64))
    split_seed = int(ckpt["args"].get("seed", 0))
    val_frac = float(ckpt["args"].get("val_frac", 0.1))

    # The archive must be the one the checkpoint was trained on, otherwise
    # "the validation split" is meaningless (audit 3.4.2).
    trained_on = Path(str(ckpt["args"].get("dataset", ""))).name
    if trained_on and Path(args.dataset).name != trained_on:
        raise SystemExit(
            f"--dataset {Path(args.dataset).name} is not the checkpoint's training "
            f"archive ({trained_on}); calibration must be fitted on its val split"
        )

    with h5py.File(args.dataset, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        n_total = len(keys)
        if "val_rooms" in ckpt:
            # Post-audit checkpoints record their room-level val split.
            all_val = [int(r) for r in ckpt["val_rooms"]]
        else:
            # Legacy checkpoints: the same room-level permutation train.py
            # used. Exact for room-level models (joint/skip) and single-pose
            # archives; flattened multi-pose checkpoints trained before the
            # audit split by pose, so no leak-free val set exists for them.
            if str(ckpt.get("model_type")) not in ("joint", "skip"):
                with h5py.File(args.dataset, "r") as g:
                    if int(g.attrs.get("poses_per_room", 1)) > 1:
                        raise SystemExit(
                            "legacy flattened multi-pose checkpoint: its val split mixed "
                            "poses of training rooms; retrain with the fixed train.py"
                        )
            _, all_val = room_level_split(n_total, val_frac, split_seed)
        val_rooms = all_val[: args.max_rooms]
        # Prior from training rooms only (never the held-out archive).
        if "train_prior" in ckpt:
            prior = float(ckpt["train_prior"])
        else:
            val_set = set(all_val)
            occ = [
                float(np.asarray(f[keys[r]]["obstacles"]).mean())
                for r in range(n_total)
                if r not in val_set
            ]
            prior = float(np.mean(occ))
        print(
            f"[calibrate] val split: {len(all_val)}/{n_total} rooms (seed {split_seed}), "
            f"fitting on {len(val_rooms)}; train prior={prior:.4f}"
        )

        logit_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for ridx in val_rooms:
                grp = f[keys[ridx]]
                sensor = np.asarray(grp["sensor"], dtype=np.float32)
                if sensor.ndim == 2:  # single-pose archive
                    sensor = sensor[None]
                sensor_t = torch.from_numpy(sensor.transpose(0, 2, 1).copy())  # (K, M, T)
                source = torch.from_numpy(np.asarray(grp["source"], dtype=np.float32))
                src_t = source[None, None].repeat(sensor_t.shape[0], 1, 1)
                logits = model(sensor_t, src_t)[:, 0].numpy()  # (K, H, W)
                truth = resize_mask(np.asarray(grp["obstacles"], dtype=np.float32), target_size)
                logit_chunks.append(logits.reshape(-1))
                label_chunks.append(np.tile(truth.reshape(-1), logits.shape[0]))

    all_logits = np.concatenate(logit_chunks)
    all_labels = np.concatenate(label_chunks)
    temperature, bias = fit_temperature_bias(all_logits, all_labels)

    # Operating point: the IoU-optimal threshold on the CALIBRATED,
    # Bayes-fused validation maps (all K poses the archive provides).
    # Selected on validation only — held-out archives never see this
    # sweep. Scalar calibration is affine-monotone in the fused logit,
    # so this threshold choice is the IoU-bearing half of Task 2.3e;
    # (T, b) supply the honest probability scale it lives on.
    prior_c = float(np.clip(prior, 1e-4, 1 - 1e-4))
    prior_logit = float(np.log(prior_c / (1.0 - prior_c)))
    sweep = np.concatenate([np.arange(0.02, 0.2, 0.02), np.arange(0.2, 0.95, 0.05)])
    # Rebuild each room's calibrated fused map from the stored chunks:
    # a room's logits chunk is (K * side * side,), its labels chunk the
    # truth tiled K times, so K = len(logits) / len(truth).
    fused_maps: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for logits_flat, labels_flat in zip(logit_chunks, label_chunks):
        side = target_size
        n_px = side * side
        k = logits_flat.size // n_px
        lg = logits_flat.reshape(k, side, side) / temperature + bias
        fused_maps.append(1.0 / (1.0 + np.exp(-(lg.sum(axis=0) - (k - 1) * prior_logit))))
        truths.append(labels_flat[:n_px].reshape(side, side))
    best_tau, best_iou = 0.5, -1.0
    for tau in sweep:
        vals = []
        for prob, tr in zip(fused_maps, truths):
            pred = (prob > tau).astype(np.float32)
            inter = float((pred * tr).sum())
            union = float(pred.sum() + tr.sum() - inter)
            vals.append((inter + 1e-6) / (union + 1e-6))
        mean_iou = float(np.mean(vals))
        if mean_iou > best_iou:
            best_tau, best_iou = float(tau), mean_iou
    out = save_calibration(args.checkpoint, temperature, bias, prior, threshold=best_tau)
    print(
        f"[calibrate] T={temperature:.4f} b={bias:.4f} prior={prior:.4f} "
        f"threshold={best_tau:.2f} (val fused IoU {best_iou:.4f}, "
        f"{all_logits.size:,} pixels) -> {out}"
    )
    print(f"[calibrate] runtime {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
