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
    prior = float(ckpt.get("acquisition", {}).get("mean_obstacle_fraction", 0.0582))

    with h5py.File(args.dataset, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        n_total = len(keys)
        n_val = max(1, int(val_frac * n_total))
        # Identical permutation to train.py's split: room-level datasets
        # (joint/skip) index rooms directly, so these indices ARE rooms.
        indices = torch.randperm(
            n_total, generator=torch.Generator().manual_seed(split_seed)
        ).tolist()
        val_rooms = indices[n_total - n_val :][: args.max_rooms]
        print(
            f"[calibrate] val split: {n_val}/{n_total} rooms (seed {split_seed}), "
            f"fitting on {len(val_rooms)}; prior={prior:.4f}"
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
