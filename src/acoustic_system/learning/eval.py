"""Eval script for the active-sensing CNN.

Loads a checkpoint, runs inference on N samples from a dataset, prints
mean IoU, and saves a side-by-side prediction-vs-truth plot grid.

CLI::

    python -m acoustic_system.learning.eval \\
        --checkpoint checkpoints/run_NAME/best.pt \\
        --dataset path/to/dataset.hdf5 \\
        --n-samples 8 \\
        --output checkpoints/run_NAME/preds.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from acoustic_system.learning.dataset import ActiveSensingDataset
from acoustic_system.learning.losses import iou_score
from acoustic_system.learning.model import DualInputCNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-samples", type=int, default=8, help="Samples to plot.")
    p.add_argument("--output", required=True, help="Path to write the plot grid.")
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_mics = int(ckpt.get("n_mics", 2))
    target_size = int(ckpt["args"].get("target_size", 64))
    model = DualInputCNN(n_mics=n_mics).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = ActiveSensingDataset(args.dataset, target_mask_size=target_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # ---- whole-dataset IoU --------------------------------------------
    ious: list[float] = []
    with torch.no_grad():
        for sensor, source, mask in loader:
            sensor = sensor.to(device)
            source = source.to(device)
            mask = mask.to(device)
            logits = model(sensor, source)
            ious.append(iou_score(logits, mask, threshold=args.threshold).item())
    mean_iou = float(np.mean(ious))
    print(
        f"[eval] mean IoU @ threshold={args.threshold}: {mean_iou:.4f} over {len(dataset)} samples"
    )

    # ---- prediction grid ----------------------------------------------
    n = min(args.n_samples, len(dataset))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(dataset), size=n, replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
    if n == 1:
        axes = np.array([axes])
    for row, idx in enumerate(sample_idx):
        sensor, source, mask = dataset[int(idx)]
        with torch.no_grad():
            logits = model(sensor.unsqueeze(0).to(device), source.unsqueeze(0).to(device))
        pred = torch.sigmoid(logits).cpu().numpy()[0, 0]
        pred_bin = (pred > args.threshold).astype(np.float32)

        # Per-sample IoU for the panel title.
        truth = mask.numpy()
        inter = float((pred_bin * truth).sum())
        union = float(pred_bin.sum() + truth.sum() - inter)
        sample_iou = inter / max(union, 1e-6)

        axes[row, 0].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"sample {idx}: truth")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(pred, cmap="viridis", vmin=0, vmax=1)
        axes[row, 1].set_title("predicted prob")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        axes[row, 2].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"pred @ {args.threshold:.2f}  IoU={sample_iou:.3f}")
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
