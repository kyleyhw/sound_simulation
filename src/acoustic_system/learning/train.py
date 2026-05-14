"""Training loop for the active-sensing CNN.

CLI::

    python -m acoustic_system.learning.train \\
        --dataset path/to/dataset.hdf5 \\
        --epochs 50 \\
        --batch-size 16 \\
        --lr 1e-3 \\
        --target-size 64 \\
        --ckpt-dir checkpoints/run_NAME

Produces ``ckpt-dir/final.pt`` (state dict + args + history) and
``ckpt-dir/loss.png`` (train + val loss + val IoU vs epoch).

Defaults are tuned for laptop CPU smoke-testing: a 200-sample dataset
trains for 50 epochs in ~5 minutes. Push to GPU + larger dataset for
real runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from acoustic_system.learning.dataset import ActiveSensingDataset
from acoustic_system.learning.losses import bce_dice_loss, iou_score
from acoustic_system.learning.model import DualInputCNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="HDF5 archive from generate_active_sensing.py")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--target-size", type=int, default=64, help="Predicted mask side length.")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on Windows; HDF5 + multiprocess can be flaky.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cpu / cuda. Defaults to cuda if available, otherwise cpu.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = ActiveSensingDataset(args.dataset, target_mask_size=args.target_size)
    n_total = len(dataset)
    n_val = max(1, int(args.val_frac * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = DualInputCNN(n_mics=dataset.n_mics).to(device)
    print(f"[train] device={device}  params={model.num_params:,}")
    print(
        f"[train] dataset n_total={n_total} n_train={n_train} n_val={n_val} "
        f"n_mics={dataset.n_mics} T_rec={dataset.t_rec} T_audio={dataset.t_audio}"
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_val = float("inf")

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        # ---- train ------------------------------------------------------
        model.train()
        train_losses: list[float] = []
        for sensor, source, mask in train_loader:
            sensor = sensor.to(device)
            source = source.to(device)
            mask = mask.to(device)
            opt.zero_grad()
            logits = model(sensor, source)
            loss = bce_dice_loss(logits, mask)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # ---- validate ---------------------------------------------------
        model.eval()
        val_losses: list[float] = []
        ious: list[float] = []
        with torch.no_grad():
            for sensor, source, mask in val_loader:
                sensor = sensor.to(device)
                source = source.to(device)
                mask = mask.to(device)
                logits = model(sensor, source)
                val_losses.append(bce_dice_loss(logits, mask).item())
                ious.append(iou_score(logits, mask).item())

        train_l = float(np.mean(train_losses))
        val_l = float(np.mean(val_losses))
        val_iou = float(np.mean(ious))
        history["train_loss"].append(train_l)
        history["val_loss"].append(val_l)
        history["val_iou"].append(val_iou)

        elapsed = time.perf_counter() - t0
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_l:.4f}  val_loss={val_l:.4f}  val_iou={val_iou:.4f}  "
            f"elapsed={elapsed:.1f}s"
        )

        if val_l < best_val:
            best_val = val_l
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "history": history,
                    "n_mics": dataset.n_mics,
                },
                ckpt_dir / "best.pt",
            )

    # Always save the final checkpoint too, even if val_loss has plateaued.
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": args.epochs,
            "history": history,
            "n_mics": dataset.n_mics,
        },
        ckpt_dir / "final.pt",
    )
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ---- loss plot -----------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    epochs = list(range(1, args.epochs + 1))
    ax1.plot(epochs, history["train_loss"], label="train loss", color="#2f5fa6")
    ax1.plot(epochs, history["val_loss"], label="val loss", color="#a64f2f")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE + Dice")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_iou"], label="val IoU", color="#3a7a3a", linestyle="--")
    ax2.set_ylabel("IoU")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(ckpt_dir / "loss.png", dpi=120)
    print(f"[train] wrote {ckpt_dir / 'loss.png'}")


if __name__ == "__main__":
    main()
