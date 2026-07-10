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

# When stdout is piped to a file (e.g. background-task output), Python
# defaults to block-buffering the stream. That hides progress for many
# minutes during a long training run — even per-epoch prints with
# flush=True can be deferred. Force line buffering at import time so
# every print() lands on disk immediately. ``reconfigure`` exists on
# TextIOWrapper (the concrete stdout in normal CPython) but not on
# the abstract TextIO type, so guard with isinstance.
import io as _io  # noqa: E402
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from acoustic_system.learning.dataset import ActiveSensingDataset
from acoustic_system.learning.losses import bce_dice_loss, iou_score
from acoustic_system.learning.model import build_model

if isinstance(sys.stdout, _io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="HDF5 archive from generate_active_sensing.py")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 weight decay for AdamW. Helps generalisation on small datasets.",
    )
    p.add_argument(
        "--scheduler",
        choices=["cosine", "step", "none"],
        default="cosine",
        help="Learning-rate schedule: cosine annealing to 0, 10x step at 60%%, or constant.",
    )
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--target-size", type=int, default=64, help="Predicted mask side length.")
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout2d probability inside each spectrogram encoder. 0 disables.",
    )
    p.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Enable train-only augmentation: per-channel gain jitter in [0.7, 1.3] and "
            "additive Gaussian noise (sigma=0.02) on the sensor recordings. Held-out "
            "validation always sees clean signal so the metric measures generalisation, "
            "not robustness to the augmentation distribution."
        ),
    )
    p.add_argument(
        "--model",
        choices=["dual", "passive"],
        default="dual",
        help=(
            "dual = DualInputCNN (active sensing: sensor + known source). "
            "passive = PassiveCNN (Task 2.2 blind setting: sensor only; the "
            "dataset's source is ignored)."
        ),
    )
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
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print per-epoch line every N epochs. Set higher for long runs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Two parallel dataset views over the same HDF5 archive: the train
    # view applies augmentation (when --augment is set), the val view
    # always sees clean signal. Same per-call HDF5 open in both — there
    # is no extra I/O cost. The split below is by index so the two views
    # see disjoint samples.
    train_dataset = ActiveSensingDataset(
        args.dataset,
        target_mask_size=args.target_size,
        augment=args.augment,
    )
    val_dataset = ActiveSensingDataset(
        args.dataset,
        target_mask_size=args.target_size,
        augment=False,
    )
    n_total = len(train_dataset)
    n_val = max(1, int(args.val_frac * n_total))
    n_train = n_total - n_val
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds = torch.utils.data.Subset(val_dataset, val_indices)
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

    # PassiveCNN.forward accepts-and-ignores the source argument, so the
    # train/val loops below stay identical across model types.
    model = build_model(args.model, n_mics=train_dataset.n_mics, dropout=args.dropout).to(device)
    print(f"[train] model={args.model}  device={device}  params={model.num_params:,}")
    print(
        f"[train] dataset n_total={n_total} n_train={n_train} n_val={n_val} "
        f"n_mics={train_dataset.n_mics} T_rec={train_dataset.t_rec} T_audio={train_dataset.t_audio}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=int(args.epochs * 0.6), gamma=0.1
        )
    else:
        scheduler = None

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_iou": [],
        "val_loss": [],
        "val_iou": [],
        "lr": [],
    }
    best_val = float("inf")
    best_iou = 0.0

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        # ---- train ------------------------------------------------------
        model.train()
        train_losses: list[float] = []
        train_ious: list[float] = []
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
            with torch.no_grad():
                train_ious.append(iou_score(logits, mask).item())

        # ---- validate ---------------------------------------------------
        model.eval()
        val_losses: list[float] = []
        val_ious: list[float] = []
        with torch.no_grad():
            for sensor, source, mask in val_loader:
                sensor = sensor.to(device)
                source = source.to(device)
                mask = mask.to(device)
                logits = model(sensor, source)
                val_losses.append(bce_dice_loss(logits, mask).item())
                val_ious.append(iou_score(logits, mask).item())

        train_l = float(np.mean(train_losses))
        train_iou = float(np.mean(train_ious))
        val_l = float(np.mean(val_losses))
        val_iou = float(np.mean(val_ious))
        current_lr = opt.param_groups[0]["lr"]
        history["train_loss"].append(train_l)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_l)
        history["val_iou"].append(val_iou)
        history["lr"].append(current_lr)

        if scheduler is not None:
            scheduler.step()

        if epoch % args.log_every == 0 or epoch == args.epochs:
            elapsed = time.perf_counter() - t0
            print(
                f"epoch {epoch:3d}/{args.epochs}  "
                f"lr={current_lr:.2e}  "
                f"train_loss={train_l:.4f} train_iou={train_iou:.3f}  "
                f"val_loss={val_l:.4f} val_iou={val_iou:.3f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        if val_l < best_val:
            best_val = val_l
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": args.model,
                    "args": vars(args),
                    "epoch": epoch,
                    "history": history,
                    "n_mics": train_dataset.n_mics,
                },
                ckpt_dir / "best.pt",
            )
        if val_iou > best_iou:
            # Best-IoU checkpoint is more useful than best-loss for an
            # eval that scores by mask overlap. Save both.
            best_iou = val_iou
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": args.model,
                    "args": vars(args),
                    "epoch": epoch,
                    "history": history,
                    "n_mics": train_dataset.n_mics,
                },
                ckpt_dir / "best_iou.pt",
            )

    # Always save the final checkpoint too, even if val_loss has plateaued.
    torch.save(
        {
            "model": model.state_dict(),
            "model_type": args.model,
            "args": vars(args),
            "epoch": args.epochs,
            "history": history,
            "n_mics": train_dataset.n_mics,
        },
        ckpt_dir / "final.pt",
    )
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ---- diagnostic plot -----------------------------------------------
    epochs = list(range(1, args.epochs + 1))
    fig, (ax_loss, ax_iou) = plt.subplots(2, 1, figsize=(7.0, 5.5), sharex=True)
    ax_loss.plot(epochs, history["train_loss"], label="train loss", color="#2f5fa6")
    ax_loss.plot(epochs, history["val_loss"], label="val loss", color="#a64f2f")
    ax_loss.set_ylabel("BCE + Dice")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right")
    ax_lr = ax_loss.twinx()
    ax_lr.plot(epochs, history["lr"], label="lr", color="#888", linestyle=":", alpha=0.6)
    ax_lr.set_ylabel("learning rate", color="#888")
    ax_lr.tick_params(axis="y", labelcolor="#888")
    ax_lr.set_yscale("log")

    ax_iou.plot(epochs, history["train_iou"], label="train IoU", color="#2f5fa6")
    ax_iou.plot(epochs, history["val_iou"], label="val IoU", color="#a64f2f")
    ax_iou.set_xlabel("epoch")
    ax_iou.set_ylabel("IoU")
    ax_iou.set_ylim(0, 1)
    ax_iou.grid(True, alpha=0.3)
    ax_iou.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(ckpt_dir / "loss.png", dpi=120)
    print(f"[train] wrote {ckpt_dir / 'loss.png'}")
    print(
        f"[train] best val_loss={best_val:.4f}, best val_iou={best_iou:.4f} "
        f"(checkpoints: best.pt, best_iou.pt, final.pt)"
    )


if __name__ == "__main__":
    main()
