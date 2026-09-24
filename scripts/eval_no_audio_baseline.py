"""No-audio baselines for the obstacle-mask IoU metric (plan audit, 2026-09-24).

Every sensing number in the Phase 2 reports is an IoU between a
predicted and a true obstacle mask. This script measures what that
metric returns for predictors that **never look at the audio**, so a
sensing result can be read as "information gained from sound" rather
than as an absolute number:

1. *predict-all* — every cell occupied. Per-room IoU equals the room's
   occupancy fraction, so the mean equals mean occupancy.
2. *prior map* — the per-pixel marginal $\\hat\\pi(x) = P(M_x = 1)$
   estimated from training masks, thresholded at the IoU-optimal
   $\\tau$ (an oracle choice, i.e. an upper bound for this family).
3. *interior box* — a fixed square that excludes a border band of
   width $k$ (the generators keep obstacles out of a margin, so the
   centre is where occupancy mass lives); best $k$ reported.

A sensing model whose held-out IoU does not exceed (2) has not
demonstrated that it extracts geometry from the recording.

IoU is the per-room mean, identical to ``learning.losses.iou_score``
and ``scripts/eval_multipose.py`` (``eps = 1e-6``).

Usage::

    # masks drawn fresh from the generators (no archive needed)
    python scripts/eval_no_audio_baseline.py --room-style mixed
    python scripts/eval_no_audio_baseline.py --room-style rect

    # or score against the exact masks of existing archives
    python scripts/eval_no_audio_baseline.py \\
        --train-archive data/training_data/active_sensing_v2_train_10kx4.hdf5 \\
        --heldout-archive data/training_data/active_sensing_v2_heldout_500x8.hdf5
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from acoustic_system.simulation.dataset import (  # noqa: E402
    generate_diverse_obstacles,
    generate_random_obstacles,
)

EPS = 1e-6


def per_room_iou(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-room IoU of a fixed (H, W) prediction against (N, H, W) truths."""
    inter = (truth & pred).sum(axis=(1, 2))
    union = (truth | pred).sum(axis=(1, 2))
    return (inter + EPS) / (union + EPS)


def sample_masks(style: str, n: int, grid: int, seed: int) -> np.ndarray:
    """Draw masks with the dataset generators' documented v1/v2 settings."""
    rng = np.random.default_rng(seed)
    if style == "mixed":
        # v2 archives: --room-style mixed, sizes 4-14 (7.0 % mean occupancy)
        masks = [generate_diverse_obstacles((grid, grid), rng=rng) for _ in range(n)]
    else:
        # v1 archives: --n-obstacles 3 --obstacle-min 4 --obstacle-max 14
        masks = [
            generate_random_obstacles((grid, grid), n_obstacles=3, min_size=4, max_size=14, rng=rng)
            for _ in range(n)
        ]
    return np.stack(masks).astype(bool)


def read_masks(path: pathlib.Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as hf:
        keys = sorted(k for k in hf.keys() if k.startswith("sample_"))
        return np.stack([hf[k]["obstacles"][()] for k in keys]).astype(bool)


def summarise(name: str, ious: np.ndarray) -> str:
    se = ious.std(ddof=1) / np.sqrt(len(ious))
    return f"{name:<34s} IoU = {ious.mean():.4f} +- {se:.4f} (SE, n={len(ious)})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--room-style", choices=["mixed", "rect"], default="mixed")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-heldout", type=int, default=500)
    ap.add_argument("--train-seed", type=int, default=1)
    ap.add_argument("--heldout-seed", type=int, default=2)
    ap.add_argument("--train-archive", type=pathlib.Path, default=None)
    ap.add_argument("--heldout-archive", type=pathlib.Path, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.train_archive is not None:
        train = read_masks(args.train_archive)
    else:
        train = sample_masks(args.room_style, args.n_train, args.grid, args.train_seed)
    if args.heldout_archive is not None:
        held = read_masks(args.heldout_archive)
    else:
        held = sample_masks(args.room_style, args.n_heldout, args.grid, args.heldout_seed)

    print(f"train masks {train.shape}, held-out masks {held.shape}")
    print(f"held-out mean occupancy {held.mean():.4f}")
    print(summarise("predict-all", per_room_iou(np.ones(held.shape[1:], bool), held)))

    prior = train.mean(axis=0)
    best_prior = max(
        ((per_room_iou(prior >= t, held), t) for t in np.linspace(0.0, prior.max(), 80)[1:]),
        key=lambda x: x[0].mean(),
    )
    print(summarise(f"prior map (oracle tau={best_prior[1]:.3f})", best_prior[0]))

    h, w = held.shape[1:]
    best_box = None
    for k in range(0, min(h, w) // 2 - 1):
        box = np.zeros((h, w), bool)
        box[k : h - k, k : w - k] = True
        ious = per_room_iou(box, held)
        if best_box is None or ious.mean() > best_box[0].mean():
            best_box = (ious, k)
    assert best_box is not None
    print(summarise(f"interior box (oracle k={best_box[1]})", best_box[0]))
    print(f"runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
