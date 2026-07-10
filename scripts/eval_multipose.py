"""Bayesian multi-pose aggregation eval for the single-pose CNN (Task 2.1.4b).

Runs a *single-pose* checkpoint independently on each of a room's K
poses and fuses the per-pose probability maps into one obstacle-mask
estimate, WITHOUT any retraining. This is the cheap decision-gate
experiment: if fusing K poses lifts held-out IoU above the single-pose
baseline, the under-determination diagnosis from
``tests/reports/training_2026_05_14_aug.md`` is confirmed and the
joint-pose model (2.1.4c) is worth building.

Mathematical formulation
------------------------
Let $\\ell_k(x)$ be the model's per-pixel logit for pose $k$, so
$p_k(x) = \\sigma(\\ell_k(x))$ approximates
$P(M_x = 1 \\mid \\mathbf{s}_k, u)$. Three fusion rules are evaluated:

1. **Geometric-mean pooling** (``geo``, the report's recommendation).
   The odds-normalised geometric mean of the $p_k$ equals logit
   averaging:

   $$ p_{\\text{geo}}(x) = \\sigma\\Bigl(\\tfrac{1}{K} \\sum_k \\ell_k(x)\\Bigr). $$

   This is a product-of-experts with each expert down-weighted by
   $1/K$; it sharpens agreement and is prior-neutral (the implicit
   prior stays whatever single-pose training baked in).

2. **Bayes product rule** (``bayes``). If poses are conditionally
   independent given the mask and the model were calibrated,

   $$ \\operatorname{logit} P(M_x \\mid \\{\\mathbf{s}_k\\}, u)
      = \\sum_k \\ell_k(x) - (K - 1)\\, \\operatorname{logit}(\\pi_x), $$

   where $\\pi_x$ is the prior $P(M_x = 1)$. We use the scalar
   $\\hat\\pi$ = mean obstacle fraction over the eval archive's masks —
   the same marginal the single-pose model empirically converges to.
   Exact posterior fusion under its assumptions, but sensitive to
   miscalibration (the sum of K logits amplifies confidence K-fold).

3. **Arithmetic-mean pooling** (``mix``): $p = \\tfrac1K \\sum_k p_k$ —
   a mixture rather than a product; robust but never sharper than its
   inputs. Included as the robustness comparison.

Every rule is evaluated for each requested K (prefix subsets of the
archive's pose axis, so K=1 reproduces the single-pose baseline on the
same rooms) and reports mean IoU at the given threshold. Output is one
grep-able line per (rule, K):

    MULTIPOSE rule=<geo|bayes|mix> K=<int> mean_iou=<float> n_rooms=<int>

plus an IoU-vs-K curve figure and a qualitative prediction grid.

Usage::

    python scripts/eval_multipose.py \\
        --checkpoint checkpoints/long_baseline/best_iou.pt \\
        --dataset data/training_data/active_sensing_heldout_multipose_500x8.hdf5 \\
        --poses 1 2 4 8 \\
        --output-dir tests/reports/multipose_eval_artifacts
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from acoustic_system.learning.model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="Multi-pose HDF5 archive (K > 1).")
    p.add_argument("--poses", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--target-size", type=int, default=None, help="Defaults to checkpoint's.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-plot", type=int, default=6, help="Rooms in the qualitative grid.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize preserving the binary mask (as in training)."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return torch.nn.functional.interpolate(t, size=(size, size), mode="nearest")[0, 0].numpy()


def iou(pred_bin: np.ndarray, truth: np.ndarray, eps: float = 1e-6) -> float:
    inter = float((pred_bin * truth).sum())
    union = float(pred_bin.sum() + truth.sum() - inter)
    return (inter + eps) / (union + eps)


def fuse(logits: np.ndarray, rule: str, prior_logit: float) -> np.ndarray:
    """Fuse per-pose logit maps (K, H, W) -> probability map (H, W)."""
    k = logits.shape[0]
    if rule == "geo":
        fused_logit = logits.mean(axis=0)
    elif rule == "bayes":
        fused_logit = logits.sum(axis=0) - (k - 1) * prior_logit
    elif rule == "mix":
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.mean(axis=0)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return 1.0 / (1.0 + np.exp(-fused_logit))


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_mics = int(ckpt.get("n_mics", 2))
    target_size = int(args.target_size or ckpt["args"].get("target_size", 64))
    model = build_model(str(ckpt.get("model_type", "dual")), n_mics=n_mics).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with h5py.File(args.dataset, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        k_avail = int(f.attrs.get("poses_per_room", 1))
        if k_avail < 2:
            raise SystemExit("dataset is single-pose; regenerate with --poses-per-room K")
        ks = sorted({k for k in args.poses if 1 <= k <= k_avail})

        # Scalar prior for the Bayes rule: mean obstacle fraction of the
        # (resized) truth masks — the marginal the single-pose model
        # empirically fits. Computed over the whole archive first.
        fractions = []
        for key in keys:
            m = resize_mask(np.asarray(f[key]["obstacles"], dtype=np.float32), target_size)
            fractions.append(float(m.mean()))
        pi = float(np.clip(np.mean(fractions), 1e-4, 1 - 1e-4))
        prior_logit = float(np.log(pi / (1.0 - pi)))
        print(f"[multipose] prior pi={pi:.4f} (logit {prior_logit:.3f}), K available={k_avail}")

        rules = ("geo", "bayes", "mix")
        ious: dict[tuple[str, int], list[float]] = {(r, k): [] for r in rules for k in ks}
        plot_rows: list[dict[str, np.ndarray]] = []

        with torch.no_grad():
            for room_idx, key in enumerate(keys):
                grp = f[key]
                sensor = np.asarray(grp["sensor"], dtype=np.float32)  # (K, T, M)
                source = np.asarray(grp["source"], dtype=np.float32)
                truth = resize_mask(np.asarray(grp["obstacles"], dtype=np.float32), target_size)

                # One batched forward for all K poses of this room.
                sens_t = torch.from_numpy(sensor.transpose(0, 2, 1).copy()).to(device)
                src_t = (
                    torch.from_numpy(source.copy())
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(sensor.shape[0], 1, 1)
                    .to(device)
                )
                logits = model(sens_t, src_t).cpu().numpy()[:, 0]  # (K, H, W)

                for rule in rules:
                    for k in ks:
                        p = fuse(logits[:k], rule, prior_logit)
                        ious[(rule, k)].append(iou((p > args.threshold).astype(np.float32), truth))

                if room_idx < args.n_plot:
                    plot_rows.append(
                        {
                            "truth": truth,
                            "single": 1.0 / (1.0 + np.exp(-logits[0])),
                            "fused": fuse(logits[: max(ks)], "geo", prior_logit),
                        }
                    )

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- summary table -------------------------------------------------
    for rule in rules:
        for k in ks:
            vals = ious[(rule, k)]
            print(f"MULTIPOSE rule={rule} K={k} mean_iou={np.mean(vals):.4f} n_rooms={len(vals)}")

    # ---- IoU vs K curves -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for rule, marker in zip(rules, ("o", "s", "^")):
        ax.plot(ks, [np.mean(ious[(rule, k)]) for k in ks], marker=marker, label=rule)
    ax.set_xlabel("poses fused (K)")
    ax.set_ylabel(f"mean IoU @ {args.threshold}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend(title="fusion rule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "iou_vs_k.png", dpi=120)

    # ---- qualitative grid ------------------------------------------------
    n = len(plot_rows)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
    if n == 1:
        axes = np.array([axes])
    for row, panels in enumerate(plot_rows):
        for col, (title, img, cmap) in enumerate(
            (
                ("truth", panels["truth"], "gray"),
                ("single-pose prob", panels["single"], "viridis"),
                (f"geo-fused prob (K={max(ks)})", panels["fused"], "viridis"),
            )
        ):
            axes[row, col].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[row, col].set_title(title if row == 0 else "")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "preds_multipose.png", dpi=120)

    print(f"[multipose] wrote {out_dir}\\iou_vs_k.png and preds_multipose.png")
    print(f"[multipose] total runtime {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
