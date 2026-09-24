"""Sensing metrics that do not depend on a decision threshold (plan 3.4.6).

The Phase 2 reports scored obstacle maps by mean IoU at a threshold. The
plan audit showed why that is not enough on its own: a predictor that
never hears the audio (the per-pixel training prior, thresholded) reaches
the published IoU. The functions here quantify what a prediction adds
*beyond the prior*.

All maps are ``(H, W)`` float arrays; truths are binary ``(H, W)``.

Information gain over a prior
-----------------------------
For a probabilistic prediction $q(x) = P(M_x = 1 \\mid \\text{audio})$ and
a no-audio prior $\\pi(x)$, the log-likelihood gain on a room with truth
$y$ is

$$ \\Delta = \\sum_x \\Bigl[\\log_2 q(x)^{y_x} (1-q(x))^{1-y_x}
            - \\log_2 \\pi(x)^{y_x} (1-\\pi(x))^{1-y_x}\\Bigr] \\text{ bits}. $$

Averaged over rooms, $\\mathbb{E}[\\Delta]$ estimates the information the
recording provides about the mask beyond the prior (a lower bound on
$I(M; \\text{audio})$, attained only by the true posterior). Positive
means the model beats the prior; a model that merely reproduces the prior
scores 0; miscalibrated confidence scores negative. It is proper, so it
cannot be gamed by shifting a threshold.

Average precision
-----------------
Area under the precision-recall curve of the per-pixel ranking: which
cells the model considers more likely than others. It is invariant to any
monotone recalibration, so it isolates *ranking* skill from calibration.

Boundary F-score
----------------
Obstacle interiors are acoustically invisible (p = 0 inside), so a
surface-level score matters: boundary cells (mask minus its 4-neighbour
erosion) of prediction and truth are matched within a tolerance of
``tol`` cells (Chebyshev), giving precision, recall and their harmonic
mean.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-6


def iou(pred_bin: NDArray, truth: NDArray, eps: float = EPS) -> float:
    """Per-room IoU with the project-wide (I + eps) / (U + eps) convention."""
    p = np.asarray(pred_bin, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    inter = float((p & t).sum())
    union = float((p | t).sum())
    return (inter + eps) / (union + eps)


def info_gain_bits(prob: NDArray, truth: NDArray, prior: NDArray | float) -> float:
    """Log-likelihood gain (bits per room) of ``prob`` over ``prior``."""
    q = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    pi = np.clip(np.broadcast_to(np.asarray(prior, dtype=np.float64), q.shape), 1e-6, 1 - 1e-6)
    y = np.asarray(truth, dtype=bool)
    ll_q = np.where(y, np.log2(q), np.log2(1 - q)).sum()
    ll_pi = np.where(y, np.log2(pi), np.log2(1 - pi)).sum()
    return float(ll_q - ll_pi)


def average_precision(score: NDArray, truth: NDArray) -> float:
    """Per-room average precision; NaN for a room with no positive cell."""
    s = np.asarray(score, dtype=np.float64).ravel()
    y = np.asarray(truth, dtype=bool).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y.size + 1)
    return float(precision[y_sorted].sum() / n_pos)


def boundary(mask: NDArray) -> NDArray[np.bool_]:
    """Cells of ``mask`` with at least one 4-neighbour outside the mask."""
    m = np.asarray(mask, dtype=bool)
    pad = np.pad(m, 1, constant_values=False)
    interior = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return m & ~interior


def _dilate(mask: NDArray[np.bool_], r: int) -> NDArray[np.bool_]:
    out = mask.copy()
    if r <= 0:
        return out
    pad = np.pad(mask, r, constant_values=False)
    h, w = mask.shape
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            out |= pad[r + di : r + di + h, r + dj : r + dj + w]
    return out


def boundary_f(pred_bin: NDArray, truth: NDArray, tol: int = 1) -> tuple[float, float, float]:
    """(precision, recall, F) of predicted vs true boundary cells within ``tol``."""
    bp = boundary(pred_bin)
    bt = boundary(truth)
    if not bp.any() and not bt.any():
        return 1.0, 1.0, 1.0
    if not bp.any() or not bt.any():
        return 0.0, 0.0, 0.0
    precision = float((bp & _dilate(bt, tol)).sum() / bp.sum())
    recall = float((bt & _dilate(bp, tol)).sum() / bt.sum())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f


def skill(score: float, baseline: float) -> float:
    """Fraction of the gap to a perfect 1.0 closed relative to ``baseline``."""
    return (score - baseline) / (1.0 - baseline) if baseline < 1.0 else 0.0


def mean_se(values: list[float] | NDArray) -> tuple[float, float]:
    """Mean and standard error, ignoring NaNs."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
    return float(v.mean()), se


def paired_diff_se(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mean and SE of the per-room paired difference a - b (NaN-safe)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return mean_se(d)
