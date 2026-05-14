"""Loss functions for the obstacle-mask CNN.

The combined BCE + soft-Dice loss is the standard recipe for binary
segmentation: BCE keeps the per-pixel signal alive when overlap is
small (the gradient on the wrong-class pixels stays informative),
while Dice handles the class imbalance that comes from rooms being
mostly empty (the obstacle fraction is typically 5-15% of cells, so
a uniform "predict no obstacle" baseline scores high BCE accuracy
but is useless).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Binary cross-entropy + soft-Dice on a single-channel mask.

    Mathematical definitions
    ------------------------
    Let $\\hat{y} = \\sigma(\\ell)$ be the per-pixel predicted probability
    from logit $\\ell$ and $y \\in \\{0, 1\\}$ the target. Then

    $$ \\mathcal{L}_{\\text{BCE}} = -\\frac{1}{N} \\sum_{i} \\bigl[
       y_i \\log \\hat{y}_i + (1 - y_i) \\log (1 - \\hat{y}_i)
       \\bigr], $$

    $$ \\mathcal{L}_{\\text{Dice}} = 1 - \\frac{2 \\sum_i \\hat{y}_i y_i + \\epsilon}{
       \\sum_i \\hat{y}_i + \\sum_i y_i + \\epsilon}, $$

    and the returned loss is

    $$ \\mathcal{L} = \\mathcal{L}_{\\text{BCE}} + \\lambda\\,
       \\mathcal{L}_{\\text{Dice}}. $$

    Implementation notes
    --------------------
    - ``binary_cross_entropy_with_logits`` is numerically stable (uses the
      log-sum-exp identity), preferred over ``sigmoid`` followed by
      ``binary_cross_entropy``.
    - The Dice term is computed per-sample (sum across spatial dims, keep
      the batch dim) and then averaged so each scene contributes equally
      regardless of obstacle count.
    - ``eps`` prevents the 0/0 case when the prediction *and* target are
      both empty (a room with zero obstacles passes through cleanly).
    """
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if target.dtype != logits.dtype:
        target = target.to(dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(logits, target)

    probs = torch.sigmoid(logits)
    dims = (2, 3)
    intersection = (probs * target).sum(dim=dims)
    sum_p = probs.sum(dim=dims)
    sum_t = target.sum(dim=dims)
    dice_per_sample = (2.0 * intersection + eps) / (sum_p + sum_t + eps)
    dice_loss = (1.0 - dice_per_sample).mean()

    return bce + dice_weight * dice_loss


def iou_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean IoU across the batch, thresholded at ``threshold`` on the
    sigmoided logits.

    Returned as a scalar tensor so the caller can keep it on-device. To
    log as a Python float, call ``.item()``.
    """
    if target.dim() == 3:
        target = target.unsqueeze(1)
    preds = (torch.sigmoid(logits) > threshold).to(dtype=target.dtype)
    dims = (2, 3)
    intersection = (preds * target).sum(dim=dims)
    union = preds.sum(dim=dims) + target.sum(dim=dims) - intersection
    iou_per_sample = (intersection + eps) / (union + eps)
    return iou_per_sample.mean()
