"""Post-hoc logit calibration for the Bayes pose-fusion rule (Task 2.3e).

Why calibration matters here
----------------------------
The Bayes product rule fuses per-pose logits as

$$ \\ell_{\\text{fused}} = \\sum_{k=1}^{K} \\ell_k - (K-1)\\,\\operatorname{logit}(\\hat\\pi), $$

which is exact only if each $\\ell_k$ is a *calibrated* posterior
log-odds. A miscalibrated model (over- or under-confident by a factor)
has its error amplified $K$-fold by the sum — empirically this is why
fused maps saturate into oversized blobs on out-of-distribution rooms.

The fix is Platt-style scalar recalibration: find temperature $T$ and
bias $b$ such that

$$ \\ell' = \\ell / T + b $$

minimises the binary cross-entropy of $\\sigma(\\ell')$ against the true
per-pixel labels on *validation* rooms (never training rooms — the
memorised split would fit an overconfident $T$). Two scalars cannot
overfit; they correct exactly the global confidence scale and offset
that the product rule is sensitive to. Fusion then uses

$$ \\ell_{\\text{fused}} = \\sum_k \\ell'_k - (K-1)\\,\\operatorname{logit}(\\hat\\pi), $$

with $\\hat\\pi$ the realised training-archive obstacle fraction (stored
in the checkpoint's acquisition attrs).

The fitted parameters are stored as ``calibration.json`` next to the
checkpoint ({"temperature", "bias", "prior"}); consumers
(``scripts/eval_multipose.py``, ``learning/sensing.py``) auto-load it
when present. An absent file means the identity calibration
$T = 1, b = 0$ — the pre-2.3e behaviour, bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from numpy.typing import NDArray


def fit_temperature_bias(
    logits: NDArray[np.float32],
    labels: NDArray[np.float32],
    max_iter: int = 200,
) -> tuple[float, float]:
    """Fit $(T, b)$ minimising BCE of $\\sigma(\\ell/T + b)$.

    Parameters
    ----------
    logits, labels
        Flattened per-pixel arrays of equal length. Labels in {0, 1}.
    max_iter
        L-BFGS iterations; two scalars converge in far fewer.

    Implementation: optimise $\\log T$ (so $T > 0$ by construction)
    and $b$ with L-BFGS on the full tensor — even 10^6 pixels is a
    trivial problem for two parameters.
    """
    x = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    log_t = torch.zeros(1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t, bias], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x / torch.exp(log_t) + bias, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item()), float(bias.item())


def save_calibration(
    ckpt_path: str | Path,
    temperature: float,
    bias: float,
    prior: float,
    threshold: float = 0.5,
) -> Path:
    """Write ``<ckpt_stem>.calibration.json`` next to the checkpoint.

    ``threshold`` is the model's operating point: the IoU-optimal
    decision threshold selected on the *validation* split (never
    held-out data). Scalar calibration is an affine monotone map of the
    fused logit, so it cannot change ranking metrics — choosing the
    operating point is the part of Task 2.3e that actually moves IoU,
    and it must travel with the calibration that defines its scale.

    The sidecar is per checkpoint file (audit 3.4.3). A single
    ``calibration.json`` per directory used to be applied to best.pt,
    best_iou.pt and final.pt alike, although it was fitted on one of
    them. The file also records the checkpoint's name and byte size so
    that a mismatch can be detected on load.
    """
    ck = Path(ckpt_path).resolve()
    out = sidecar_path(ck)
    out.write_text(
        json.dumps(
            {
                "temperature": temperature,
                "bias": bias,
                "prior": prior,
                "threshold": threshold,
                "checkpoint": ck.name,
                "checkpoint_bytes": ck.stat().st_size if ck.exists() else None,
            },
            indent=2,
        )
    )
    return out


def sidecar_path(ckpt_path: str | Path) -> Path:
    ck = Path(ckpt_path).resolve()
    return ck.with_name(f"{ck.stem}.calibration.json")


def load_calibration(ckpt_path: str | Path) -> Optional[dict[str, Any]]:
    """Return the checkpoint's calibration dict, or None.

    Looks for ``<stem>.calibration.json``. The pre-audit per-directory
    ``calibration.json`` is accepted only when the file itself names this
    checkpoint, or when the checkpoint is ``best_iou.pt`` (the only file
    the legacy sidecars were ever fitted on).
    """
    ck = Path(ckpt_path).resolve()
    p = sidecar_path(ck)
    if p.exists():
        return json.loads(p.read_text())
    legacy = ck.parent / "calibration.json"
    if legacy.exists():
        data = json.loads(legacy.read_text())
        if data.get("checkpoint", "best_iou.pt") == ck.name:
            return data
    return None


def calibrated_bayes_fuse(
    logits: NDArray[np.float32],
    prior: float,
    temperature: float = 1.0,
    bias: float = 0.0,
) -> NDArray[np.float32]:
    """Fuse per-pose logit maps ``(K, H, W)`` into one probability map.

    Applies the scalar calibration to each pose's logits, sums, and
    subtracts the prior once per extra pose (the Bayes product rule).
    With ``temperature=1, bias=0`` this reduces exactly to the
    uncalibrated rule used by the 2.1.4b experiments.
    """
    k = int(logits.shape[0])
    prior = float(np.clip(prior, 1e-4, 1.0 - 1e-4))
    prior_logit = float(np.log(prior / (1.0 - prior)))
    cal = np.asarray(logits, dtype=np.float64) / float(temperature) + float(bias)
    fused = cal.sum(axis=0) - (k - 1) * prior_logit
    return (1.0 / (1.0 + np.exp(-fused))).astype(np.float32)
