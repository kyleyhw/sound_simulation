"""Gates for the calibration module (Task 2.3e).

Plain script (no pytest), exits non-zero on failure.

What is tested and why:

1. **Parameter recovery**: generate synthetic logits whose true
   calibration is known — labels drawn as Bernoulli(sigma(l/T* + b*))
   — and check the fit recovers (T*, b*) closely. This validates the
   optimisation end to end rather than just "loss went down".
2. **Fusion identity**: ``calibrated_bayes_fuse`` with T=1, b=0 must
   reproduce the raw 2.1.4b product rule exactly (the no-sidecar
   path must be bit-compatible with the published benchmarks).
3. **Sidecar roundtrip**: save/load preserves values; loading from a
   directory without a sidecar returns None.

Test data rationale: 200k synthetic pixels with T*=2.5, b*=-0.8 —
a strongly miscalibrated regime (overconfident by 2.5x with an offset)
where a broken fit would visibly fail; the recovery tolerance (5%)
reflects Bernoulli sampling noise at this sample size.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.learning.calibration import (  # noqa: E402
    calibrated_bayes_fuse,
    fit_temperature_bias,
    load_calibration,
    save_calibration,
)


def main() -> None:
    rng = np.random.default_rng(0)

    # 1. Parameter recovery on synthetic data with known truth.
    t_true, b_true = 2.5, -0.8
    logits = rng.normal(0.0, 4.0, size=200_000).astype(np.float32)
    p_true = 1.0 / (1.0 + np.exp(-(logits / t_true + b_true)))
    labels = (rng.uniform(size=logits.size) < p_true).astype(np.float32)
    t_fit, b_fit = fit_temperature_bias(logits, labels)
    if abs(t_fit - t_true) / t_true > 0.05 or abs(b_fit - b_true) > 0.1:
        print(f"FAIL: recovered T={t_fit:.3f} b={b_fit:.3f}, true T={t_true} b={b_true}")
        sys.exit(1)
    print(f"OK: recovered T={t_fit:.4f} (true 2.5), b={b_fit:.4f} (true -0.8) on 200k pixels")

    # 2. Identity: T=1, b=0 must equal the raw product rule.
    k, h, w = 4, 16, 16
    lmaps = rng.normal(0.0, 2.0, size=(k, h, w)).astype(np.float32)
    prior = 0.0582
    prior_logit = np.log(prior / (1 - prior))
    raw = 1.0 / (1.0 + np.exp(-(lmaps.sum(axis=0) - (k - 1) * prior_logit)))
    fused = calibrated_bayes_fuse(lmaps, prior, temperature=1.0, bias=0.0)
    if not np.allclose(fused, raw, atol=1e-6):
        print("FAIL: identity calibration does not reproduce the raw Bayes rule")
        sys.exit(1)
    print("OK: T=1, b=0 fusion identical to the raw 2.1.4b product rule")

    # 3. Sidecar roundtrip + absent-file behaviour.
    with tempfile.TemporaryDirectory() as td:
        fake_ckpt = Path(td) / "best_iou.pt"
        fake_ckpt.write_bytes(b"")
        assert load_calibration(fake_ckpt) is None
        save_calibration(fake_ckpt, 1.7, -0.2, 0.061)
        got = load_calibration(fake_ckpt)
        assert got is not None and abs(got["temperature"] - 1.7) < 1e-9
        assert abs(got["bias"] + 0.2) < 1e-9 and abs(got["prior"] - 0.061) < 1e-9
    print("OK: calibration.json save/load roundtrip; absent sidecar -> None")

    print()
    print("all calibration checks passed")


if __name__ == "__main__":
    main()
