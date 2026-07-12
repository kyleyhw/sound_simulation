"""Correctness gates for the sense -> infer -> fuse pipeline.

Plain script (no pytest), exits non-zero on failure — same convention
as ``test_model.py``. Uses the committed joint checkpoint; skips with
a message (exit 0) if no checkpoint is present, so the gate does not
fail on a fresh clone that has not trained anything yet.

What is tested and why:

- **Shapes and ranges**: the pipeline's contract with its two
  consumers (demo script, web UI) is the ``SenseResult`` layout.
- **Determinism**: same seed => identical result; different seed =>
  different poses. Reproducibility is load-bearing for reports.
- **Fusion math at K=1**: with one pose the Bayes correction term
  vanishes, so ``fused_probs[0]`` must equal sigmoid(logits[0])
  exactly — a direct check that the product rule is implemented as
  documented.
- **Geometry legality**: drivers and mics must land on free interior
  cells (the placement helpers' contract, re-checked end to end).

Test room: a fixed three-rectangle 64x64 layout (built by the same
generator as training data, seed 20260712) rather than a random one,
so failures reproduce; K=3 keeps the runtime ~10 s on CPU while still
exercising the multi-pose accumulation path (K=1 tests none of it,
K=2 only one accumulation step).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CHECKPOINT = ROOT / "checkpoints" / "joint_baseline" / "best_iou.pt"


def main() -> None:
    if not CHECKPOINT.exists():
        print(f"SKIP: no checkpoint at {CHECKPOINT} — train one first (docs/learning.md)")
        return

    from acoustic_system.learning.sensing import GRID, sense_room
    from acoustic_system.simulation.dataset import generate_random_obstacles

    room = generate_random_obstacles(
        grid_shape=(GRID, GRID),
        n_obstacles=3,
        min_size=4,
        max_size=14,
        rng=np.random.default_rng(20260712),
    )
    k = 3

    r1 = sense_room(room, CHECKPOINT, n_poses=k, seed=7)
    r2 = sense_room(room, CHECKPOINT, n_poses=k, seed=7)
    r3 = sense_room(room, CHECKPOINT, n_poses=k, seed=8)

    # Shapes and ranges
    assert r1.logits.shape == (k, GRID, GRID), r1.logits.shape
    assert r1.fused_probs.shape == (k, GRID, GRID)
    assert len(r1.ious) == k
    assert all(0.0 <= v <= 1.0 for v in r1.ious), r1.ious
    assert float(r1.fused_probs.min()) >= 0.0 and float(r1.fused_probs.max()) <= 1.0
    assert r1.driver_positions.shape == (k, 2) and r1.mic_positions.shape == (k, 2, 2)
    print(f"OK: shapes/ranges (K={k}, ious={[round(v, 4) for v in r1.ious]})")

    # Determinism
    assert np.array_equal(r1.logits, r2.logits) and r1.ious == r2.ious
    assert not np.array_equal(r1.driver_positions, r3.driver_positions)
    print("OK: seeded determinism (seed 7 == seed 7, seed 7 != seed 8)")

    # Bayes fusion at K=1 reduces to plain sigmoid
    sig = 1.0 / (1.0 + np.exp(-r1.logits[0]))
    assert np.allclose(r1.fused_probs[0], sig, atol=1e-6)
    print("OK: K=1 fusion == sigmoid(logits[0]) (prior term vanishes)")

    # Geometry legality: no driver or mic inside an obstacle or wall
    mask = r1.truth > 0.5
    for pos in list(map(tuple, r1.driver_positions)) + [
        tuple(m) for pose in r1.mic_positions for m in pose
    ]:
        assert 0 < pos[0] < GRID - 1 and 0 < pos[1] < GRID - 1, pos
        assert not mask[pos], f"pose element {pos} inside an obstacle"
    print("OK: all pose geometry on free interior cells")

    print()
    print("all sensing-pipeline checks passed")


if __name__ == "__main__":
    main()
