"""Gates for the web UI's `sense_room` bridge (SimulationManager.sense).

Plain script (no pytest), exits non-zero on failure. Exercises the
manager method directly — the socket handler is a thin argument-
coercion wrapper around it — on three paths:

1. **Happy path**: a rectangle drawn into the default 200x200 room,
   sensed at K=2. Checks the payload contract the frontend renders
   from (ok, 64x64 prob/truth, ious length K, elapsed) and that the
   truth field really is the drawn rectangle after 200->64 resampling
   (the resize is the one lossy step in the bridge).
2. **3D guard**: sensing a 3D room must return ok=False with a
   message, not raise — the UI shows the error string.
3. **Missing-checkpoint guard**: same contract when the checkpoint
   path is wrong.

Why K=2: exercises the multi-pose accumulation with the smallest
runtime (~5 s including model load). Why a centred 40x40 rectangle:
after 200->64 nearest-neighbour resampling it stays a clean ~13-cell
square, so the truth-preservation check is exact.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import socketio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.app import main as app_main  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "joint_baseline" / "best_iou.pt"


def main() -> None:
    if not CHECKPOINT.exists():
        print(f"SKIP: no checkpoint at {CHECKPOINT} — train one first (docs/learning.md)")
        return

    mgr = app_main.SimulationManager(socketio.AsyncServer(async_mode="asgi"))

    # Draw a 40x40 rectangle in the 200x200 default room.
    assert mgr.simulation is not None
    cells = [(i, j) for i in range(80, 120) for j in range(80, 120)]
    mgr.simulation.set_obstacle(cells)

    res = asyncio.run(mgr.sense(n_poses=2, seed=11))
    assert res["ok"] is True, res
    assert res["shape"] == [64, 64] and len(res["prob"]) == 64 and len(res["prob"][0]) == 64
    assert res["poses"] == 2 and len(res["ious"]) == 2
    assert all(0.0 <= v <= 1.0 for row in res["prob"] for v in row)
    truth = np.asarray(res["truth"])
    # 200 -> 64 resampling maps the [80, 120) square to ~[26, 39): a
    # ~13x13 block. Check total mass and location rather than exact
    # cell identity (nearest-neighbour edge cells may round either way).
    assert 100 <= truth.sum() <= 200, truth.sum()
    assert truth[32, 32] == 1 and truth[5, 5] == 0
    print(
        f"OK: happy path — 64x64 prob map, K=2, ious={res['ious']}, "
        f"truth mass {int(truth.sum())} cells, {res['elapsed']}s"
    )

    # 3D guard
    async def _try_3d() -> dict:
        await mgr.configure({"grid_shape": [32, 32, 32]})
        return await mgr.sense(n_poses=1)

    res3d = asyncio.run(_try_3d())
    assert res3d["ok"] is False and "2D" in res3d["error"], res3d
    print(f"OK: 3D room rejected with message: {res3d['error']!r}")

    # Missing-checkpoint guard
    async def _try_missing() -> dict:
        await mgr.configure({"grid_shape": [64, 64]})
        return await mgr.sense(n_poses=1)

    original = app_main.SENSE_CHECKPOINT
    try:
        app_main.SENSE_CHECKPOINT = str(ROOT / "checkpoints" / "does_not_exist.pt")
        res_missing = asyncio.run(_try_missing())
    finally:
        app_main.SENSE_CHECKPOINT = original
    assert res_missing["ok"] is False and "checkpoint" in res_missing["error"], res_missing
    print(f"OK: missing checkpoint rejected with message: {res_missing['error']!r}")

    print()
    print("all sense_room bridge checks passed")


if __name__ == "__main__":
    main()
