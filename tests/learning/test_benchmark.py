"""Benchmark scorer (plan 10.10) on tiny synthetic archives."""

from __future__ import annotations

import importlib.util
import pathlib

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
assert spec and spec.loader
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def _archive(path: pathlib.Path, masks: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        for k, m in enumerate(masks):
            f.create_group(f"sample_{k:04d}").create_dataset("obstacles", data=m.astype(np.uint8))


def test_baseline_scores_zero_delta_and_an_oracle_wins(tmp_path):
    rng = np.random.default_rng(0)
    train = (rng.random((40, 16, 16)) < 0.2).astype(np.uint8)
    held = (rng.random((12, 16, 16)) < 0.2).astype(np.uint8)
    tp, hp = tmp_path / "train.h5", tmp_path / "held.h5"
    _archive(tp, train)
    _archive(hp, held)
    out = tmp_path / "prior.npz"
    assert (
        bench.main(["baseline", "--out", str(out), "--train", str(tp), "--heldout", str(hp)]) == 0
    )
    res = bench.score(out, tp, hp)
    for m in res["metrics"].values():
        assert abs(m["delta"]) < 1e-12
    # An oracle submission (the truth, slightly softened) beats the baseline on every metric.
    oracle = tmp_path / "oracle.npz"
    np.savez(oracle, prob=np.clip(held * 0.9 + 0.05, 0, 1), threshold=0.5, poses=1)
    res = bench.score(oracle, tp, hp)
    assert res["metrics"]["iou"]["mean"] == 1.0
    for k in ("iou", "ap", "bf", "info_bits"):
        assert res["metrics"][k]["delta"] > 0
        assert res["metrics"][k]["z"] > 3
