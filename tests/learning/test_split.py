"""Room-level train/val split (plan 3.4.1): no room may straddle the split."""

from __future__ import annotations

from acoustic_system.learning.train import room_level_split


def test_room_level_split_is_disjoint_and_deterministic() -> None:
    train, val = room_level_split(1000, 0.1, seed=42)
    assert len(val) == 100 and len(train) == 900
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(1000))
    assert room_level_split(1000, 0.1, seed=42) == (train, val)


def test_flattened_pose_indices_never_share_a_room() -> None:
    k = 4
    train, val = room_level_split(250, 0.1, seed=0)
    tr = {i // k for i in (r * k + p for r in train for p in range(k))}
    va = {i // k for i in (r * k + p for r in val for p in range(k))}
    assert not tr & va
