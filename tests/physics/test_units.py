"""SI scale conversions and the dataset plausibility check (plan 5.2)."""

from __future__ import annotations

import pytest

from acoustic_system.simulation.units import PhysicalScale, plausibility


def test_round_trips() -> None:
    sc = PhysicalScale(dx=0.02)
    assert sc.grid_frequency(sc.hertz(0.1)) == pytest.approx(0.1)
    assert sc.cells(sc.metres(64)) == pytest.approx(64)
    assert sc.seconds(343 / 0.02) == pytest.approx(1.0)
    assert PhysicalScale.for_band(4000, 8).max_frequency(8) == pytest.approx(4000)


def test_v2_protocol_is_not_a_laptop_in_a_room() -> None:
    # v2 archives: 64-cell grid, 12-cell mic baseline, chirp 0.02 -> 0.45.
    laptop = plausibility(64, 12, 0.02, 0.45, 400, mic_baseline_m=0.2)
    assert laptop.room_m == pytest.approx(64 * 0.2 / 12)  # ~1.07 m room
    assert any("room width" in n for n in laptop.notes)
    room = plausibility(64, 12, 0.02, 0.45, 400, dx=5.0 / 64)  # 5 m room
    assert any("mic baseline" in n for n in room.notes)
