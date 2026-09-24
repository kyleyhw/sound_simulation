"""FxLMS noise cancellation against the engine (plan 7.4)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.control.anc import band_noise, fxlms, quiet_zone, stride_for, tone
from acoustic_system.control.transfer import Room, dtft, measure_transfer

PRIMARY, SECONDARY, MIC = (0.5, 0.5), (2.3, 1.5), (2.0, 1.5)


@pytest.fixture(scope="module")
def plant(make_room):
    room = make_room(0.3)
    ts = measure_transfer(room, [SECONDARY], [MIC], duration=0.15)
    return room, ts, ts.impulse_responses(n=2048)[0, 0]


def test_secondary_path_estimate(plant) -> None:
    _, ts, s_hat = plant
    f = np.arange(150.0, 1500.0, 25.0)
    ratio = dtft(s_hat, f, ts.dt) / ts.at(f)[:, 0, 0]
    assert np.all(np.abs(np.degrees(np.angle(ratio))) < 30.0)  # FxLMS needs < 90
    assert np.all(np.abs(np.abs(ratio) - 1) < 0.3)


def test_fxlms_tone_converges_and_makes_a_quiet_zone(plant) -> None:
    room, ts, s_hat = plant
    diam = {}
    for f in (300.0, 1000.0):
        x = tone(f, ts.dt, 12000)
        r = fxlms(room, PRIMARY, SECONDARY, MIC, x, s_hat, n_taps=8,
                  stride=stride_for(f, ts.dt), mu=0.1, measure_steps=3000)  # fmt: skip
        assert r.attenuation_db > 30.0, (f, r.attenuation_db)
        assert r.learning_db[-1] > r.learning_db[0] + 20.0
        assert r.att_map is not None
        qz = quiet_zone(r.att_map, room, MIC)
        assert qz.mask[tuple(room.cells([MIC])[0])]
        lam = room.c / f
        assert 0.0 < qz.diameter_m < lam / 2
        diam[f] = qz.diameter_m
    assert diam[300.0] > 2 * diam[1000.0]  # the zone shrinks with wavelength


def test_fxlms_broadband_noise(plant) -> None:
    room, ts, s_hat = plant
    x = band_noise((100.0, 500.0), ts.dt, 40000, seed=0)
    r = fxlms(room, PRIMARY, SECONDARY, MIC, x, s_hat, n_taps=96,
              stride=stride_for(500.0, ts.dt), mu=0.2, measure_steps=8000, field=False)  # fmt: skip
    assert r.attenuation_db > 10.0


def test_quiet_zone_region() -> None:
    room = Room(size=(1.0, 1.0))
    att = np.zeros(room.shape)
    att[18:23, 19:22] = 12.0  # 5 x 3 cells around (20, 20)
    att[30:35, 30:35] = 20.0  # disconnected
    qz = quiet_zone(att, room, (0.5, 0.5))
    assert qz.mask.sum() == 15
    assert qz.area_m2 == pytest.approx(15 * room.dx**2)
    assert qz.extent_m == pytest.approx((5 * room.dx, 3 * room.dx))
