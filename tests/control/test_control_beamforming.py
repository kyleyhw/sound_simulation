"""Classical controllers and broadband FIR design (plan 7.2)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.control.beamforming import (
    acoustic_contrast_control,
    delay_and_sum,
    design_broadband,
    design_grid,
    fir_from_weights,
    fir_response,
    normalise_to_reference,
    pressure_matching,
    time_reversal,
    time_reversal_fir,
    verify_fir,
)
from acoustic_system.control.metrics import (
    array_effort_db,
    band_contrast_db,
    contrast_spectrum_db,
    reproduction_error,
    zone_energy,
)
from acoustic_system.control.transfer import bandpass_pulse

BAND = (300.0, 1500.0)
# Stated tolerance for "the time-domain engine run reproduces the
# frequency-domain prediction": the only differences are float32 round-off
# and truncation of the records at the 150 ms measurement length.
FIR_TOLERANCE_DB = 0.2


def _narrowband(zone_scene):
    _, ts, ib, id_ = zone_scene
    f = np.arange(300.0, 1501.0, 50.0)
    H = ts.at(f)
    return ts, f, H[:, ib], H[:, id_]


def test_acc_beats_other_controllers(zone_scene) -> None:
    ts, f, Hb, Hd = _narrowband(zone_scene)
    ref = 4
    acc = acoustic_contrast_control(Hb, Hd, reg=1e-3)
    pm = pressure_matching(Hb, Hd, Hb[:, :, ref], reg=1e-3)
    das = delay_and_sum(ts.speakers, f, focus=ts.points[0])
    tr = time_reversal(Hb[:, 0, :])
    c = {k: band_contrast_db(Hb, Hd, q) for k, q in dict(acc=acc, pm=pm, das=das, tr=tr).items()}
    assert c["acc"] >= 20.0, c  # the plan target is 10 dB
    assert c["acc"] > c["pm"] > c["das"], c
    # ACC is the per-frequency optimum of the (regularised) contrast.
    for q in (pm, das, tr):
        assert np.all(contrast_spectrum_db(Hb, Hd, acc) >= contrast_spectrum_db(Hb, Hd, q) - 0.5)
    # Pressure matching reproduces its bright-zone target.
    p = np.einsum("fms,fs->fm", Hb, pm)
    assert np.mean(reproduction_error(p, Hb[:, :, ref])) < 0.3


def test_time_reversal_maximises_focus_pressure(zone_scene) -> None:
    ts, f, Hb, _ = _narrowband(zone_scene)
    h = Hb[:, 0, :]
    rng = np.random.default_rng(0)
    tr = time_reversal(h)
    for q in (delay_and_sum(ts.speakers, f, focus=ts.points[0]), rng.standard_normal(tr.shape)):
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        assert np.all(np.abs(np.sum(h * tr, 1)) >= np.abs(np.sum(h * q, 1)) - 1e-12)


def test_delay_and_sum_geometry() -> None:
    spk = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    f = np.array([500.0])
    q = delay_and_sum(spk, f, focus=(0.0, 1.0))
    np.testing.assert_allclose(np.abs(q), 1 / np.sqrt(3))
    r = np.linalg.norm(spk - [0.0, 1.0], axis=1)
    # Phase advance of each speaker is its extra path: arrivals align at the focus.
    arrival = np.angle(q[0] * np.exp(-2j * np.pi * 500.0 * r / 343.0))
    np.testing.assert_allclose(np.angle(np.exp(1j * (arrival - arrival[0]))), 0.0, atol=1e-12)
    s = delay_and_sum(spk, f, direction=(1.0, 0.0))
    assert np.angle(s[0, 1] / s[0, 0]) == pytest.approx(-2 * np.pi * 500.0 * 0.1 / 343.0)


def test_fir_from_weights_is_a_delay() -> None:
    dt, L = 1e-4, 256
    bins, f = design_grid(L, dt, (500.0, 4000.0))
    tau0 = 10
    q = np.exp(-2j * np.pi * f * tau0 * dt)[:, None]
    taps = fir_from_weights(q, bins, L, dt, window="none", taper_bins=0)
    R = fir_response(taps, f[5:-5], dt)[:, 0]
    np.testing.assert_allclose(
        np.angle(R * np.exp(2j * np.pi * f[5:-5] * (tau0 + L / 2) * dt)), 0.0, atol=0.05
    )
    assert np.argmax(np.abs(taps[0])) == tau0 + L // 2


def test_normalisation_keeps_contrast_and_sets_effort(zone_scene) -> None:
    _, _, Hb, Hd = _narrowband(zone_scene)
    q = acoustic_contrast_control(Hb, Hd)
    n = normalise_to_reference(q, Hb, ref_speaker=4, ref_point=0)
    np.testing.assert_allclose(contrast_spectrum_db(Hb, Hd, n), contrast_spectrum_db(Hb, Hd, q))
    np.testing.assert_allclose(zone_energy(Hb, n), np.mean(np.abs(Hb[:, :, 4]) ** 2, -1))
    np.testing.assert_allclose(
        array_effort_db(n, Hb, 4), 10 * np.log10(np.sum(np.abs(n) ** 2, 1)), atol=1e-9
    )


def test_broadband_acc_verified_in_time_domain(zone_scene) -> None:
    """Headline (7.2.5): >= 10 dB broadband contrast, measured in the engine."""
    room, ts, ib, id_ = zone_scene
    d = design_broadband(ts, ib, id_, method="acc", band=BAND)
    x = bandpass_pulse(ts.dt, BAND, 511)
    chk = verify_fir(room, ts, d.taps, x, ib, id_, BAND)
    assert chk.measured_db >= 10.0
    assert chk.measured_db >= 20.0  # current value about 25 dB; guards regressions
    assert abs(chk.measured_db - chk.predicted_db) < FIR_TOLERANCE_DB
    # The FIR keeps most of the narrowband optimum (L = 2048 taps, 75 ms).
    assert np.mean(d.fir_db) > np.mean(d.ideal_db) - 6.0


def test_time_reversal_fir_focuses(zone_scene) -> None:
    room, ts, ib, id_ = zone_scene
    ir = ts.impulse_responses(n=2048)[:, ib[0]]
    taps = time_reversal_fir(ir, 2048)
    x = bandpass_pulse(ts.dt, BAND, 511)
    chk = verify_fir(room, ts, taps, x, ib, id_, BAND)
    assert chk.measured_db > 3.0
    assert abs(chk.measured_db - chk.predicted_db) < FIR_TOLERANCE_DB
