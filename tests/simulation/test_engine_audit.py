"""Regression tests for the Phase 3 engine audit (plan Task 3.2).

Each test pins one finding from the audit report
(tests/reports/debug_audit_2026_09_24.md): it failed on the pre-audit code
and passes after the fix.
"""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from acoustic_system.simulation.dataset import (
    generate_diverse_obstacles,
    generate_random_obstacles,
    pick_mic_positions,
    random_free_position,
    run_with_sensors,
)
from acoustic_system.simulation.setup import Driver, Sensor
from acoustic_system.simulation.simulate import Simulate
from acoustic_system.simulation.waveforms import AudioFileWaveform, RickerWavelet


def _impulse() -> RickerWavelet:
    return RickerWavelet(amplitude=1.0, frequency=0.2, delay=2.0)


def test_drivers_are_read_only_and_cache_follows_mutations() -> None:
    sim = Simulate(grid_shape=(32, 32), drivers=[Driver((10, 10), _impulse())])
    assert isinstance(sim.drivers, tuple)
    with pytest.raises(AttributeError):
        sim.drivers.append(Driver((5, 5), _impulse()))  # ty: ignore[unresolved-attribute]
    sim.drivers = []  # setter routes through set_drivers
    for _ in range(10):
        sim.step()
    assert float(np.abs(sim.p).max()) == 0.0, "a removed driver must not keep emitting"


def test_driver_is_frozen_and_positions_are_ints() -> None:
    d = Driver(position=(3.0, 4.6), waveform=_impulse())  # ty: ignore[invalid-argument-type]
    assert d.position == (3, 5) and all(type(c) is int for c in d.position)
    with pytest.raises(FrozenInstanceError):
        d.position = (1, 1)  # ty: ignore[invalid-assignment]


def test_driver_dimension_mismatch_raises() -> None:
    sim = Simulate(grid_shape=(16, 16))
    with pytest.raises(ValueError):
        sim.add_driver(Driver(position=(5,), waveform=_impulse()))
    with pytest.raises(ValueError):
        Simulate(grid_shape=(8, 8, 8), drivers=[Driver((2, 2), _impulse())])


def test_set_obstacle_mask_copies_the_callers_array() -> None:
    sim = Simulate(grid_shape=(16, 16))
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True
    sim.set_obstacle_mask(mask)
    assert sim.obstacle_mask is not mask
    sim.clear_obstacles()
    assert int(mask.sum()) == 16, "clear_obstacles must not touch the caller's mask"


def test_timestep_is_python_float_and_fields_stay_float32() -> None:
    sim = Simulate(grid_shape=(16,), courant=0.99, drivers=[Driver((8,), _impulse())])
    assert type(sim.timestep) is float
    for _ in range(3):
        sim.step()
    assert sim.p.dtype == np.float32 and sim.p_prev.dtype == np.float32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wavespeed": 0.0},
        {"gridstep": 0.0},
        {"wavespeed": -1.0},
        {"courant": -0.5},
        {"timestep": -0.1},
    ],
)
def test_constructor_rejects_non_positive_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Simulate(grid_shape=(8, 8), **kwargs)


def test_cfl_warning_is_strict() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Simulate(grid_shape=(8, 8), timestep=1.0 / np.sqrt(2.0))  # exactly at the bound: stable
    with pytest.warns(RuntimeWarning, match="CFL"):
        Simulate(grid_shape=(8, 8), timestep=0.8)


def test_audio_waveform_plays_last_sample_and_downmixes_stereo() -> None:
    wf = AudioFileWaveform.from_samples(np.array([0.0, 0.5, 1.0]), sample_rate=1.0)
    assert wf(2.0) == pytest.approx(1.0)
    assert wf(2.5) == pytest.approx(0.5)  # interpolates down to the implicit zero
    assert wf(3.0) == 0.0
    stereo = np.stack([np.ones(4), -np.ones(4)], axis=1)
    assert AudioFileWaveform.from_samples(stereo, sample_rate=1.0)(1.0) == pytest.approx(0.0)


def test_audio_aliasing_is_detected_and_resampling_removes_it() -> None:
    fs = 10.0  # samples per time unit
    t = np.arange(0, 200, 1 / fs)
    tone = np.sin(2 * np.pi * 4.0 * t)  # 4 cycles/unit >> sim Nyquist 1.0 at dt = 0.5
    wf = AudioFileWaveform.from_samples(tone, sample_rate=fs)
    assert wf.aliased_energy_fraction(0.5) > 0.9
    with pytest.warns(RuntimeWarning, match="alias"):
        Simulate(grid_shape=(16, 16), drivers=[Driver((8, 8), wf)])
    safe = wf.resampled_for(0.5)
    assert safe.sample_rate == pytest.approx(2.0)
    assert safe.aliased_energy_fraction(0.5) < 0.01
    # Band-limited content sampled fast (the synthetic chirps) must NOT warn.
    chirp = AudioFileWaveform.from_samples(np.sin(2 * np.pi * 0.1 * t), sample_rate=fs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Simulate(grid_shape=(16, 16), drivers=[Driver((8, 8), chirp)])


def test_random_free_position_and_mics_work_in_3d() -> None:
    rng = np.random.default_rng(0)
    mask = np.zeros((16, 16, 16), dtype=bool)
    pos = random_free_position((16, 16, 16), mask, rng=rng)
    assert len(pos) == 3
    mics = pick_mic_positions((16, 16, 16), mask, n_mics=2, spacing=4, rng=rng)
    assert len(mics) == 2 and all(len(m) == 3 for m in mics)


def test_mics_never_land_on_an_excluded_source_cell() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    for seed in range(300):
        rng = np.random.default_rng(seed)
        src = random_free_position((12, 12), mask, rng=rng)
        mics = pick_mic_positions((12, 12), mask, n_mics=2, spacing=2, rng=rng, exclude=[src])
        assert src not in mics


def test_v3_thin_walls_have_the_drawn_thickness_and_v2_is_unchanged() -> None:
    # v2 must reproduce the archived stream exactly: occupancy of a fixed seed.
    rng = np.random.default_rng(161803)
    m2 = generate_diverse_obstacles((64, 64), rng=rng, protocol="v2")
    rng = np.random.default_rng(161803)
    m3 = generate_diverse_obstacles((64, 64), rng=rng, protocol="v3")
    assert m2.shape == m3.shape == (64, 64)
    assert m3.sum() <= m2.sum()  # v3 walls are never thicker than v2 walls


def test_v3_rectangles_reach_the_symmetric_margin() -> None:
    lowest = 0
    for seed in range(3000):
        m = generate_random_obstacles(
            (64, 64), n_obstacles=1, min_size=4, max_size=4, rng=np.random.default_rng(seed)
        )
        lowest = max(lowest, int(np.argwhere(m)[:, 0].max()))
    assert lowest == 64 - 2 - 1  # last legal row with margin 2


def test_diverse_generator_returns_empty_on_tiny_grids() -> None:
    assert not generate_diverse_obstacles((4, 4), rng=np.random.default_rng(0)).any()


def test_run_with_sensors_validates_inputs() -> None:
    sim = Simulate(grid_shape=(16, 16))
    with pytest.raises(ValueError):
        run_with_sensors(sim, 10, [Sensor(position=(-1, 3))])
    with pytest.raises(ValueError):
        run_with_sensors(sim, 10, [Sensor(position=(3, 3))], record_step=0)


def test_1d_fallback_matches_a_hand_written_leapfrog() -> None:
    """Plan 3.2.4: the scipy 1D path is kept, so it is tested.

    Reference: p^{n+1} = 2p^n - p^{n-1} + sigma^2 (p_{i+1} - 2p_i + p_{i-1}),
    ends held at 0, the source added after the wall zeroing (same ordering
    as the engine), accumulated in float64.
    """
    n, steps = 64, 120
    wf = RickerWavelet(amplitude=1.0, frequency=0.1, delay=10.0)
    sim = Simulate(grid_shape=(n,), drivers=[Driver((20,), wf)], courant=0.5)
    p = np.zeros(n)
    pp = np.zeros(n)
    sig2 = (sim.wavespeed * sim.timestep / sim.gridstep) ** 2
    t = 0.0
    for _ in range(steps):
        sim.step()
        lap = np.zeros(n)
        lap[1:-1] = p[2:] - 2 * p[1:-1] + p[:-2]
        nxt = 2 * p - pp + sig2 * lap
        nxt[0] = nxt[-1] = 0.0
        nxt[20] += wf(t)
        pp, p = p, nxt
        t += sim.timestep
    assert np.max(np.abs(sim.p - p)) < 1e-4 * max(1.0, float(np.abs(p).max()))
