"""Back-projection, echo ellipses, carving and time reversal (plan 6.2.2-6.2.4)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.imaging.backprojection import backproject, sample_trace, travel_lags
from acoustic_system.imaging.image_source import carve_free_space, image_source_maps, pick_echoes
from acoustic_system.imaging.ir import (
    SampledWaveform,
    TikhonovDeconvolver,
    empty_room_response,
    envelope,
)
from acoustic_system.imaging.time_reversal import backpropagate, imaging_condition
from acoustic_system.simulation.setup import Driver
from acoustic_system.simulation.simulate import Simulate

N = 64
T = 400
SCATTERER = (40, 34)


def _record(mask, src, mics, drive):
    sim = Simulate((N, N))
    sim.set_obstacle_mask(mask)
    sim.set_drivers([Driver(tuple(src), SampledWaveform(drive, sim.timestep))])
    rec = np.zeros((len(mics), T))
    for k in range(T):
        sim.step()
        for i, m in enumerate(mics):
            rec[i, k] = sim.p[tuple(m)]
    return rec


@pytest.fixture(scope="module")
def point_scene(v2_drive):
    """A 3x3 soft scatterer seen from four poses (source + 2 mics each)."""
    mask = np.zeros((N, N), dtype=bool)
    i, j = SCATTERER
    mask[i - 1 : i + 2, j - 1 : j + 2] = True
    sources = np.array([[20, 20], [20, 48], [52, 16], [50, 52]])
    mics = np.array(
        [
            [[14, 16], [26, 18]],
            [[16, 54], [26, 50]],
            [[46, 12], [56, 22]],
            [[44, 56], [56, 50]],
        ]
    )
    residual = np.zeros((4, 2, T))
    fields = []
    for k in range(4):
        y = _record(mask, sources[k], mics[k], v2_drive)
        y0, P = empty_room_response((N, N), sources[k], v2_drive, mics[k], return_field=True)
        residual[k] = y - y0
        fields.append(P)
    return mask, sources, mics, residual, fields


def test_travel_lags_and_sampling():
    lag, rs, rm = travel_lags((8, 8), np.array([0, 0]), np.array([0, 4]), dt=0.5)
    assert lag[0, 2] == pytest.approx((2 + 2) / 0.5)
    assert rs[3, 4] == pytest.approx(5.0)
    tr = np.arange(10, dtype=float)
    np.testing.assert_allclose(sample_trace(tr, np.array([2.5, -1.0, 20.0])), [2.5, 0.0, 0.0])


def test_backprojection_localises_point_scatterer(point_scene, v2_drive):
    _, sources, mics, residual, _ = point_scene
    h = TikhonovDeconvolver(v2_drive, T, lam=1e-2)(residual)
    # Gate at 100 lags (50 cells of path): single scattering off the object,
    # before its echoes off the outer walls arrive.
    img = backproject((N, N), sources, mics, h, 0.5, spreading=False, max_lag=100, lag_offset=-2)
    peak = np.unravel_index(int(np.argmax(img)), img.shape)
    assert np.hypot(peak[0] - SCATTERER[0], peak[1] - SCATTERER[1]) <= 2.5


def test_echo_ellipses_and_carving(point_scene, v2_drive):
    mask, sources, mics, residual, _ = point_scene
    env = envelope(TikhonovDeconvolver(v2_drive, T, lam=1e-2)(residual))
    env[..., 100:] = 0.0  # same single-scattering gate as the back-projection test
    maps = image_source_maps((N, N), sources, mics, env, 0.5, max_echoes=1)
    stacked = maps.sum(0)
    peak = np.unravel_index(int(np.argmax(stacked)), stacked.shape)
    assert np.hypot(peak[0] - SCATTERER[0], peak[1] - SCATTERER[1]) <= 3.0
    carve = carve_free_space((N, N), sources, mics, residual, 0.5, rel_threshold=1e-3)
    # First-arrival ellipses never cover the scatterer itself...
    assert carve[mask].max() == 0
    # ... but every one covers its own foci (the devices stand in free space).
    for k in range(4):
        assert carve[tuple(sources[k])] >= 1


def test_pick_echoes_orders_and_separates():
    e = np.zeros(100)
    e[[20, 23, 60]] = [1.0, 0.5, 0.8]
    lags, amps = pick_echoes(e, max_echoes=3, rel_threshold=0.1, min_separation=6)
    assert lags.tolist() == [20, 60]
    np.testing.assert_allclose(amps, [1.0, 0.8])


def test_time_reversal_images_the_scatterer(point_scene):
    _, sources, mics, residual, fields = point_scene
    img = np.zeros((N, N))
    for k in range(4):
        Q = backpropagate((N, N), mics[k], residual[k])
        img += imaging_condition(fields[k], Q, normalise=True)
    # Positive (obstacle-like) image energy concentrates near the scatterer.
    near = np.zeros((N, N), dtype=bool)
    i, j = SCATTERER
    near[i - 3 : i + 4, j - 3 : j + 4] = True
    assert img[near].max() > 0
    assert np.abs(img[near]).mean() > 3 * np.abs(img[~near]).mean()


def test_time_reversal_is_the_fwi_adjoint(v2_drive):
    """Re-emitting reversed residuals = the autograd gradient of the misfit at o = 0."""
    torch = pytest.importorskip("torch")
    from acoustic_system.simulation.torch_engine import TorchFDTD, TorchGrid

    n, steps = 32, 160
    drive = v2_drive[:steps]
    src, mics = np.array([10, 9]), np.array([[20, 22], [24, 12]])
    mask = np.zeros((n, n), dtype=bool)
    mask[14:18, 15:19] = True
    sim = Simulate((n, n))
    sim.set_obstacle_mask(mask)
    sim.set_drivers([Driver(tuple(src), SampledWaveform(drive, sim.timestep))])
    y = np.zeros((2, steps))
    for k in range(steps):
        sim.step()
        y[:, k] = [sim.p[tuple(m)] for m in mics]
    y0, P = empty_room_response((n, n), src, drive, mics, return_field=True)
    assert P is not None
    img = imaging_condition(P, backpropagate((n, n), mics, y - y0), normalise=False)

    eng = TorchFDTD(
        TorchGrid((n, n)),
        torch.tensor(src[None, None]),
        torch.tensor(mics[None]),
        dtype=torch.float64,
    )
    occ = torch.zeros(1, n, n, dtype=torch.float64, requires_grad=True)
    out = eng.run(torch.tensor(drive[None, None]), occ=occ)
    (0.5 * ((torch.tensor(y) - out[0]) ** 2).sum()).backward()
    assert occ.grad is not None
    grad = -occ.grad[0].numpy()
    keep = np.ones((n, n), dtype=bool)
    for p in np.concatenate([src[None], mics]):
        keep[tuple(p)] = False  # injection cells differ by the injected term
    np.testing.assert_allclose(img[keep], grad[keep], rtol=0, atol=1e-5 * np.abs(grad).max())
