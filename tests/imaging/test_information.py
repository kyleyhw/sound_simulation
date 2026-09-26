"""Information limits: Born sensitivity, emission algebra, resolution (study 2026-09-26)."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_system.imaging import information as inf

SMALL = inf.Room("cpml", n=48, wall=8)


def _pair_config(speakers, mics, steps=150, room=SMALL) -> inf.Config:
    drive = inf.ricker_samples(steps)
    return inf.Config("t", inf.seq_shots(speakers, mics, drive, steps), room)


def test_born_monopole_matches_central_difference():
    """The monopole kernel is the engine's exact derivative w.r.t. C at a cell."""
    cfg = _pair_config([(36, 20)], [(36, 26)])
    grid = inf.PixelGrid(r0=18, r1=22, c0=20, c1=24, size=1)
    j = 5
    km, _ = inf.config_rows(cfg, grid, [j])
    eps = 1e-2
    ys = []
    for sgn in (1.0, -1.0):
        sp = np.ones((48, 48))
        sp[grid.cells(j)] = 1 + sgn * eps
        ys.append(
            np.concatenate(
                [
                    inf.run_shot(SMALL, s.drives, s.steps, None, s.mics, speed=sp)[1].ravel()
                    for s in cfg.shots
                ]
            )
        )
    dC = (1 + eps) ** 2 - (1 - eps) ** 2
    fd = (ys[0] - ys[1]) / dC
    assert np.linalg.norm(fd - km[0]) / np.linalg.norm(fd) < 5e-3


def _random_kernels(rng, S=2, P=24, T=60):
    t = np.arange(T)
    K = np.zeros((S, P, T))
    for s in range(S):
        for p in range(P):
            d = rng.uniform(5, 40)
            K[s, p] = rng.normal() * np.exp(-0.5 * ((t - d) / 2.0) ** 2) * np.cos(0.8 * (t - d))
    return K


def test_summed_and_steered_rows_cannot_exceed_separate_rows():
    rng = np.random.default_rng(1)
    K = _random_kernels(rng)
    r = inf.ricker_samples(40, f0=0.2, delay=6.0)
    z = np.zeros_like(r)
    steps = 100
    sep = np.concatenate(
        [inf.drive_rows(K, np.stack([r, z]), steps), inf.drive_rows(K, np.stack([z, r]), steps)], 1
    )
    summed = inf.drive_rows(K, np.stack([r, r]), steps)
    beams = np.concatenate(
        [
            inf.drive_rows(K, np.stack([r, g * inf.delayed(r, d, 1.0, 40)]), steps)
            for g, d in ((1, 0), (-1, 0), (1, 3))
        ],
        1,
    )
    rank = np.linalg.matrix_rank
    assert rank(summed) <= rank(sep)
    assert rank(beams) <= rank(sep)
    # Rows of any linear emission lie in the row space of the separate rows' drive-convolved kernels:
    # the summed rows are exactly the sum of the separate ones.
    np.testing.assert_allclose(summed, sep[:, :steps] + sep[:, steps:], atol=1e-12)
    # Information: sum-of-squares of the summed rows never exceeds 2x the separate (coherent array gain bound).
    Fs, Fq = summed @ summed.T, sep @ sep.T
    assert np.max(np.linalg.eigvalsh(Fs - 2 * Fq)) < 1e-9


def test_emission_gram_rank_and_beam_eigenvalues():
    r = inf.ricker_samples(64, f0=0.2, delay=8.0)
    n = 256
    M_sum = inf.emission_gram([np.stack([r, r])], n)
    assert np.all(np.linalg.matrix_rank(M_sum, tol=1e-9) <= 1)
    tau = 5
    shots = [np.stack([r, g * inf.delayed(r, d, 1.0, 64)]) for g, d in ((1, 0), (-1, 0))] + [
        np.stack([inf.delayed(r, tau, 1.0, 64), r])
    ]
    M = inf.emission_gram(shots, n)
    R2 = np.abs(np.fft.rfft(r, n)) ** 2
    w = 2 * np.pi * np.arange(n // 2 + 1) / n
    ev = np.linalg.eigvalsh(M)
    # shots (1,1), (1,-1), (e^{-iwt},1): M = |R|^2 [[3, e^{iwt}], [e^{-iwt}, 3]] -> 3 +- 1 (up to truncation of the delayed copy)
    good = R2 > 1e-3 * R2.max()
    np.testing.assert_allclose(ev[good, 0] / R2[good], 2.0, atol=0.05)
    np.testing.assert_allclose(ev[good, 1] / R2[good], 4.0, atol=0.05)
    assert w.shape[0] == ev.shape[0]


def test_fully_separable_codes_give_the_sequential_gram():
    """Codes that never overlap in lag (here: time-multiplexed in one recording) carry exactly the in-turn information."""
    rng = np.random.default_rng(2)
    K = _random_kernels(rng, T=60)
    r = inf.ricker_samples(30, f0=0.2, delay=6.0)
    z = np.zeros(30)
    steps = 90
    seq = np.concatenate(
        [inf.drive_rows(K, np.stack([r, z]), steps), inf.drive_rows(K, np.stack([z, r]), steps)], 1
    )
    gap = 90
    dA = np.concatenate([r, np.zeros(gap)])
    dB = np.concatenate([np.zeros(gap), r])
    sim = inf.drive_rows(K, np.stack([dA, dB]), gap + steps)
    np.testing.assert_allclose(sim @ sim.T, seq @ seq.T, rtol=1e-9, atol=1e-12)


def test_chirp_crosstalk_falls_with_code_length():
    rng = np.random.default_rng(3)
    K = _random_kernels(rng, P=16, T=80)
    diffs = []
    for L in (64, 512):
        up = inf.unit_energy(inf.chirp_code(L), 1.0)
        dn = inf.unit_energy(inf.chirp_code(L, down=True), 1.0)
        z = np.zeros(L)
        steps = L + 80
        seq = np.concatenate(
            [
                inf.drive_rows(K, np.stack([up, z]), steps),
                inf.drive_rows(K, np.stack([z, dn]), steps),
            ],
            1,
        )
        sim = inf.drive_rows(K, np.stack([up, dn]), steps)
        Gs, Gq = sim @ sim.T, seq @ seq.T
        diffs.append(np.linalg.norm(Gs - Gq) / np.linalg.norm(Gq))
    assert diffs[1] < 0.6 * diffs[0]


def test_information_measures():
    lam = np.array([100.0, 10.0, 1.0, 0.1])
    assert inf.dof(lam, 1.0, 1.0) == 3
    assert inf.dof(lam, 0.1, 1.0) == 4
    assert inf.dfs(lam, 1.0, 1.0) == pytest.approx(sum(x / (1 + x) for x in lam))
    V = np.eye(4)
    np.testing.assert_allclose(inf.posterior_std(lam, V, 1.0, 1.0), 1 / np.sqrt(lam + 1))
    R = inf.resolution_matrix(lam, V, 1.0, 1.0)
    np.testing.assert_allclose(np.diag(R), lam / (lam + 1))


def test_psf_narrows_with_aperture():
    """A point scatterer's cross-range PSF is narrower for the wider speaker pair."""
    grid = inf.PixelGrid(r0=10, r1=30, c0=10, c1=38, size=2)
    j = grid.index(18, 24)
    widths = []
    for sp, mi in (
        (((40, 21), (40, 27)), ((40, 22), (40, 26))),
        (((40, 11), (40, 37)), ((40, 12), (40, 36))),
    ):
        g = inf.config_grams(_pair_config(sp, mi, steps=160), grid).combine(1.3, 2.5)
        lam, V = inf.eig_gram(g)
        sig = np.sqrt(lam[0]) * 1e-3  # high SNR
        R = inf.resolution_matrix(lam, V, sig, 0.5)
        _, wc = inf.psf_widths(R, grid)
        widths.append(wc[j])
    assert widths[1] < widths[0]


def test_standard_configs_match_the_training_study():
    cfg = inf.standard_configs()
    assert cfg["bar8"].n_traces == 64
    assert cfg["seq"].n_traces == 4 and len(cfg["seq"].shots) == 2
    assert [p for p, _ in cfg["seq"].shots[0].drives] == [(86, 39)]
    assert cfg["seq"].shots[0].mics == [(86, 41), (86, 49)]
    assert cfg["wide_seq"].positions()[0] == [(86, 31), (86, 59)]
    assert cfg["code"].shots[0].steps == 2 * inf.LISTEN
    assert len(cfg["beams"].shots) == 4
    assert cfg["seq_k4"].n_traces == 16
    # Equal energy per speaker per shot.
    for name in ("band", "code", "seq_f16"):
        for s in cfg[name].shots:
            for _, d in s.drives:
                assert inf.energy(d) == pytest.approx(inf.RICKER_ENERGY, rel=1e-9)
