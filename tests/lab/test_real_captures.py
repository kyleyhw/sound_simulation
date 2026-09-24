"""Offline capture scoring (plan 9.7) on a synthetic capture with known answers."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np

from acoustic_system.utils.room_ir import decay_metrics, eyring, f32_to_b64, find_echoes

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "eval_real_captures", ROOT / "scripts" / "eval_real_captures.py"
)
assert spec and spec.loader
erc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(erc)

FS = 48000.0


def synthetic_rir(
    t60: float, echo_m: list[float], length_s: float = 1.0, seed: int = 3
) -> np.ndarray:
    n = int(length_s * FS)
    h = np.zeros(n)
    d0 = 240
    h[d0] = 1.0
    for k, d in enumerate(echo_m):
        h[d0 + round(2 * d / 343.0 * FS)] += 0.6 / (k + 1)
    rng = np.random.default_rng(seed)
    t = np.arange(n - d0 - 800) / FS
    h[d0 + 800 :] += 0.04 * rng.standard_normal(t.size) * np.exp(-3 * np.log(10) * t / t60)
    return h


def test_echoes_and_decay_recover_ground_truth():
    h = synthetic_rir(0.5, [0.72, 1.4])
    direct, echoes = find_echoes(h, FS)
    assert direct == 240
    assert abs(echoes[0].distance - 0.72) < 0.005
    assert abs(echoes[1].distance - 1.4) < 0.005
    dm = decay_metrics(h, FS, direct)
    assert dm["t30"] is not None and abs(dm["t30"] / 0.5 - 1) < 0.1


def test_eval_script_end_to_end(tmp_path):
    caps = []
    for k, (t60, d) in enumerate([(0.4, 0.6), (0.7, 1.1)]):
        h = synthetic_rir(t60, [d], seed=k)
        caps.append(
            {
                "id": str(k),
                "label": f"room {k}",
                "sampleRate": FS,
                "irs": [f32_to_b64(h), f32_to_b64(0.8 * h)],
                "measuredDistance": d + 0.02,
                "room": {"lx": 5, "ly": 4, "lz": 2.7},
                "estimates": {"firstEchoDistance": d, "t30": t60},
            }
        )
    path = tmp_path / "captures.json"
    path.write_text(
        json.dumps({"format": "acoustic-sandbox-captures", "version": 1, "captures": caps})
    )
    out = tmp_path / "res.json"
    assert erc.main([str(path), "--alpha", "0.2", "--json", str(out)]) == 0
    res = json.loads(out.read_text())
    s = res["summary"]
    assert s["n_captures"] == 2 and s["n_with_tape"] == 2
    assert abs(s["distance_mae_cm"] - 2.0) < 0.6
    for r, (t60, _) in zip(res["captures"], [(0.4, 0.6), (0.7, 1.1)]):
        assert abs(r["browser_t30_delta_pct"]) < 10
        assert abs(r["t60_eyring"] - eyring(5, 4, 2.7, 0.2)) < 1e-9
        # Implied alpha reproduces the measured T30 through Eyring.
        assert abs(eyring(5, 4, 2.7, r["alpha_implied"]) - r["t30"]) < 1e-6
