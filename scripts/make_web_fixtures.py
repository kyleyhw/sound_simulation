"""Write Python-engine fixtures for the browser engine's parity tests.

The TypeScript FDTD engine (web/src/engine/simulation.ts) must reproduce
``Simulate`` on its Python-parity path (p = 0 walls and obstacles, soft
additive sources). This script runs two small scenes through the Python
engine and stores inputs + final fields as JSON in web/tests/fixtures/, which
web/tests/unit/parity.test.ts replays.

    uv run python scripts/make_web_fixtures.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from acoustic_system.simulation.setup import Driver  # noqa: E402
from acoustic_system.simulation.simulate import Simulate  # noqa: E402
from acoustic_system.simulation.waveforms import Cosine, RickerWavelet  # noqa: E402

OUT = _ROOT / "web" / "tests" / "fixtures"


def run(
    shape, obstacles, drivers, steps, boundary="soft", outer_beta=1.0, materials=None, speed=None
):
    sim = Simulate(
        grid_shape=shape, courant=0.5, boundary=boundary, boundary_beta=outer_beta, sponge_cells=12
    )
    if obstacles:
        sim.set_obstacle(obstacles)
    for pos, mid in materials or []:
        sim.set_material([tuple(pos)], mid)
    if speed is not None:
        sim.set_speed_map(speed)
    sim.set_drivers([Driver(position=tuple(p), waveform=w) for p, w in drivers])
    for _ in range(steps):
        sim.step()
    return sim


def wf_json(w) -> dict:
    if isinstance(w, RickerWavelet):
        return {
            "type": "ricker",
            "amplitude": w.amplitude,
            "frequency": w.frequency,
            "delay": w.delay,
        }
    return {"type": "cosine", "amplitude": w.amplitude, "frequency": w.frequency}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {
        "parity2d": (
            (64, 48),
            [(i, j) for i in range(20, 30) for j in range(10, 16)]
            + [(40, j) for j in range(5, 40)],
            [((32, 30), RickerWavelet(5.0, 0.1, 20.0)), ((10, 10), Cosine(0.04, 0.5))],
            150,
        ),
        "parity3d": (
            (24, 20, 16),
            [(i, j, k) for i in range(8, 12) for j in range(5, 9) for k in range(4, 12)],
            [((12, 14, 8), RickerWavelet(5.0, 0.12, 12.0))],
            60,
        ),
    }
    # General path (Phase 5): every material, outer boundary kind and c(x).
    shape = (60, 50)
    mats = [((i, 10), 2) for i in range(10, 50)]
    mats += [((i, 30), 3) for i in range(5, 25)] + [((i, 31), 4) for i in range(25, 45)]
    mats += [((50, j), 5) for j in range(5, 25)] + [((52, j), 6) for j in range(25, 45)]
    soft = [(20, j) for j in range(35, 45)]
    speed = np.ones(shape, dtype=np.float32)
    speed[30:45, 15:28] = 0.6
    general = {
        "general_rigid": ("rigid", 1.0, None),
        "general_absorb": ("absorb", 0.4, None),
        "general_mur": ("mur", 1.0, None),
        "general_sponge": ("sponge", 1.0, None),
        "general_speed": ("rigid", 1.0, speed),
        "general_faces": (("mur", "rigid", "sponge", "absorb"), 0.4, None),
    }
    for name, (boundary, beta, spd) in general.items():
        drv = [((30, 22), RickerWavelet(5.0, 0.1, 15.0)), ((12, 40), Cosine(0.03, 0.4))]
        sim = run(shape, soft, drv, 220, boundary, beta, mats, spd)
        data = {
            "shape": list(shape),
            "courant": 0.5,
            "boundary": boundary if isinstance(boundary, str) else "soft",
            "faces": list(boundary) if not isinstance(boundary, str) else None,
            "outer_beta": beta,
            "sponge_cells": 12,
            "obstacles": [list(o) for o in soft],
            "materials": [[*pos, mid] for pos, mid in mats],
            "speed": None if spd is None else [float(v) for v in spd.ravel()],
            "drivers": [{"pos": list(p), "waveform": wf_json(w)} for p, w in drv],
            "steps": 220,
            "timestep": sim.timestep,
            "p": [float(f"{v:.8g}") for v in np.asarray(sim.p, dtype=np.float64).ravel()],
        }
        (OUT / f"{name}.json").write_text(json.dumps(data))
        print(f"wrote {name}.json  |p|max={float(np.abs(sim.p).max()):.4g}")

    for name, (shape, obst, drivers, steps) in cases.items():
        sim = run(shape, obst, drivers, steps)
        data = {
            "shape": list(shape),
            "courant": 0.5,
            "obstacles": [list(o) for o in obst],
            "drivers": [{"pos": list(p), "waveform": wf_json(w)} for p, w in drivers],
            "steps": steps,
            "timestep": sim.timestep,
            "p": [float(f"{v:.8g}") for v in np.asarray(sim.p, dtype=np.float64).ravel()],
        }
        (OUT / f"{name}.json").write_text(json.dumps(data))
        print(f"wrote {name}.json  |p|max={float(np.abs(sim.p).max()):.4g}")


if __name__ == "__main__":
    main()
