"""Parallel dataset generation reproduces the serial archive (plan 5.7.3)."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _gen(out: pathlib.Path, workers: int) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_active_sensing.py"),
            "--output", str(out), "--num-samples", "5", "--grid", "48", "--duration", "200",
            "--poses-per-room", "2", "--seed", "3", "--workers", str(workers),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip


def test_workers_do_not_change_the_archive(tmp_path):
    a, b = tmp_path / "serial.h5", tmp_path / "parallel.h5"
    _gen(a, 1)
    _gen(b, 2)
    with h5py.File(a) as fa, h5py.File(b) as fb:
        assert sorted(fa) == sorted(fb)
        for k in fa:
            for name in fa[k]:
                assert np.array_equal(np.asarray(fa[k][name]), np.asarray(fb[k][name])), (k, name)
            for attr, v in fa[k].attrs.items():
                assert np.array_equal(np.asarray(v), np.asarray(fb[k].attrs[attr])), (k, attr)
        assert fa.attrs["mean_obstacle_fraction"] == fb.attrs["mean_obstacle_fraction"]
