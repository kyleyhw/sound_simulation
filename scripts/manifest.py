"""Seed and checksum manifest for datasets and checkpoints (plan 3.5.3).

Datasets are regenerated from seeds rather than stored. HDF5 files embed
creation timestamps, so two byte-for-byte identical simulations produce
different file hashes. The manifest therefore records a **content
digest** instead: SHA-256 over every group, dataset (dtype, shape, bytes)
and attribute in sorted order, skipping only provenance attrs (``SKIP_ATTRS``). It also
records the command that regenerates the archive from its file-level
attrs. Checkpoints are hashed as files.

    uv run python scripts/manifest.py build     # write data/MANIFEST.json
    uv run python scripts/manifest.py verify    # recompute and compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "MANIFEST.json"
# Provenance-only attrs: they describe how an archive was made, not what
# it contains, and older archives predate some of them.
SKIP_ATTRS = {"created_at", "protocol", "n_obstacles", "obstacle_min", "obstacle_max"}
# Values used for archives written before these attrs existed (the v2
# archives in docs/learning.md section 8).
LEGACY_DEFAULTS = {"protocol": "v2", "n_obstacles": 3, "obstacle_min": 4, "obstacle_max": 14}


def _update_attrs(h, attrs) -> None:
    for k in sorted(attrs):
        if k in SKIP_ATTRS:
            continue
        v = np.asarray(attrs[k])
        h.update(k.encode())
        h.update(str(v.dtype).encode() + str(v.shape).encode())
        h.update(v.tobytes() if v.dtype.kind not in "OUS" else repr(v.tolist()).encode())


def content_digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        _update_attrs(h, f.attrs)

        def visit(name: str, obj) -> None:
            h.update(name.encode())
            _update_attrs(h, obj.attrs)
            if isinstance(obj, h5py.Dataset):
                a = np.asarray(obj)
                h.update(str(a.dtype).encode() + str(a.shape).encode())
                h.update(np.ascontiguousarray(a).tobytes())

        names: list[str] = []
        f.visit(names.append)
        for n in sorted(names):
            visit(n, f[n])
    return h.hexdigest()


def file_digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def regenerate_command(path: pathlib.Path) -> str:
    """Rebuild the generate_active_sensing.py command from the file attrs."""
    with h5py.File(path, "r") as f:
        a = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    a = {**LEGACY_DEFAULTS, **a}
    flags = [
        f"--output {path.relative_to(ROOT)}",
        f"--num-samples {a['num_samples_requested']}",
        f"--grid {a['grid']}",
        f"--duration {a['duration']}",
        f"--record-step {a['record_step']}",
        f"--poses-per-room {a['poses_per_room']}",
        f"--n-mics {a['n_mics']}",
        f"--mic-spacing {a['mic_spacing']:g}",
        f"--room-style {a['room_style']}",
        f"--n-obstacles {a['n_obstacles']}",
        f"--obstacle-min {a['obstacle_min']}",
        f"--obstacle-max {a['obstacle_max']}",
        f"--synth-f-start {a['synth_f_start']:g}",
        f"--synth-f-end {a['synth_f_end']:g}",
        f"--protocol {a['protocol']}",
        f"--seed {a['seed']}",
    ]
    if a.get("randomize_source"):
        flags.append("--randomize-source")
    return "uv run python scripts/generate_active_sensing.py " + " ".join(flags)


def build() -> dict:
    entries = []
    for p in sorted((ROOT / "data" / "training_data").glob("*.hdf5")):
        with h5py.File(p, "r") as f:
            seed = int(f.attrs["seed"])
            protocol = str(f.attrs.get("protocol", "v2"))
        entries.append(
            {
                "path": str(p.relative_to(ROOT)),
                "kind": "dataset",
                "seed": seed,
                "protocol": protocol,
                "bytes": p.stat().st_size,
                "content_sha256": content_digest(p),
                "regenerate": regenerate_command(p),
            }
        )
    for p in sorted((ROOT / "checkpoints").glob("*/*.pt")):
        e: dict[str, Any] = {
            "path": str(p.relative_to(ROOT)),
            "kind": "checkpoint",
            "bytes": p.stat().st_size,
            "sha256": file_digest(p),
        }
        try:
            import torch

            ck = torch.load(p, map_location="cpu", weights_only=False)
            args = ck.get("args", {})
            e["train_args"] = {
                k: v for k, v in (args.items() if isinstance(args, dict) else vars(args).items())
            }
            e["epoch"] = ck.get("epoch")
        except Exception as exc:  # checkpoint unreadable: record the hash only
            e["note"] = f"metadata unavailable: {exc}"
        entries.append(e)
    return {"format": "acoustic-system-manifest", "version": 1, "entries": entries}


def verify(manifest: dict) -> int:
    bad = 0
    for e in manifest["entries"]:
        p = ROOT / e["path"]
        if not p.exists():
            print(f"missing  {e['path']}  (regenerate: {e.get('regenerate', 'retrain')})")
            continue
        got = content_digest(p) if e["kind"] == "dataset" else file_digest(p)
        want = e["content_sha256"] if e["kind"] == "dataset" else e["sha256"]
        ok = got == want
        bad += not ok
        print(f"{'ok      ' if ok else 'MISMATCH'} {e['path']}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=("build", "verify"))
    args = ap.parse_args(argv)
    if args.action == "build":
        m = build()
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(m, indent=2, default=str) + "\n")
        print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(m['entries'])} entries)")
        return 0
    return verify(json.loads(MANIFEST.read_text()))


if __name__ == "__main__":
    sys.exit(main())
