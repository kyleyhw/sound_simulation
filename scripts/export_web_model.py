"""Export a sensing checkpoint for in-browser inference (plan 4.3.5, 10.3).

The browser runs the model with a small TypeScript implementation
(``web/src/sensing/``): the exact torchaudio STFT front-end followed by
the conv, pool and transposed-conv layers. There is no ONNX runtime to
download, and the STFT front-end, which ONNX exporters handle poorly,
stays exact. This script writes:

* ``<out>.bin``: every state-dict tensor as little-endian float32,
  concatenated;
* ``<out>.json``: the tensor table (name, shape, offset), the
  acquisition protocol, the calibration (temperature, bias, prior,
  threshold), and the per-pixel training prior map (the no-audio
  baseline, computed from training rooms only);
* ``web/tests/fixtures/sensing_parity.json`` (with ``--fixtures``): a few
  held-out rooms with their poses, the archive's stored recordings, and
  PyTorch's per-pose logits and fused map. The web tests check the
  browser's simulation and inference against them.

    uv run python scripts/export_web_model.py checkpoints/skip_v2/best_iou.pt \\
        --out web/public/models/skip_v2 --fixtures
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import h5py
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.learning.calibration import calibrated_bayes_fuse  # noqa: E402
from acoustic_system.learning.sensing import load_sensing_model  # noqa: E402
from acoustic_system.simulation.dataset import synthetic_chirp  # noqa: E402
from acoustic_system.simulation.simulate import Simulate  # noqa: E402


def prior_map(archive: pathlib.Path, rooms: list[int] | None, size: int) -> np.ndarray:
    """Mean obstacle mask over the training rooms (the per-pixel prior)."""
    with h5py.File(archive, "r") as f:
        keys = sorted(k for k in f if k.startswith("sample_"))
        idx = rooms if rooms is not None else range(len(keys))
        acc = np.zeros((size, size))
        n = 0
        for r in idx:
            m = np.asarray(f[keys[int(r)]]["obstacles"], dtype=np.float64)
            if m.shape != (size, size):
                continue
            acc += m
            n += 1
    return (acc / max(n, 1)).astype(np.float32)


def prior_threshold(
    archive: pathlib.Path, rooms: list[int] | None, pmap: np.ndarray, n: int = 400
) -> float:
    """The no-audio baseline's operating point: the threshold on the prior map
    that maximises mean IoU over (a sample of) the training rooms."""
    with h5py.File(archive, "r") as f:
        keys = sorted(k for k in f if k.startswith("sample_"))
        idx = list(rooms if rooms is not None else range(len(keys)))[:n]
        masks = np.stack([np.asarray(f[keys[int(r)]]["obstacles"]) > 0 for r in idx])
    best, best_tau = -1.0, 0.5
    for tau in np.linspace(0.01, 0.5, 50):
        pred = pmap > tau
        inter = (masks & pred).sum(axis=(1, 2))
        union = (masks | pred).sum(axis=(1, 2))
        iou = float(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0)))
        if iou > best:
            best, best_tau = iou, float(tau)
    return best_tau


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("checkpoint", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "web/public/models/skip_v2")
    ap.add_argument(
        "--train-archive",
        type=pathlib.Path,
        default=ROOT / "data/training_data/active_sensing_v2_train_10kx4.hdf5",
    )
    ap.add_argument(
        "--heldout-archive",
        type=pathlib.Path,
        default=ROOT / "data/training_data/active_sensing_v2_heldout_500x8.hdf5",
    )
    ap.add_argument(
        "--fixtures", action="store_true", help="also write web/tests/fixtures/sensing_parity.json"
    )
    ap.add_argument("--fixture-rooms", type=int, nargs="+", default=[3, 17])
    ap.add_argument("--fixture-poses", type=int, default=4)
    args = ap.parse_args()

    model, cfg = load_sensing_model(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if str(ckpt.get("model_type")) != "skip":
        raise SystemExit("only SkipSensingCNN checkpoints are supported in the browser")
    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    tensors = []
    blobs = []
    offset = 0
    for name, arr in sd.items():
        a = np.ascontiguousarray(arr, dtype="<f4")
        tensors.append({"name": name, "shape": list(a.shape), "offset": offset})
        blobs.append(a.tobytes())
        offset += a.size
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".bin").write_bytes(b"".join(blobs))

    val_rooms = ckpt.get("val_rooms")
    train_rooms = None
    with h5py.File(args.train_archive, "r") as f:
        n_rooms = sum(1 for k in f if k.startswith("sample_"))
    if val_rooms is not None:
        vs = {int(v) for v in val_rooms}
        train_rooms = [r for r in range(n_rooms) if r not in vs]
    pmap = prior_map(args.train_archive, train_rooms, cfg.grid)
    prior_tau = prior_threshold(args.train_archive, train_rooms, pmap)
    manifest = {
        "format": "acoustic-sensing-model",
        "version": 1,
        "model_type": "skip",
        "checkpoint": args.checkpoint.name,
        "epoch": ckpt.get("epoch"),
        "n_values": offset,
        "tensors": tensors,
        "protocol": {
            "grid": cfg.grid,
            "duration": cfg.duration,
            "courant": cfg.courant,
            "mic_spacing": cfg.mic_spacing,
            "f_start": cfg.f_start,
            "f_end": cfg.f_end,
            "sample_rate": cfg.sample_rate,
            "amplitude": cfg.amplitude,
            "target_size": cfg.target_size,
            "protocol": cfg.protocol,
            "sensor_n_fft": 64,
            "sensor_hop": 16,
            "source_n_fft": 512,
            "source_hop": 256,
        },
        "calibration": {
            "temperature": cfg.temperature,
            "bias": cfg.bias,
            "prior": cfg.prior,
            "threshold": cfg.threshold,
        },
        "prior_map": [round(float(v), 5) for v in pmap.ravel()],
        "prior_threshold": prior_tau,
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest))
    print(f"wrote {args.out}.bin ({offset * 4 / 1e6:.2f} MB) and {args.out}.json")

    if not args.fixtures:
        return
    dt = Simulate(grid_shape=(cfg.grid, cfg.grid), courant=cfg.courant).timestep
    n_audio = int(cfg.sample_rate * cfg.duration * dt)
    chirp = synthetic_chirp(n_audio, cfg.sample_rate, cfg.f_start, cfg.f_end, 1.0)
    src = torch.from_numpy(chirp.copy())[None, None]
    rooms = []
    with h5py.File(args.heldout_archive, "r") as f:
        keys = sorted(k for k in f if k.startswith("sample_"))
        for r in args.fixture_rooms:
            g = f[keys[r]]
            sensor = np.asarray(g["sensor"], dtype=np.float32)[: args.fixture_poses]  # (K, T, 2)
            drivers = np.asarray(g.attrs["driver_positions"])[: args.fixture_poses]
            mics = np.asarray(g.attrs["sensor_positions"])[: args.fixture_poses]
            mask = np.asarray(g["obstacles"], dtype=np.uint8)
            with torch.no_grad():
                logits = np.stack(
                    [model(torch.from_numpy(s.T.copy())[None], src)[0, 0].numpy() for s in sensor]
                )
            fused = calibrated_bayes_fuse(logits, cfg.prior, cfg.temperature, cfg.bias)
            # PyTorch's own front-end features for pose 0, so the web test can
            # check the CNN in isolation from the STFT's float32 phase noise.
            with torch.no_grad():
                x0 = torch.from_numpy(sensor[0].T.copy())[None]
                feats = model.front(x0)[0].numpy()
                cross = model.front.spec(x0)[0]
                cross_mag = (cross[0] * cross[1].conj()).abs().numpy()
            rooms.append(
                {
                    "room": r,
                    "mask": mask.ravel().tolist(),
                    "drivers": drivers.tolist(),
                    "mics": mics.tolist(),
                    "sensor": [[round(float(v), 7) for v in s.T.ravel()] for s in sensor],
                    "logits": [[round(float(v), 5) for v in lg.ravel()] for lg in logits],
                    "fused": [round(float(v), 6) for v in fused.ravel()],
                    "front0": [float(v) for v in feats.ravel()],
                    "cross_mag0": [float(v) for v in cross_mag.ravel()],
                }
            )
    fx = ROOT / "web/tests/fixtures/sensing_parity.json"
    fx.write_text(json.dumps({"model": args.out.name, "rooms": rooms}))
    print(f"wrote {fx.relative_to(ROOT)} ({len(rooms)} rooms x {args.fixture_poses} poses)")


if __name__ == "__main__":
    main()
