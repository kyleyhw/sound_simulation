"""Learned imaging models, pose error, passive and 3D sensing (plan 6.3, 6.6).

Protocol (nothing is tuned on held-out rooms):

* **Rooms.** Training archive rooms ``[0, 500)`` are the *validation*
  rooms: the logistic "all physics" fusion of ``eval_imaging.py`` is fitted
  there (as in the physics report), and every learned model takes its IoU
  threshold and temperature there. Rooms ``[500, n_train_rooms)`` train
  the networks. The priors are those of ``eval_imaging.py`` (filled mask:
  mean of training rooms ``[500, 10000)``; illuminated boundary: mean over
  rooms ``[500, 3500)`` with their own poses).
* **Held-out.** All 500 rooms of the held-out archive, first K poses
  (K = 4 unless stated; K = 1, 2, 8 for the pose sweep). Per room: IoU at
  the validation threshold, AP, boundary F, information gain over the
  prior; paired per-room differences against the prior and against the
  logistic fusion (mean, SE, z).
* **Stages** (``--stages``, each cached under ``--cache-dir``, outside the
  repo, so a re-run only re-scores):

  - ``extract``: per-pose physics images, residuals, impulse responses and
    targets for training rooms ``[0, n_train_rooms)`` and the held-out rooms;
  - ``logistic``: the no-audio prior and the all-physics fusion (always run);
  - ``unet`` (6.3.1), ``irnet`` (6.3.2), ``setnet`` (6.3.3),
    ``uncertainty`` (6.3.4);
  - ``nbv`` (6.6.3 next-best pose), ``passive`` (6.6.4);
  - ``poseimg`` (images under pose error and after refinement, the slow
    part of 6.6.1-6.6.2) and ``pose`` (their scores);
  - ``room3d`` (6.6.5);
  - ``report``: leakage check, figures and ``<out-dir>/results.json``.

Usage (the full run; about 4 h on one shared core, dominated by training
and the joint pose search)::

    NUMBA_NUM_THREADS=1 uv run python scripts/eval_imaging_models.py \\
        --out-dir tests/reports/imaging_models_2026_09_24_artifacts
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from typing import Any

import h5py
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from acoustic_system.imaging import models as M  # noqa: E402
from acoustic_system.imaging.ir import TikhonovDeconvolver, envelope  # noqa: E402
from acoustic_system.imaging.pipeline import ArchiveRoom  # noqa: E402
from acoustic_system.imaging.pose_images import (  # noqa: E402
    POSE_CHANNELS,
    aggregate,
    compute_pose_images,
    geometry_channels,
)
from acoustic_system.imaging.targets import illuminated_boundary  # noqa: E402
from acoustic_system.learning.metrics import mean_se  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_imaging", _REPO_ROOT / "scripts/eval_imaging.py"
)
assert _spec is not None and _spec.loader is not None
EI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EI)

DATA = _REPO_ROOT / "data" / "training_data"
TRAIN_ARCHIVE = DATA / "active_sensing_v2_train_10kx4.hdf5"
HELD_ARCHIVE = DATA / "active_sensing_v2_heldout_500x8.hdf5"
IR_LAGS = 256
CHUNK = 500
N_GRID = 64
DT = 0.5
TARGETS = ("mask", "illum")
LOG = print


# ---------------------------------------------------------------------------
# Stage: extract
# ---------------------------------------------------------------------------


def extract_chunk(path: pathlib.Path, rooms: range, n_poses: int, cache: pathlib.Path) -> None:
    """Per-pose images, residuals, IRs, recordings and targets for ``rooms``."""
    if cache.exists():
        return
    out: dict[str, list] = {
        k: [] for k in ("images", "residual", "ir", "rec", "mask", "illum_pose", "sources", "mics")
    }
    dec = None
    drive = None
    t0 = time.perf_counter()
    with h5py.File(path, "r") as hf:
        for n, r in enumerate(rooms):
            room = ArchiveRoom.from_group(hf[f"sample_{r:04d}"], n_poses)
            if dec is None:
                dec = TikhonovDeconvolver(room.drive, room.recordings.shape[-1], lam=1e-2)
                drive = room.drive
            pi = compute_pose_images(room, dec)
            out["images"].append(pi.images.astype(np.float32))
            out["residual"].append(pi.residual.astype(np.float32))
            out["ir"].append(pi.ir[..., :IR_LAGS].astype(np.float32))
            out["rec"].append(room.recordings.astype(np.float32))
            out["mask"].append(room.mask)
            out["illum_pose"].append(
                np.stack(
                    [
                        illuminated_boundary(
                            room.mask, room.sources[k : k + 1], room.mics[k : k + 1]
                        )
                        for k in range(len(room.sources))
                    ]
                )
            )
            out["sources"].append(room.sources)
            out["mics"].append(room.mics)
            if (n + 1) % 100 == 0:
                LOG(f"  {path.name} rooms {rooms.start}+{n + 1}: {time.perf_counter() - t0:.0f} s")
    arr = {k: np.stack(v) for k, v in out.items()}
    arr["drive"] = np.asarray(drive)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez(tmp, allow_pickle=False, **arr)
    tmp.rename(cache)


def train_chunk_path(cache_dir: pathlib.Path, start: int) -> pathlib.Path:
    return cache_dir / f"train_pose_{start:05d}.npz"


def held_path(cache_dir: pathlib.Path, n_heldout: int) -> pathlib.Path:
    return cache_dir / f"held_pose_k8_{n_heldout}.npz"


def extract_all(cache_dir: pathlib.Path, n_train_rooms: int, n_heldout: int) -> None:
    starts = list(range(0, n_train_rooms, CHUNK))
    extract_chunk(
        TRAIN_ARCHIVE, range(0, min(CHUNK, n_train_rooms)), 4, train_chunk_path(cache_dir, 0)
    )
    extract_chunk(HELD_ARCHIVE, range(n_heldout), 8, held_path(cache_dir, n_heldout))
    for start in starts[1:]:
        rooms = range(start, min(start + CHUNK, n_train_rooms))
        extract_chunk(TRAIN_ARCHIVE, rooms, 4, train_chunk_path(cache_dir, start))
        LOG(f"train chunk {start} done")


def load_npz(paths: list[pathlib.Path], keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    parts: dict[str, list] = {}
    drive = None
    for p in paths:
        with np.load(p) as z:
            for k in z.files:
                if k == "drive":
                    drive = z[k]
                elif keys is None or k in keys:
                    parts.setdefault(k, []).append(z[k])
    out: dict[str, Any] = {k: np.concatenate(v) for k, v in parts.items()}
    out["drive"] = drive
    return out


# ---------------------------------------------------------------------------
# Priors, inputs, targets
# ---------------------------------------------------------------------------


def priors(cache_dir: pathlib.Path, n_train_rooms: int) -> tuple[np.ndarray, np.ndarray]:
    """Filled-mask prior (rooms 500-9999) and illuminated prior (rooms 500-3499)."""
    path = cache_dir / "priors.npz"
    if path.exists():
        with np.load(path) as z:
            return z["fill"], z["illum"]
    with h5py.File(TRAIN_ARCHIVE, "r") as hf:
        n = len([k for k in hf.keys() if k.startswith("sample_")])
        acc = np.zeros((N_GRID, N_GRID))
        for r in range(500, n):
            acc += hf[f"sample_{r:04d}"]["obstacles"][()].astype(bool)
    fill = acc / (n - 500)
    ill_acc = np.zeros((N_GRID, N_GRID))
    cnt = 0
    for start in range(500, 3500, CHUNK):
        d = load_npz([train_chunk_path(cache_dir, start)], ("illum_pose",))
        ill_acc += d["illum_pose"].any(1).sum(0)
        cnt += len(d["illum_pose"])
    illum = ill_acc / cnt
    np.savez(path, fill=fill, illum=illum)
    return fill, illum


def zscore_maps(x: np.ndarray) -> np.ndarray:
    """Standardise each map over its last two axes."""
    x = np.asarray(x, dtype=np.float32)
    m = x.mean(axis=(-1, -2), keepdims=True)
    s = x.std(axis=(-1, -2), keepdims=True)
    return (x - m) / (s + 1e-6)


def aligned_inputs(images: np.ndarray, k: int) -> np.ndarray:
    """``(N, 4, H, W)`` per-room standardised pose-summed images."""
    agg = aggregate(images, k)
    return np.stack([zscore_maps(agg[c]) for c in POSE_CHANNELS], 1)


def targets(d: dict, k: int) -> np.ndarray:
    """``(N, 2, H, W)`` float targets: filled mask, illuminated boundary of the first k poses."""
    return np.stack([d["mask"], d["illum_pose"][:, :k].any(1)], 1).astype(np.float32)


def ir_inputs(ir: np.ndarray) -> np.ndarray:
    """``(N, K, M, 2, L)``: IR and its envelope, each divided by the IR's RMS."""
    ir = np.asarray(ir, dtype=np.float64)
    rms = np.sqrt(np.mean(ir**2, axis=-1, keepdims=True)) + 1e-12
    return np.stack([ir / rms, envelope(ir) / rms], -2).astype(np.float32)


def eval_dict(d: dict, k: int) -> dict:
    """The feature dictionary ``eval_imaging.Evaluator`` expects, for the first k poses."""
    agg = aggregate(d["images"], k)
    out = {c: agg[c].astype(np.float64) for c in POSE_CHANNELS}
    out["residual"] = d["residual"][:, :k]
    out["sources"] = d["sources"][:, :k]
    out["mics"] = d["mics"][:, :k]
    out["mask"] = d["mask"]
    out["illum"] = d["illum_pose"][:, :k].any(1)
    out["drive"] = np.broadcast_to(d["drive"], (len(d["mask"]), d["drive"].size))
    dev = np.zeros(d["mask"].shape, dtype=bool)
    n = np.arange(len(dev))
    for pos in [out["sources"]] + [out["mics"][:, :, m] for m in range(out["mics"].shape[2])]:
        for kk in range(pos.shape[1]):
            dev[n, pos[:, kk, 0], pos[:, kk, 1]] = True
    out["devices"] = dev
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class Scorer:
    """Scores probability maps on held-out rooms against the prior and a reference."""

    def __init__(self, prior: dict[str, np.ndarray], val_truth: dict[str, np.ndarray]):
        self.prior = prior
        self.val_truth = val_truth
        self.per_room: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
        self.table: dict[str, dict[str, dict[str, dict]]] = {}

    def add(
        self,
        method: str,
        target: str,
        k: int,
        prob_held: np.ndarray,
        truth_held: np.ndarray,
        tau: float,
        refs: tuple[str, ...] = ("prior", "logistic"),
    ) -> dict:
        sc = EI.score(prob_held, truth_held, tau, self.prior[target])
        self.per_room[(method, target, k)] = sc
        entry: dict = {"tau": tau, "n": int(len(truth_held))}
        for key, v in sc.items():
            m, se = mean_se(v)
            entry[key] = {"mean": m, "se": se}
        for ref in refs:
            base = self.per_room.get((ref, target, k))
            if base is None or ref == method:
                continue
            for key in ("iou", "ap", "bf", "ig"):
                dm, dse = mean_se(sc[key] - base[key])
                entry[key][f"d_{ref}"] = dm
                entry[key][f"d_{ref}_se"] = dse
                entry[key][f"z_{ref}"] = dm / dse if dse > 0 else float("nan")
        self.table.setdefault(target, {}).setdefault(str(k), {})[method] = entry
        return entry

    def tau(self, prob_val: np.ndarray, target: str) -> float:
        return EI.tune_tau(prob_val, self.val_truth[target])


def fmt(entry: dict, ref: str = "prior") -> str:
    s = f"IoU {entry['iou']['mean']:.4f} AP {entry['ap']['mean']:.4f} IG {entry['ig']['mean']:7.1f}"
    for r in (ref, "logistic"):
        if f"d_{r}" in entry["iou"]:
            e = entry["iou"]
            s += f" | dIoU[{r}] {e[f'd_{r}']:+.4f}±{e[f'd_{r}_se']:.4f} (z {e[f'z_{r}']:.1f})"
    return s


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class Context:
    """Everything the model stages share (loaded once)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        cd = args.cache_dir
        self.cache = cd
        fill, illum = priors(cd, args.n_train_rooms)
        self.prior = {"mask": fill, "illum": illum}
        self.prior_logit = np.stack([EI.logit(fill), EI.logit(illum)]).astype(np.float32)
        self.val = load_npz([train_chunk_path(cd, 0)])
        self.held = load_npz([held_path(cd, args.n_heldout)])
        self.val_y = targets(self.val, 4)
        self.scorer = Scorer(
            self.prior, {"mask": self.val["mask"], "illum": self.val_y[:, 1].astype(bool)}
        )
        self._train: dict | None = None
        self.results: dict = {}
        self.evaluators: dict = {}

    @property
    def train(self) -> dict:
        if self._train is None:
            starts = range(CHUNK, self.args.n_train_rooms, CHUNK)
            self._train = load_npz(
                [train_chunk_path(self.cache, s) for s in starts],
                ("images", "ir", "mask", "illum_pose", "sources", "mics"),
            )
            LOG(f"loaded {len(self._train['mask'])} training rooms")
        return self._train

    def held_truth(self, target: str, k: int) -> np.ndarray:
        return targets(self.held, k)[:, TARGETS.index(target)].astype(bool)


def iterate(
    n: int, batch: int, rng: np.random.Generator, shuffle: bool = True
) -> Iterator[np.ndarray]:
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, batch):
        yield idx[s : s + batch]


def prior_batch(ctx: Context, b: int, t: int) -> torch.Tensor:
    pl = M.d4_grid(ctx.prior_logit, t)
    return torch.from_numpy(np.broadcast_to(pl, (b,) + pl.shape).copy())


def aligned_batches(
    ctx: Context, X: np.ndarray, Y: np.ndarray, batch: int, seed: int
) -> Callable[[int], Iterator[M.Batch]]:
    def gen(ep: int) -> Iterator[M.Batch]:
        rng = np.random.default_rng(seed * 1000 + ep)
        for idx in iterate(len(X), batch, rng):
            t = int(rng.integers(8))
            x = torch.from_numpy(M.d4_grid(X[idx], t))
            y = torch.from_numpy(M.d4_grid(Y[idx], t))
            yield (x, prior_batch(ctx, len(idx), t)), y

    return gen


def aligned_eval_batches(ctx: Context, X: np.ndarray, batch: int = 50) -> Iterator[tuple]:
    for s in range(0, len(X), batch):
        x = torch.from_numpy(np.ascontiguousarray(X[s : s + batch]))
        yield (x, prior_batch(ctx, len(x), 0))


def ir_batches(
    ctx: Context, d: dict, Y: np.ndarray, batch: int, seed: int
) -> Callable[[int], Iterator[M.Batch]]:
    traces = ir_inputs(d["ir"])

    def gen(ep: int) -> Iterator[M.Batch]:
        rng = np.random.default_rng(seed * 1000 + ep)
        for idx in iterate(len(Y), batch, rng):
            t = int(rng.integers(8))
            src = torch.from_numpy(M.d4_points(d["sources"][idx], t, N_GRID).astype(np.float32))
            mic = torch.from_numpy(M.d4_points(d["mics"][idx], t, N_GRID).astype(np.float32))
            y = torch.from_numpy(M.d4_grid(Y[idx], t))
            yield (torch.from_numpy(traces[idx]), src, mic, prior_batch(ctx, len(idx), t)), y

    return gen


def ir_eval_batches(ctx: Context, d: dict, k: int, batch: int = 50) -> Iterator[tuple]:
    traces = ir_inputs(d["ir"][:, :k])
    for s in range(0, len(traces), batch):
        sl = slice(s, s + batch)
        yield (
            torch.from_numpy(traces[sl]),
            torch.from_numpy(d["sources"][sl, :k].astype(np.float32)),
            torch.from_numpy(d["mics"][sl, :k].astype(np.float32)),
            prior_batch(ctx, len(traces[sl]), 0),
        )


def pose_inputs(images: np.ndarray, sources: np.ndarray, mics: np.ndarray) -> np.ndarray:
    """``(B, K, 4 + 3, H, W)`` per-pose standardised images plus geometry channels."""
    B, K = images.shape[:2]
    geo = np.stack(
        [
            np.stack(
                [
                    geometry_channels((N_GRID, N_GRID), sources[b, k], mics[b, k], DT)
                    for k in range(K)
                ]
            )
            for b in range(B)
        ]
    )
    return np.concatenate([zscore_maps(images), geo], 2).astype(np.float32)


def set_batches(
    ctx: Context, d: dict, batch: int, seed: int, max_k: int = 4
) -> Callable[[int], Iterator[M.Batch]]:
    def gen(ep: int) -> Iterator[M.Batch]:
        rng = np.random.default_rng(seed * 1000 + ep)
        for idx in iterate(len(d["mask"]), batch, rng):
            t = int(rng.integers(8))
            k = int(rng.integers(1, max_k + 1))
            sel = np.stack([rng.permutation(d["images"].shape[1])[:k] for _ in idx])
            rows = idx[:, None]
            img = M.d4_grid(d["images"][rows, sel], t)
            src = M.d4_points(d["sources"][rows, sel], t, N_GRID)
            mic = M.d4_points(d["mics"][rows, sel], t, N_GRID)
            y = np.stack([d["mask"][idx], d["illum_pose"][rows, sel].any(1)], 1)
            y = torch.from_numpy(M.d4_grid(y.astype(np.float32), t))
            x = torch.from_numpy(pose_inputs(img, src, mic))
            yield (x, prior_batch(ctx, len(idx), t)), y

    return gen


def set_eval_batches(
    ctx: Context, d: dict, k: int, batch: int = 25, override: dict | None = None
) -> Iterator[tuple]:
    src_all = d["sources"] if override is None else override["sources"]
    mic_all = d["mics"] if override is None else override["mics"]
    for s in range(0, len(d["mask"]), batch):
        sl = slice(s, s + batch)
        x = pose_inputs(d["images"][sl, :k], src_all[sl, :k], mic_all[sl, :k])
        yield (torch.from_numpy(x), prior_batch(ctx, x.shape[0], 0))


# ---------------------------------------------------------------------------
# Model stages
# ---------------------------------------------------------------------------


def train_or_load(
    ctx: Context,
    name: str,
    build: Callable[[], torch.nn.Module],
    batches: Callable[[], Callable[[int], Iterator[M.Batch]]],
    epochs: int,
    seed: int,
) -> torch.nn.Module:
    path = ctx.cache / "models" / f"{name}.pt"
    torch.manual_seed(seed)
    model = build()
    if path.exists():
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    path.parent.mkdir(exist_ok=True)
    LOG(f"training {name} ({M.n_params(model)} parameters, {epochs} epochs)")
    t0 = time.perf_counter()
    hist = M.fit(model, batches(), epochs, lr=ctx.args.lr, log=LOG)
    torch.save(model.state_dict(), path)
    meta = {"history": hist, "train_s": time.perf_counter() - t0, "params": M.n_params(model)}
    (ctx.cache / "models" / f"{name}.json").write_text(json.dumps(meta))
    model.eval()
    return model


def model_meta(ctx: Context, name: str) -> dict:
    p = ctx.cache / "models" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def logits_cached(ctx: Context, key: str, fn: Callable[[], np.ndarray]) -> np.ndarray:
    path = ctx.cache / "logits" / f"{key}.npy"
    if path.exists():
        return np.load(path)
    path.parent.mkdir(exist_ok=True)
    z = fn().astype(np.float32)
    np.save(path, z)
    return z


def score_method(
    ctx: Context,
    method: str,
    val_logits: np.ndarray,
    held_logits: dict[int, np.ndarray],
    temperature: tuple[float, float] = (1.0, 1.0),
) -> None:
    """Threshold on validation rooms, score each target and K on held-out rooms."""
    for ti, target in enumerate(TARGETS):
        tau = ctx.scorer.tau(sigmoid(val_logits[:, ti] / temperature[ti]), target)
        for k, z in held_logits.items():
            if target == "illum" and k != 4:
                continue
            e = ctx.scorer.add(
                method,
                target,
                k,
                sigmoid(z[:, ti] / temperature[ti]),
                ctx.held_truth(target, k),
                tau,
            )
            LOG(f"  {method:22s} {target:5s} K={k}: {fmt(e)}")


def logistic_probs(ev, images: np.ndarray, k: int | None = None) -> np.ndarray:
    """The all-physics logistic fusion evaluated on per-pose images ``(N, K, 4, H, W)``.

    Equal to ``ev.probs(d, "all")`` when the chosen carving variant is the
    per-pose default (raw residual onset at 1e-3), which the training-room
    selection picks for both targets; it avoids ``Evaluator``'s carving cache,
    which is keyed by ``id()`` of the feature dictionary and can return stale
    images when a dictionary is freed and a new one reuses its id.
    """
    c = ev.choices
    for n in POSE_CHANNELS:
        th = c[n]["threshold"]
        assert th is None or tuple(th) == ("raw", 1e-3), th
    agg = aggregate(images, k)
    feats = [EI.zscore(-agg[n] if n == "carving" else agg[n], c[n]["sigma"]) for n in POSE_CHANNELS]
    off = np.broadcast_to(EI.logit(ev.prior), feats[0].shape)
    return EI.predict(off, feats, c["all"]["w"])


def stage_logistic(ctx: Context) -> None:
    """Prior and the all-physics logistic fusion, K = 1, 2, 4, 8 (K = 4 fit)."""
    val = eval_dict(ctx.val, 4)
    res: dict = {}
    ctx.evaluators = {}
    for target in TARGETS:
        ev = EI.Evaluator(val, ctx.prior[target], target)
        ev.fit_all()
        tau_p = EI.tune_tau(ev.probs(val, "prior"), val[target])
        tau_a = EI.tune_tau(logistic_probs(ev, ctx.val["images"], 4), val[target])
        ev.carve_cache.clear()
        hd4 = eval_dict(ctx.held, 4)
        check = float(
            np.abs(ev.probs(hd4, "all") - logistic_probs(ev, ctx.held["images"], 4)).max()
        )
        ev.carve_cache.clear()
        res[target] = {
            "w": [float(x) for x in ev.choices["all"]["w"]],
            "tau": tau_a,
            "tau_prior": tau_p,
            "check_max_abs_diff": check,
            "choices": {
                n: {"sigma": ev.choices[n]["sigma"], "threshold": ev.choices[n]["threshold"]}
                for n in POSE_CHANNELS
            },
        }
        for k in (1, 2, 4, 8):
            if target == "illum" and k != 4:
                continue
            truth = ctx.held_truth(target, k)
            prior = np.broadcast_to(ctx.prior[target], truth.shape).copy()
            ctx.scorer.add("prior", target, k, prior, truth, tau_p)
            p = logistic_probs(ev, ctx.held["images"], k)
            e = ctx.scorer.add("logistic", target, k, p, truth, tau_a)
            LOG(f"  logistic all physics {target} K={k}: {fmt(e)}")
            np.save(ctx.cache / f"logistic_{target}_k{k}.npy", p.astype(np.float32))
        ctx.evaluators[target] = ev
    ctx.results["logistic"] = res


def stage_unet(ctx: Context, seeds: tuple[int, ...]) -> None:
    """6.3.1: U-Net on the pose-summed aligned images (K = 4 training)."""
    a = ctx.args
    tr = ctx.train
    X = aligned_inputs(tr["images"], 4)
    Y = targets(tr, 4)
    Xv = aligned_inputs(ctx.val["images"], 4)
    for seed in seeds:
        name = f"unet_s{seed}"
        model = train_or_load(
            ctx,
            name,
            lambda: M.AlignedUNet(4, a.width, dropout=a.dropout),
            lambda seed=seed: aligned_batches(ctx, X, Y, a.batch, seed),
            a.epochs,
            seed,
        )
        logits_cached(
            ctx, f"{name}_val", lambda m=model: M.predict_logits([m], aligned_eval_batches(ctx, Xv))
        )
        for k in (1, 2, 4, 8):
            Xh = aligned_inputs(ctx.held["images"], k)
            logits_cached(
                ctx,
                f"{name}_held_k{k}",
                lambda m=model, Xh=Xh: M.predict_logits([m], aligned_eval_batches(ctx, Xh)),
            )
    s0 = f"unet_s{seeds[0]}"
    score_method(
        ctx,
        "unet",
        np.load(ctx.cache / "logits" / f"{s0}_val.npy"),
        {k: np.load(ctx.cache / "logits" / f"{s0}_held_k{k}.npy") for k in (1, 2, 4, 8)},
    )
    ctx.results.setdefault("models", {})["unet"] = model_meta(ctx, s0)


def stage_unet_control(ctx: Context) -> None:
    """Budget-matched control for 6.3.3: the U-Net trained for the set model's epoch count."""
    a = ctx.args
    tr = ctx.train
    X = aligned_inputs(tr["images"], 4)
    Y = targets(tr, 4)
    name = f"unet_e{a.epochs_set}"
    model = train_or_load(
        ctx,
        name,
        lambda: M.AlignedUNet(4, a.width, dropout=a.dropout),
        lambda: aligned_batches(ctx, X, Y, a.batch, 0),
        a.epochs_set,
        0,
    )
    zv = logits_cached(
        ctx,
        f"{name}_val",
        lambda: M.predict_logits(
            [model], aligned_eval_batches(ctx, aligned_inputs(ctx.val["images"], 4))
        ),
    )
    zh = {
        k: logits_cached(
            ctx,
            f"{name}_held_k{k}",
            lambda k=k: M.predict_logits(
                [model], aligned_eval_batches(ctx, aligned_inputs(ctx.held["images"], k))
            ),
        )
        for k in (1, 2, 4, 8)
    }
    score_method(ctx, f"unet ({a.epochs_set} epochs)", zv, zh)
    ctx.results.setdefault("models", {})[name] = model_meta(ctx, name)


def stage_irnet(ctx: Context) -> None:
    """6.3.2: impulse responses as inputs (learned migration, and a global encoder)."""
    a = ctx.args
    tr = ctx.train
    Y = targets(tr, 4)
    for name, build in (
        ("irmig", lambda: M.IRMigrationNet((N_GRID, N_GRID), DT, 8, a.width)),
        ("irglobal", lambda: M.IRGlobalNet(2, (N_GRID, N_GRID))),
    ):
        model = train_or_load(
            ctx, name, build, lambda: ir_batches(ctx, tr, Y, a.batch, 0), a.epochs_ir, 0
        )
        zv = logits_cached(
            ctx,
            f"{name}_val",
            lambda m=model: M.predict_logits([m], ir_eval_batches(ctx, ctx.val, 4)),
        )
        zh = {
            k: logits_cached(
                ctx,
                f"{name}_held_k{k}",
                lambda m=model, k=k: M.predict_logits([m], ir_eval_batches(ctx, ctx.held, k)),
            )
            for k in (1, 2, 4, 8)
        }
        score_method(ctx, name, zv, zh)
        ctx.results.setdefault("models", {})[name] = model_meta(ctx, name)


def stage_setnet(ctx: Context) -> None:
    """6.3.3: pose-aware set model, trained on random subsets of 1-4 poses."""
    a = ctx.args
    tr = ctx.train
    model = train_or_load(
        ctx,
        "setnet",
        lambda: M.PoseSetNet(7, a.width, a.width),
        lambda: set_batches(ctx, tr, a.batch, 0),
        a.epochs_set,
        0,
    )
    zv = logits_cached(
        ctx, "setnet_val", lambda: M.predict_logits([model], set_eval_batches(ctx, ctx.val, 4))
    )
    zh = {
        k: logits_cached(
            ctx,
            f"setnet_held_k{k}",
            lambda k=k: M.predict_logits([model], set_eval_batches(ctx, ctx.held, k)),
        )
        for k in (1, 2, 4, 8)
    }
    score_method(ctx, "setnet", zv, zh)
    ctx.results.setdefault("models", {})["setnet"] = model_meta(ctx, "setnet")


def stage_uncertainty(ctx: Context, seeds: tuple[int, ...], mc: int) -> None:
    """6.3.4: MC dropout and a deep ensemble of U-Nets, each with a temperature."""
    a = ctx.args
    Xv = aligned_inputs(ctx.val["images"], 4)
    Xh = aligned_inputs(ctx.held["images"], 4)
    members = []
    for seed in seeds:
        m = M.AlignedUNet(4, a.width, dropout=a.dropout)
        m.load_state_dict(torch.load(ctx.cache / "models" / f"unet_s{seed}.pt"))
        members.append(m)
    variants = {
        "unet+T": ([members[0]], 0),
        f"unet MC-dropout x{mc} +T": ([members[0]], mc),
        f"unet ensemble x{len(seeds)} +T": (members, 0),
    }
    out: dict = {}
    for name, (ms, n_mc) in variants.items():
        key = name.split(" +T")[0].replace(" ", "_").replace("+", "")
        torch.manual_seed(0)
        zv = logits_cached(
            ctx,
            f"unc_{key}_val",
            lambda ms=ms, n_mc=n_mc: M.predict_logits(ms, aligned_eval_batches(ctx, Xv), n_mc),
        )
        torch.manual_seed(1)
        zh = logits_cached(
            ctx,
            f"unc_{key}_held",
            lambda ms=ms, n_mc=n_mc: M.predict_logits(ms, aligned_eval_batches(ctx, Xh), n_mc),
        )
        temps = (
            M.fit_temperature(zv[:, 0], ctx.val_y[:, 0]),
            M.fit_temperature(zv[:, 1], ctx.val_y[:, 1]),
        )
        score_method(ctx, name, zv, {4: zh}, temps)
        cal = {}
        for i, target in enumerate(TARGETS):
            truth = ctx.held_truth(target, 4)
            for tag, T in (("raw", 1.0), ("T", temps[i])):
                mp, fy, w, ece = M.reliability(sigmoid(zh[:, i] / T), truth)
                cal[f"{target}_{tag}"] = {
                    "ece": ece,
                    "pred": mp.tolist(),
                    "freq": fy.tolist(),
                    "weight": w.tolist(),
                }
            # Mean predictive entropy against error rate (does uncertainty track mistakes?).
            p = sigmoid(zh[:, i] / temps[i])
            ent = -(p * np.log2(p + 1e-12) + (1 - p) * np.log2(1 - p + 1e-12))
            err = np.abs(truth - p)
            cal[f"{target}_entropy_error_corr"] = float(np.corrcoef(ent.ravel(), err.ravel())[0, 1])
        out[name] = {"temperature": temps, "calibration": cal}
    # Prior and logistic reliability for reference.
    for i, target in enumerate(TARGETS):
        truth = ctx.held_truth(target, 4)
        p_log = np.load(ctx.cache / f"logistic_{target}_k4.npy")
        mp, fy, w, ece = M.reliability(p_log, truth)
        out.setdefault("logistic", {})[target] = {
            "ece": ece,
            "pred": mp.tolist(),
            "freq": fy.tolist(),
        }
        pp = np.broadcast_to(ctx.prior[target], truth.shape)
        mp, fy, w, ece = M.reliability(pp, truth)
        out.setdefault("prior", {})[target] = {"ece": ece, "pred": mp.tolist(), "freq": fy.tolist()}
    ctx.results["uncertainty"] = out


# ---------------------------------------------------------------------------
# Paired statistics helper
# ---------------------------------------------------------------------------


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean ± SE and z of the per-room difference ``a - b``."""
    m, se = mean_se(np.asarray(a) - np.asarray(b))
    return {"delta": m, "se": se, "z": m / se if se > 0 else float("nan")}


def room_scores(prob: np.ndarray, truth: np.ndarray, tau: float, prior: np.ndarray) -> dict:
    return EI.score(prob, truth, tau, prior)


def summary(sc: dict) -> dict:
    return {k: dict(zip(("mean", "se"), mean_se(v))) for k, v in sc.items()}


# ---------------------------------------------------------------------------
# 6.6.3 Next-best pose
# ---------------------------------------------------------------------------


def binary_entropy(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def pose_centroid(d: dict, r: int, k: int) -> np.ndarray:
    return np.concatenate([d["sources"][r, k][None], d["mics"][r, k]]).mean(0)


def device_distance(d: dict, r: int, k: int) -> np.ndarray:
    """Distance of every cell to the nearest device of pose k (cells)."""
    ii, jj = np.meshgrid(np.arange(N_GRID), np.arange(N_GRID), indexing="ij")
    devs = np.concatenate([d["sources"][r, k][None], d["mics"][r, k]])
    return np.min([np.hypot(ii - p[0], jj - p[1]) for p in devs], axis=0)


def stage_nbv(ctx: Context) -> None:
    """6.6.3: greedy next-best pose from K = 2 to 4 among the 8 held-out poses.

    Strategies: (a) the web panel's heuristic, the candidate whose device
    centroid has the most binary entropy of the fused map within 12 cells;
    (b) sensitivity-weighted entropy :math:`\\sum_x H(q_x)\\,v(d_c(x))`, with
    :math:`v(d) = E[(\\Delta\\operatorname{logit})^2]` the mean squared change of
    the fused logit when a pose is added, as a function of the distance to the
    new pose's nearest device, learned on validation rooms (training archive);
    (c) random, scored exactly as the mean over all choices; and the
    hindsight oracle (best IoU) as a ceiling. The fused map is the logistic
    all-physics fusion (fitted at K = 4 on validation rooms).
    """
    ev = ctx.evaluators["mask"]
    tau = ctx.results["logistic"]["mask"]["tau"]
    prior = ctx.prior["mask"]
    # (b) sensitivity profile from validation rooms: add pose c in {2, 3} to {0, 1}.
    val = ctx.val
    z01 = EI.logit(logistic_probs(ev, val["images"][:, :2]))
    bins = np.arange(0, 70, 2.0)
    acc = np.zeros(len(bins))
    cnt = np.zeros(len(bins))
    for c in (2, 3):
        zc = EI.logit(logistic_probs(ev, val["images"][:, [0, 1, c]]))
        d2 = (zc - z01) ** 2
        for r in range(len(d2)):
            b = np.minimum((device_distance(val, r, c) / 2).astype(int), len(bins) - 1)
            acc += np.bincount(b.ravel(), d2[r].ravel(), len(bins))
            cnt += np.bincount(b.ravel(), minlength=len(bins))
    profile = acc / np.maximum(cnt, 1)
    held = ctx.held
    truth = held["mask"]
    n = len(truth)
    ii, jj = np.meshgrid(np.arange(N_GRID), np.arange(N_GRID), indexing="ij")

    def fused(r: int, poses: list[int]) -> np.ndarray:
        return logistic_probs(ev, held["images"][r : r + 1, poses])[0]

    def choose(r: int, q: np.ndarray, cand: list[int], rule: str) -> int:
        h = binary_entropy(q)
        scores = []
        for c in cand:
            if rule == "entropy12":
                cx = pose_centroid(held, r, c)
                scores.append(float(h[np.hypot(ii - cx[0], jj - cx[1]) <= 12].sum()))
            else:
                b = np.minimum((device_distance(held, r, c) / 2).astype(int), len(bins) - 1)
                scores.append(float((h * profile[b]).sum()))
        return cand[int(np.argmax(scores))]

    per: dict[str, dict[int, dict[str, list]]] = {}
    for r in range(n):
        q2 = fused(r, [0, 1])
        rest = list(range(2, 8))
        for rule in ("entropy12", "sensitivity"):
            c3 = choose(r, q2, rest, rule)
            q3 = fused(r, [0, 1, c3])
            c4 = choose(r, q3, [c for c in rest if c != c3], rule)
            q4 = fused(r, [0, 1, c3, c4])
            for k, q in ((3, q3), (4, q4)):
                sc = room_scores(q[None], truth[r : r + 1], tau, prior)
                e = per.setdefault(rule, {}).setdefault(k, {"iou": [], "ig": []})
                e["iou"].append(sc["iou"][0])
                e["ig"].append(sc["ig"][0])
        # Random (exact expectation) and hindsight oracle.
        for k, subsets in (
            (3, [[0, 1, c] for c in rest]),
            (4, [[0, 1, a, b] for i, a in enumerate(rest) for b in rest[i + 1 :]]),
        ):
            qs = logistic_probs(ev, np.stack([held["images"][r, s] for s in subsets]))
            sc = room_scores(qs, np.broadcast_to(truth[r], qs.shape), tau, prior)
            for name, fn in (("random", np.mean), ("oracle", np.max)):
                e = per.setdefault(name, {}).setdefault(k, {"iou": [], "ig": []})
                e["iou"].append(float(fn(sc["iou"])))
                e["ig"].append(
                    float(fn(sc["ig"]))
                    if name == "random"
                    else float(sc["ig"][np.argmax(sc["iou"])])
                )
        if (r + 1) % 100 == 0:
            LOG(f"  nbv {r + 1}/{n}")
    base = room_scores(np.stack([fused(r, [0, 1]) for r in range(n)]), truth, tau, prior)
    out: dict = {
        "profile_v": profile.tolist(),
        "profile_bins": bins.tolist(),
        "k2": summary({"iou": base["iou"], "ig": base["ig"]}),
    }
    for name, byk in per.items():
        for k, e in byk.items():
            entry: dict[str, dict[str, Any]] = {
                m: dict(zip(("mean", "se"), mean_se(np.array(v)))) for m, v in e.items()
            }
            if name != "random":
                for m in ("iou", "ig"):
                    entry[m]["vs_random"] = paired(np.array(e[m]), np.array(per["random"][k][m]))
            out.setdefault(name, {})[str(k)] = entry
            LOG(
                f"  nbv {name:11s} K={k}: IoU {entry['iou']['mean']:.4f} IG {entry['ig']['mean']:.1f}"
                + (
                    f" | dIoU vs random {entry['iou']['vs_random']['delta']:+.4f}±{entry['iou']['vs_random']['se']:.4f} "
                    f"(z {entry['iou']['vs_random']['z']:.1f})"
                    if "vs_random" in entry["iou"]
                    else ""
                )
            )
    ctx.results["nbv"] = out


# ---------------------------------------------------------------------------
# 6.6.4 Passive
# ---------------------------------------------------------------------------


def passive_cache(ctx: Context, name: str, d: dict, k: int) -> np.ndarray:
    """Per-pose passive images ``(N, K, 2, H, W)`` (raw, background removed)."""
    from acoustic_system.imaging.passive import passive_images

    path = ctx.cache / f"passive_{name}.npy"
    if path.exists():
        return np.load(path)
    out = np.zeros((len(d["mask"]), k, 2, N_GRID, N_GRID), dtype=np.float32)
    t0 = time.perf_counter()
    for r in range(len(d["mask"])):
        for kk in range(k):
            raw, sub = passive_images(
                (N_GRID, N_GRID),
                d["sources"][r, kk : kk + 1],
                d["mics"][r, kk : kk + 1],
                d["rec"][r, kk : kk + 1].astype(np.float64),
                DT,
            )
            out[r, kk, 0] = raw
            out[r, kk, 1] = sub
        if (r + 1) % 100 == 0:
            LOG(f"  passive {name} {r + 1}/{len(d['mask'])} ({time.perf_counter() - t0:.0f} s)")
    np.save(path, out)
    return out


def stage_passive(ctx: Context) -> None:
    """6.6.4: GCC-PHAT interferometric images (source signal unknown) fused with the prior."""
    pv = passive_cache(ctx, "val", ctx.val, 4)
    ph = passive_cache(ctx, "held", ctx.held, 8)
    out: dict = {}
    for target in ("mask",):
        prior = ctx.prior[target]
        yv = ctx.val["mask"]
        off_v = np.broadcast_to(EI.logit(prior), yv.shape)
        feats_def = {"passive raw": (0,), "passive bg-removed": (1,), "passive both": (0, 1)}
        for name, chans in feats_def.items():
            best = None
            for sig in (0.0, 1.0, 2.0, 3.0):
                fv = [EI.zscore(pv[:, :4, c].sum(1), sig) for c in chans]
                w, loss = EI.fit_logistic(off_v, fv, yv)
                if best is None or loss < best[2]:
                    best = (sig, w, loss)
            assert best is not None
            sig, w, loss = best
            fv = [EI.zscore(pv[:, :4, c].sum(1), sig) for c in chans]
            tau = EI.tune_tau(EI.predict(off_v, fv, w), yv)
            for k in (4, 8):
                fh = [EI.zscore(ph[:, :k, c].sum(1), sig) for c in chans]
                p = EI.predict(np.broadcast_to(EI.logit(prior), fh[0].shape), fh, w)
                e = ctx.scorer.add(name, target, k, p, ctx.held_truth(target, k), tau)
                LOG(f"  {name:20s} K={k}: {fmt(e)}")
            out[name] = {
                "sigma": sig,
                "w": [float(x) for x in w],
                "train_logloss": loss,
                "tau": tau,
            }
            # Standalone AP (image alone, signed by its weight).
            ap_vals = [
                EI.average_precision(np.sign(w[0]) * z, t)
                for z, t in zip(EI.zscore(ph[:, :4, chans[0]].sum(1), sig), ctx.held["mask"])
            ]
            out[name]["standalone_ap"] = dict(zip(("mean", "se"), mean_se(np.array(ap_vals))))
    ctx.results["passive"] = out


def stage_passive_net(ctx: Context) -> None:
    """6.6.4, learned: the compact U-Net on the two passive images (no source signal)."""
    a = ctx.args
    parts = []
    for start in range(CHUNK, a.n_train_rooms, CHUNK):
        d = load_npz([train_chunk_path(ctx.cache, start)], ("rec", "mask", "sources", "mics"))
        parts.append(passive_cache(ctx, f"train_{start:05d}", d, 4))
    P = np.concatenate(parts)

    def inputs(pimg: np.ndarray, k: int) -> np.ndarray:
        return np.stack([zscore_maps(pimg[:, :k, c].sum(1)) for c in (0, 1)], 1)

    X = inputs(P, 4)
    Y = targets(ctx.train, 4)
    name = f"passive_unet_e{a.epochs_set}"
    model = train_or_load(
        ctx,
        name,
        lambda: M.AlignedUNet(2, a.width, dropout=a.dropout),
        lambda: aligned_batches(ctx, X, Y, a.batch, 0),
        a.epochs_set,
        0,
    )
    pv = passive_cache(ctx, "val", ctx.val, 4)
    ph = passive_cache(ctx, "held", ctx.held, 8)
    zv = logits_cached(
        ctx,
        f"{name}_val",
        lambda: M.predict_logits([model], aligned_eval_batches(ctx, inputs(pv, 4))),
    )
    zh = {
        k: logits_cached(
            ctx,
            f"{name}_held_k{k}",
            lambda k=k: M.predict_logits([model], aligned_eval_batches(ctx, inputs(ph, k))),
        )
        for k in (4, 8)
    }
    score_method(ctx, "passive U-Net", zv, zh)
    ctx.results.setdefault("models", {})[name] = model_meta(ctx, name)


# ---------------------------------------------------------------------------
# 6.6.1 / 6.6.2 Pose error and refinement
# ---------------------------------------------------------------------------


def pose_variants(ctx: Context, sigma: float) -> dict:
    """Perturbed, alternating-refined and joint-refined images for held-out rooms (K = 4)."""
    from acoustic_system.imaging.pose_images import perturb_poses
    from acoustic_system.imaging.pose_refine import refine_room

    n = ctx.args.n_pose_rooms
    path = ctx.cache / f"pose_s{sigma:g}_n{n}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    rng = np.random.default_rng(10_000 + int(round(10 * sigma)))
    radius = int(np.ceil(2 * sigma))
    out: dict[str, list] = {}
    dec = None
    t0 = time.perf_counter()
    with h5py.File(HELD_ARCHIVE, "r") as hf:
        for r in range(n):
            room = ArchiveRoom.from_group(hf[f"sample_{r:04d}"], 4)
            if dec is None:
                dec = TikhonovDeconvolver(room.drive, room.recordings.shape[-1], lam=1e-2)
            pr, _ = perturb_poses(room, sigma, rng)
            variants = {"perturbed": (pr, None)}
            for method in ("alternating", "joint"):
                ref, info = refine_room(pr, radius, method=method)
                variants[method] = (ref, [i.incident for i in info])
            for name, (rm, inc) in variants.items():
                im = compute_pose_images(rm, dec, incident=inc).images.astype(np.float32)
                out.setdefault(f"{name}_images", []).append(im)
                out.setdefault(f"{name}_sources", []).append(rm.sources)
                out.setdefault(f"{name}_mics", []).append(rm.mics)
            if (r + 1) % 50 == 0:
                LOG(f"  pose sigma={sigma:g}: {r + 1}/{n} ({time.perf_counter() - t0:.0f} s)")
    arr = {k: np.stack(v) for k, v in out.items()}
    np.savez(path, allow_pickle=False, **arr)
    return arr


def stage_pose(ctx: Context, seeds: tuple[int, ...]) -> None:
    """6.6.1 (degradation under pose error) and 6.6.2 (refinement), K = 4, filled mask."""
    a = ctx.args
    n = a.n_pose_rooms
    ev = ctx.evaluators["mask"]
    truth = ctx.held["mask"][:n]
    prior = ctx.prior["mask"]
    unet = M.AlignedUNet(4, a.width, dropout=a.dropout)
    unet.load_state_dict(torch.load(ctx.cache / "models" / f"unet_s{seeds[0]}.pt"))
    setnet = None
    if (ctx.cache / "models" / "setnet.pt").exists():
        setnet = M.PoseSetNet(7, a.width, a.width)
        setnet.load_state_dict(torch.load(ctx.cache / "models" / "setnet.pt"))
    taus = {
        m: ctx.scorer.table["mask"]["4"][m]["tau"]
        for m in ("logistic", "unet", "setnet")
        if m in ctx.scorer.table["mask"]["4"]
    }
    true_dev = np.concatenate([ctx.held["sources"][:n, :4, None], ctx.held["mics"][:n, :4]], 2)

    def probs(method: str, images: np.ndarray, sources: np.ndarray, mics: np.ndarray) -> np.ndarray:
        if method == "logistic":
            return logistic_probs(ev, images)
        if method == "unet":
            X = aligned_inputs(images, 4)
            return sigmoid(M.predict_logits([unet], aligned_eval_batches(ctx, X))[:, 0])
        assert setnet is not None
        d = {"images": images, "mask": truth}
        z = M.predict_logits(
            [setnet], set_eval_batches(ctx, d, 4, override={"sources": sources, "mics": mics})
        )
        return sigmoid(z[:, 0])

    methods = [m for m in ("logistic", "unet", "setnet") if m in taus]
    exact = {
        m: room_scores(
            probs(
                m, ctx.held["images"][:n, :4], ctx.held["sources"][:n, :4], ctx.held["mics"][:n, :4]
            ),
            truth,
            taus[m],
            prior,
        )
        for m in methods
    }
    prior_sc = room_scores(
        np.broadcast_to(prior, truth.shape).copy(),
        truth,
        ctx.scorer.table["mask"]["4"]["prior"]["tau"],
        prior,
    )
    out: dict = {"n_rooms": n, "exact": {m: summary(v) for m, v in exact.items()}}
    for sigma in a.pose_sigmas:
        var = pose_variants(ctx, sigma)
        so: dict = {}
        for vname in ("perturbed", "alternating", "joint"):
            dev = np.concatenate([var[f"{vname}_sources"][:, :, None], var[f"{vname}_mics"]], 2)
            err = np.linalg.norm(dev - true_dev, axis=-1)
            vo: dict = {
                "pose_error_mean": float(err.mean()),
                "pose_exact_frac": float((err == 0).mean()),
            }
            for m in methods:
                sc = room_scores(
                    probs(m, var[f"{vname}_images"], var[f"{vname}_sources"], var[f"{vname}_mics"]),
                    truth,
                    taus[m],
                    prior,
                )
                vo[m] = summary(sc) | {
                    "vs_exact": paired(sc["iou"], exact[m]["iou"]),
                    "vs_prior": paired(sc["iou"], prior_sc["iou"]),
                    "ap_vs_exact": paired(sc["ap"], exact[m]["ap"]),
                }
                if vname != "perturbed":
                    pert = so["perturbed"][m]["iou"]["mean"]
                    ex = float(np.mean(exact[m]["iou"]))
                    vo[m]["recovered_frac"] = (float(np.mean(sc["iou"])) - pert) / (ex - pert)
                LOG(
                    f"  sigma={sigma:g} {vname:11s} {m:8s} IoU {np.mean(sc['iou']):.4f} "
                    f"(exact {np.mean(exact[m]['iou']):.4f}) pose err {vo['pose_error_mean']:.2f}"
                )
            so[vname] = vo
        out[f"sigma_{sigma:g}"] = so
    out["dev_checks"] = pose_dev_checks(ctx)
    ctx.results["pose"] = out


def pose_dev_checks(ctx: Context, n_rooms: int = 10, sigma: float = 1.0) -> dict:
    """Why exact pose recovery fails, on validation (training-archive) rooms.

    (1) Lag of the first scattered arrival (carving onset detector) minus
    the direct-path lag, over all validation traces. (2) Fraction of devices
    on their exact cell after the joint fit with the empty box and with the
    *true* obstacle map in the model.
    """
    from acoustic_system.imaging.image_source import first_arrival
    from acoustic_system.imaging.pose_images import perturb_poses
    from acoustic_system.imaging.pose_refine import refine_pose_joint

    v = ctx.val
    gaps = []
    for r in range(len(v["mask"])):
        for k in range(v["residual"].shape[1]):
            for m in range(v["residual"].shape[2]):
                j1 = first_arrival(v["residual"][r, k, m], 1e-3)
                if j1 is not None:
                    d = np.hypot(*(v["sources"][r, k] - v["mics"][r, k, m])) / DT
                    gaps.append(j1 - d)
    g = np.array(gaps)
    rng = np.random.default_rng(777)
    radius = int(np.ceil(2 * sigma))
    hits: dict[str, list] = {"assumed": [], "empty box": [], "true map": []}
    with h5py.File(TRAIN_ARCHIVE, "r") as hf:
        for r in range(n_rooms):
            room = ArchiveRoom.from_group(hf[f"sample_{r:04d}"], 4)
            pr, _ = perturb_poses(room, sigma, rng)
            for k in range(4):
                true = np.concatenate([room.sources[k][None], room.mics[k]])
                hits["assumed"] += list(
                    (np.concatenate([pr.sources[k][None], pr.mics[k]]) == true).all(-1)
                )
                for name, mask in (("empty box", None), ("true map", room.mask)):
                    res = refine_pose_joint(
                        room.mask.shape,
                        room.recordings[k],
                        room.drive,
                        pr.sources[k],
                        pr.mics[k],
                        radius,
                        mask=mask,
                    )
                    est = np.concatenate([res.source[None], res.mics])
                    hits[name] += list((est == true).all(-1))
    return {
        "onset_minus_direct_lag_percentiles": dict(
            zip(("p5", "p25", "p50", "p75", "p95"), np.percentile(g, [5, 25, 50, 75, 95]).tolist())
        ),
        "onset_before_direct_frac": float((g < 0).mean()),
        "joint_exact_frac_sigma1": {k: float(np.mean(v)) for k, v in hits.items()},
        "n_rooms": n_rooms,
    }


# ---------------------------------------------------------------------------
# 6.6.5 3D rooms
# ---------------------------------------------------------------------------


def zscore3(x: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    from scipy import ndimage

    x = np.asarray(x, dtype=np.float64)
    if sigma > 0:
        x = ndimage.gaussian_filter(x, (0, sigma, sigma, sigma))
    f = x.reshape(len(x), -1)
    f = (f - f.mean(1, keepdims=True)) / (f.std(1, keepdims=True) + 1e-12)
    return f.reshape(x.shape)


def room3d_data(ctx: Context, split: str, n_rooms: int, seed: int) -> dict:
    from acoustic_system.imaging.room3d import (
        drive_offsets,
        image_room_3d,
        random_pose_3d,
        random_room_3d,
        record_3d,
        ricker,
    )

    a = ctx.args
    n = a.n3d
    path = ctx.cache / f"room3d_{split}_n{n}_{n_rooms}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    rng = np.random.default_rng(seed)
    T = a.steps3d
    f0 = 0.12
    drive = 5.0 * ricker(T, DT, f0)
    peak, onset = drive_offsets(drive)
    shape = (n, n, n)
    out: dict[str, list] = {k: [] for k in ("mask", "bp", "carve", "sources", "mics")}
    t0 = time.perf_counter()
    for r in range(n_rooms):
        mask = random_room_3d(n, rng)
        srcs, mics, res = [], [], []
        for _ in range(4):
            s, m = random_pose_3d(mask, rng)
            y = record_3d(mask, s, m, drive, shape)
            y0 = record_3d(None, s, m, drive, shape)
            srcs.append(s)
            mics.append(m)
            res.append(y - y0)
        srcs_a, mics_a, res_a = np.stack(srcs), np.stack(mics), np.stack(res)
        # Per-pose images so the pose count can be varied.
        bp, carve = [], []
        for k in range(4):
            im = image_room_3d(
                shape, srcs_a[k : k + 1], mics_a[k : k + 1], res_a[k : k + 1], DT, peak, onset
            )
            bp.append(im.backprojection.astype(np.float32))
            carve.append(im.carving.astype(np.float32))
        out["mask"].append(mask)
        out["bp"].append(np.stack(bp))
        out["carve"].append(np.stack(carve))
        out["sources"].append(srcs_a)
        out["mics"].append(mics_a)
        if (r + 1) % 25 == 0:
            LOG(f"  3D {split}: {r + 1}/{n_rooms} ({time.perf_counter() - t0:.0f} s)")
    arr = {k: np.stack(v) for k, v in out.items()}
    np.savez(path, allow_pickle=False, **arr)
    return arr


def score3d(prob: np.ndarray, truth: np.ndarray, tau: float, prior: np.ndarray) -> dict:
    """Per-room voxel IoU, AP and information gain (boundary F is 2D only)."""
    return {
        "iou": np.array([EI.iou(p >= tau, t) for p, t in zip(prob, truth)]),
        "ap": np.array([EI.average_precision(p, t) for p, t in zip(prob, truth)]),
        "ig": np.array([EI.info_gain_bits(p, t, prior) for p, t in zip(prob, truth)]),
    }


def stage_room3d(ctx: Context) -> None:
    """6.6.5: 3D back-projection and carving fused with a 3D prior."""
    from acoustic_system.imaging.room3d import random_room_3d

    a = ctx.args
    tr = room3d_data(ctx, "train", a.n3d_train, 3001)
    ho = room3d_data(ctx, "held", a.n3d_held, 3002)
    prior_rng = np.random.default_rng(3003)
    prior = np.mean([random_room_3d(a.n3d, prior_rng) for _ in range(2000)], axis=0)
    y_tr = tr["mask"]
    out: dict = {
        "n_train": int(len(y_tr)),
        "n_held": int(len(ho["mask"])),
        "grid": a.n3d,
        "mean_occupancy_held": float(ho["mask"].mean()),
    }
    feats_def = {"backprojection": ("bp",), "carving": ("carve",), "both": ("bp", "carve")}

    def feats(d: dict, chans: tuple[str, ...], sig: float, k: int) -> list[np.ndarray]:
        return [zscore3((-1 if c == "carve" else 1) * d[c][:, :k].sum(1), sig) for c in chans]

    off_tr = np.broadcast_to(EI.logit(prior), y_tr.shape)
    tau_p = EI.tune_tau(np.broadcast_to(prior, y_tr.shape), y_tr)
    base = {}
    for k in (1, 2, 4):
        base[k] = score3d(np.broadcast_to(prior, ho["mask"].shape), ho["mask"], tau_p, prior)
    out["prior"] = summary(base[4])
    for name, chans in feats_def.items():
        best = None
        for sig in (0.0, 1.0, 2.0):
            w, loss = EI.fit_logistic(off_tr, feats(tr, chans, sig, 4), y_tr)
            if best is None or loss < best[2]:
                best = (sig, w, loss)
        assert best is not None
        sig, w, loss = best
        tau = EI.tune_tau(EI.predict(off_tr, feats(tr, chans, sig, 4), w), y_tr)
        entry: dict = {"sigma": sig, "w": [float(x) for x in w], "tau": tau}
        for k in (1, 2, 4):
            p = EI.predict(
                np.broadcast_to(EI.logit(prior), ho["mask"].shape), feats(ho, chans, sig, k), w
            )
            sc = score3d(p, ho["mask"], tau, prior)
            entry[f"k{k}"] = summary(sc) | {
                "vs_prior_iou": paired(sc["iou"], base[k]["iou"]),
                "vs_prior_ap": paired(sc["ap"], base[k]["ap"]),
            }
            LOG(
                f"  3D {name:14s} K={k}: IoU {np.mean(sc['iou']):.4f} AP {np.nanmean(sc['ap']):.4f} "
                f"(prior {np.mean(base[k]['iou']):.4f}/{np.nanmean(base[k]['ap']):.4f}) "
                f"z_IoU {entry[f'k{k}']['vs_prior_iou']['z']:.1f}"
            )
        out[name] = entry
    ctx.results["room3d"] = out


# ---------------------------------------------------------------------------
# Report: leakage check, figures, results.json
# ---------------------------------------------------------------------------


def leakage_check(ctx: Context) -> dict:
    """Exact-duplicate masks between held-out rooms and the rooms used for training."""
    import hashlib

    def h(m: np.ndarray) -> str:
        return hashlib.md5(np.packbits(m.astype(bool)).tobytes()).hexdigest()

    held = {h(m) for m in ctx.held["mask"]}
    with h5py.File(TRAIN_ARCHIVE, "r") as hf:
        masks = np.stack(
            [hf[f"sample_{r:04d}"]["obstacles"][()] for r in range(ctx.args.n_train_rooms)]
        ).astype(bool)
    train = {h(m) for m in masks}
    # Nearest training layout of each held-out room (what memorisation could at best recall).
    H = ctx.held["mask"].reshape(len(ctx.held["mask"]), -1).astype(np.float32)
    Tm = masks.reshape(len(masks), -1).astype(np.float32)
    inter = H @ Tm.T
    union = H.sum(1)[:, None] + Tm.sum(1)[None] - inter
    nn_iou = (inter / np.maximum(union, 1)).max(1)
    return {
        "held_unique": len(held),
        "exact_duplicates_in_training_rooms": len(held & train),
        "nearest_training_mask_iou_median": float(np.median(nn_iou)),
        "nearest_training_mask_iou_max": float(nn_iou.max()),
    }


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_examples(ctx: Context, path: pathlib.Path, rooms=(0, 1, 2, 3)) -> None:
    plt = _plt()
    lg = ctx.cache / "logits"
    cols = [("truth", ctx.held["mask"].astype(float), "gray_r")]
    cols.append(("prior", np.broadcast_to(ctx.prior["mask"], ctx.held["mask"].shape), "magma"))
    cols.append(("logistic K=4", np.load(ctx.cache / "logistic_mask_k4.npy"), "magma"))
    for name, key in (
        ("U-Net K=4", "unet_s0_held_k4"),
        ("set model K=4", "setnet_held_k4"),
        ("set model K=8", "setnet_held_k8"),
    ):
        if (lg / f"{key}.npy").exists():
            cols.append((name, sigmoid(np.load(lg / f"{key}.npy")[:, 0]), "magma"))
    unc = [p for p in lg.glob("unc_unet_ensemble*_held.npy")]
    if unc:
        q = sigmoid(np.load(unc[0])[:, 0])
        cols.append(("ensemble entropy", binary_entropy(q), "viridis"))
    fig, ax = plt.subplots(len(rooms), len(cols), figsize=(2.0 * len(cols), 2.1 * len(rooms)))
    for i, r in enumerate(rooms):
        for j, (title, arr, cmap) in enumerate(cols):
            a = ax[i, j]
            a.imshow(arr[r], cmap=cmap, vmin=0, vmax=1)
            s, m = ctx.held["sources"][r, :4], ctx.held["mics"][r, :4]
            a.plot(s[:, 1], s[:, 0], "c*", ms=4)
            a.plot(m[:, :, 1].ravel(), m[:, :, 0].ravel(), "g.", ms=2)
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_k_sweep(ctx: Context, path: pathlib.Path) -> None:
    plt = _plt()
    tab = ctx.scorer.table["mask"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for m in ("prior", "logistic", "irglobal", "irmig", "unet", "setnet"):
        ks = [k for k in (1, 2, 4, 8) if m in tab.get(str(k), {})]
        if not ks:
            continue
        for a, metric in zip(ax, ("iou", "ap")):
            y = [tab[str(k)][m][metric]["mean"] for k in ks]
            e = [tab[str(k)][m][metric]["se"] for k in ks]
            a.errorbar(ks, y, yerr=e, marker="o", ms=3, capsize=2, label=m)
    for a, metric in zip(ax, ("IoU", "AP")):
        a.set_xscale("log", base=2)
        a.set_xticks([1, 2, 4, 8])
        a.set_xticklabels(["1", "2", "4", "8"])
        a.set_xlabel("poses K (held-out)")
        a.set_ylabel(f"filled-mask {metric}")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_reliability(ctx: Context, path: pathlib.Path) -> None:
    plt = _plt()
    unc = ctx.results.get("uncertainty", {})
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, target in zip(ax, TARGETS):
        a.plot([0, 1], [0, 1], "k:", lw=0.8)
        for name in ("prior", "logistic"):
            c = unc.get(name, {}).get(target)
            if c:
                a.plot(c["pred"], c["freq"], "o-", ms=3, label=f"{name} (ECE {c['ece']:.3f})")
        for name, v in unc.items():
            if name in ("prior", "logistic"):
                continue
            for tag in ("raw", "T"):
                c = v["calibration"][f"{target}_{tag}"]
                if tag == "raw" and not name.startswith("unet+T"):
                    continue
                label = f"{name.replace(' +T', '')}{' +T' if tag == 'T' else ' raw'} (ECE {c['ece']:.3f})"
                a.plot(c["pred"], c["freq"], "o-", ms=3, label=label)
        a.set_xscale("symlog", linthresh=0.01)
        a.set_yscale("symlog", linthresh=0.01)
        a.set_xlabel("predicted probability")
        a.set_ylabel("observed frequency")
        a.set_title(
            f"reliability, {'filled mask' if target == 'mask' else 'illuminated boundary'} (K=4)",
            fontsize=9,
        )
        a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_pose(ctx: Context, path: pathlib.Path) -> None:
    plt = _plt()
    po = ctx.results.get("pose")
    if not po:
        return
    sig = [float(k.split("_")[1]) for k in po if k.startswith("sigma_")]
    methods = [m for m in ("logistic", "unet", "setnet") if m in po["exact"]]
    fig, ax = plt.subplots(1, len(methods), figsize=(3.4 * len(methods), 3.2), squeeze=False)
    for a, m in zip(ax[0], methods):
        a.axhline(po["exact"][m]["iou"]["mean"], color="k", ls=":", label="exact poses")
        for v, st in (("perturbed", "o-"), ("alternating", "s--"), ("joint", "^-")):
            y = [po[f"sigma_{s:g}"][v][m]["iou"]["mean"] for s in sig]
            e = [po[f"sigma_{s:g}"][v][m]["iou"]["se"] for s in sig]
            a.errorbar(sig, y, yerr=e, fmt=st, ms=4, capsize=2, label=v)
        a.set_xlabel("pose error σ (cells, per coordinate)")
        a.set_ylabel("held-out IoU (K=4)")
        a.set_title(m, fontsize=9)
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_room3d(ctx: Context, path: pathlib.Path) -> None:
    plt = _plt()
    a = ctx.args
    p = ctx.cache / f"room3d_held_n{a.n3d}_{a.n3d_held}.npz"
    if not p.exists() or "room3d" not in ctx.results:
        return
    with np.load(p) as z:
        mask, bp, carve = z["mask"][:3], z["bp"][:3].sum(1), z["carve"][:3].sum(1)
    fig, ax = plt.subplots(3, 4, figsize=(8.4, 6.3))
    for i in range(3):
        zc = int(np.argmax(mask[i].sum((0, 1))))
        for j, (t, arr, cm) in enumerate(
            (
                ("truth (z-slice)", mask[i][:, :, zc], "gray_r"),
                ("back-projection", bp[i][:, :, zc], "magma"),
                ("carving count", carve[i][:, :, zc], "viridis"),
                ("truth, max over z", mask[i].max(2), "gray_r"),
            )
        ):
            ax[i, j].imshow(arr, cmap=cm)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if i == 0:
                ax[0, j].set_title(t, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


PAIRS = (
    ("unet", "logistic"),
    ("irmig", "unet"),
    ("irglobal", "prior"),
    ("setnet", "unet"),
    ("setnet", "unet (6 epochs)"),
    ("unet MC-dropout x16 +T", "unet"),
    ("unet ensemble x2 +T", "unet"),
    ("passive both", "prior"),
    ("passive U-Net", "prior"),
    ("passive U-Net", "passive both"),
)


def extra_pairs(ctx: Context) -> dict:
    """Paired per-room differences between methods other than the prior and logistic."""
    out: dict = {}
    for a, b in PAIRS:
        for (m, target, k), sc in ctx.scorer.per_room.items():
            if m != a or (b, target, k) not in ctx.scorer.per_room:
                continue
            base = ctx.scorer.per_room[(b, target, k)]
            out[f"{a} - {b} | {target} K={k}"] = {
                key: paired(sc[key], base[key]) for key in ("iou", "ap", "ig")
            }
    return out


def stage_report(ctx: Context) -> None:
    out = ctx.args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ctx.results["pairs"] = extra_pairs(ctx)
    ctx.results["leakage_check"] = leakage_check(ctx)
    fig_examples(ctx, out / "examples.png")
    fig_k_sweep(ctx, out / "k_sweep.png")
    fig_reliability(ctx, out / "reliability.png")
    fig_pose(ctx, out / "pose_robustness.png")
    fig_room3d(ctx, out / "room3d.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_STAGES = (
    "extract",
    "logistic",
    "unet",
    "irnet",
    "setnet",
    "unetctl",
    "uncertainty",
    "nbv",
    "passive",
    "passivenet",
    "poseimg",
    "pose",
    "room3d",
    "report",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path(tempfile.gettempdir()) / "acoustic_imaging_models_cache",
    )
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=_REPO_ROOT / "tests/reports/imaging_models_2026_09_24_artifacts",
    )
    ap.add_argument(
        "--stages", default="all", help="comma-separated subset of " + ",".join(ALL_STAGES)
    )
    ap.add_argument("--n-train-rooms", type=int, default=4000)
    ap.add_argument("--n-heldout", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--epochs-set", type=int, default=6)
    ap.add_argument("--epochs-ir", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seeds", default="0,1", help="U-Net ensemble seeds (first = the 6.3.1 model)")
    ap.add_argument("--mc", type=int, default=16)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--n-pose-rooms", type=int, default=500)
    ap.add_argument("--pose-sigmas", default="0.5,1,2,3")
    ap.add_argument("--n3d", type=int, default=32, help="3D grid size")
    ap.add_argument("--n3d-train", type=int, default=200)
    ap.add_argument("--n3d-held", type=int, default=100)
    ap.add_argument("--steps3d", type=int, default=240)
    args = ap.parse_args()
    if os.environ.get("NUMBA_NUM_THREADS") != "1":
        print("note: set NUMBA_NUM_THREADS=1 on a shared machine", file=sys.stderr)
    torch.set_num_threads(args.torch_threads)
    stages = ALL_STAGES if args.stages == "all" else tuple(args.stages.split(","))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    args.pose_sigmas = tuple(float(x) for x in args.pose_sigmas.split(","))
    t0 = time.perf_counter()
    if "extract" in stages:
        extract_all(args.cache_dir, args.n_train_rooms, args.n_heldout)
    if stages == ("extract",):
        return
    ctx = Context(args)
    stage_logistic(ctx)
    if "unet" in stages:
        stage_unet(ctx, seeds)
    if "irnet" in stages:
        stage_irnet(ctx)
    if "setnet" in stages:
        stage_setnet(ctx)
    if "unetctl" in stages:
        stage_unet_control(ctx)
    if "uncertainty" in stages:
        stage_uncertainty(ctx, seeds, args.mc)
    if "nbv" in stages:
        stage_nbv(ctx)
    if "passive" in stages:
        stage_passive(ctx)
    if "passivenet" in stages:
        stage_passive_net(ctx)
    if "poseimg" in stages:
        for sigma in args.pose_sigmas:
            pose_variants(ctx, sigma)
    if "pose" in stages:
        stage_pose(ctx, seeds)
    if "room3d" in stages:
        stage_room3d(ctx)
    ctx.results["table"] = ctx.scorer.table
    ctx.results["config"] = {k: str(v) for k, v in vars(args).items()}
    if "report" in stages:
        stage_report(ctx)
    ctx.results["runtime_s"] = time.perf_counter() - t0
    out = args.cache_dir / f"results_{'_'.join(stages)}.json"
    if "report" in stages:
        out = args.out_dir / "results.json"
    out.write_text(json.dumps(ctx.results, indent=1, default=float))
    LOG(f"wrote {out}")


if __name__ == "__main__":
    main()
