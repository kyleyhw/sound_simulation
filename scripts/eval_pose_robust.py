"""Pose-robust sensing: rigid pose error, refinement, autofocus, augmentation.

Follow-up to plan 6.6.1-6.6.2 (``tests/reports/imaging_models_2026_09_24.md``).
Report: ``tests/reports/pose_robust_2026_09_25.md``; maths: ``docs/imaging.md`` §10.

Protocol (nothing is tuned on held-out rooms):

* **Rooms.** As in ``eval_imaging_models.py``: training-archive rooms
  ``[0, 500)`` are validation rooms (thresholds, method settings), rooms
  ``[500, 4000)`` train the networks, and the 500 held-out rooms (first
  K = 4 poses) are scored once.
* **Error models.** ``indep``: every device displaced independently by
  ``round(N(0, sigma^2))`` per coordinate (``pose_images.perturb_poses``,
  the 6.6.1 model; the same random draws as that report). ``rigid``: one
  rigid transform per pose (translation ``N(0, sigma_t^2)`` per axis in
  cells, rotation ``N(0, sigma_theta^2)`` degrees about the pose centroid,
  rounded; ``pose_robust.perturb_poses_rigid``).
* **Corrections** (all from ``imaging/pose_robust.py`` and ``pose_refine.py``):
  ``perturbed`` (none), ``joint`` (the 6.6.2 per-device joint least squares,
  window ``ceil(2 sigma_dev)``), ``rigid`` (3-DOF per pose against the
  empty-box model, the loss chosen on validation rooms), ``autofocus``
  (cross-pose image coherence, no model of the map), ``autofocus+rigid``,
  and ``rigid+polish`` (rigid, then a per-device joint fit within one cell).
* **Networks.** ``unet``: the 6.3.1 U-Net trained on exact poses (cached by
  ``eval_imaging_models.py``). ``unet_aug``: the same architecture trained
  on exact images plus ``--n-aug`` copies of every training room imaged at
  randomly perturbed poses (a mixture of both error models and three
  magnitudes each); its threshold is chosen on validation rooms perturbed
  by the same mixture.
* **Stages** (``--stages``; cached under ``--cache-dir``/pose_robust):
  ``dev`` (settings on validation rooms), ``augdata`` (perturbed training
  images; ``--chunks`` splits it across processes), ``augtrain``,
  ``held`` and ``polish`` (held-out variants per condition;
  ``--conditions`` splits them), ``score`` (tables, figures,
  ``<out-dir>/results.json``).

Usage (one numba/torch thread per process, two processes; about 3.5 h of
wall time, of which the held-out search is 2 h)::

    export NUMBA_NUM_THREADS=1
    uv run python scripts/eval_pose_robust.py --stages dev
    uv run python scripts/eval_pose_robust.py --stages augdata --chunks 0,2,4,...,28
    uv run python scripts/eval_pose_robust.py --stages augdata --chunks 1,3,5,...,27
    uv run python scripts/eval_pose_robust.py --stages augtrain
    uv run python scripts/eval_pose_robust.py --stages held,polish --conditions rigid_1_2,rigid_2_5
    uv run python scripts/eval_pose_robust.py --stages held,polish --conditions indep_1,...
    uv run python scripts/eval_pose_robust.py --stages score \\
        --out-dir tests/reports/pose_robust_2026_09_25_artifacts
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile
import time
from typing import Any

import h5py
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from acoustic_system.imaging import models as M  # noqa: E402
from acoustic_system.imaging import pose_robust as PR  # noqa: E402
from acoustic_system.imaging.ir import TikhonovDeconvolver  # noqa: E402
from acoustic_system.imaging.pipeline import ArchiveRoom  # noqa: E402
from acoustic_system.imaging.pose_images import compute_pose_images, perturb_poses  # noqa: E402
from acoustic_system.imaging.pose_refine import refine_room  # noqa: E402
from acoustic_system.learning.metrics import mean_se  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_imaging_models", _REPO_ROOT / "scripts/eval_imaging_models.py"
)
assert _spec is not None and _spec.loader is not None
EM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EM)
EI = EM.EI

TRAIN_ARCHIVE = EM.TRAIN_ARCHIVE
HELD_ARCHIVE = EM.HELD_ARCHIVE
CHUNK = EM.CHUNK
K = 4
LOG = EM.LOG

#: Error conditions scored on held-out rooms: name -> (model, sigma or sigma_t, sigma_theta).
CONDITIONS: dict[str, tuple[str, float, float]] = {
    "rigid_0.5_1": ("rigid", 0.5, 1.0),
    "rigid_1_2": ("rigid", 1.0, 2.0),
    "rigid_2_5": ("rigid", 2.0, 5.0),
    "rigid_1_0": ("rigid", 1.0, 0.0),
    "rigid_0_3": ("rigid", 0.0, 3.0),
    "indep_0.5": ("indep", 0.5, 0.0),
    "indep_1": ("indep", 1.0, 0.0),
    "indep_2": ("indep", 2.0, 0.0),
}

#: Training / validation perturbation mixture for the augmented network.
AUG_MIXTURE: tuple[tuple[str, float, float], ...] = (
    ("indep", 0.5, 0.0),
    ("indep", 1.0, 0.0),
    ("indep", 2.0, 0.0),
    ("rigid", 0.5, 1.0),
    ("rigid", 1.0, 2.0),
    ("rigid", 2.0, 5.0),
)

#: Lever arm (RMS device distance from the pose centroid, cells) of the v2 poses,
#: used to turn a rotation into an equivalent per-device displacement.
LEVER_RMS = 16.3


def sigma_dev(model: str, s: float, s_theta: float) -> float:
    """Per-coordinate device error SD of an error model (cells)."""
    if model == "indep":
        return s
    rot = np.deg2rad(s_theta) * LEVER_RMS / np.sqrt(2.0)
    return float(np.hypot(s, rot))


def search_setup(model: str, s: float, s_theta: float) -> tuple[int, list[float], float, float]:
    """Rigid candidate grid for a declared error model.

    Translations within ``ceil(2 sigma_t)``; rotations ``{-2..2} sigma_theta``.
    For the independent model (no rotation parameter) the rigid search uses
    translations within ``ceil(2 sigma)`` and the rotation whose displacement
    at the RMS lever arm equals ``sigma``.
    """
    if model == "rigid":
        radius = max(1, int(np.ceil(2 * s)))
        return radius, PR.candidate_thetas(s_theta), max(s, 0.5), s_theta
    th = float(np.rad2deg(s / LEVER_RMS))
    return max(1, int(np.ceil(2 * s))), PR.candidate_thetas(th), s, th


def perturb(room: ArchiveRoom, model: str, s: float, s_theta: float, rng: np.random.Generator):
    if model == "indep":
        return perturb_poses(room, s, rng)[0]
    return PR.perturb_poses_rigid(room, s, s_theta, rng)[0]


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------


class Ctx:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cache = args.cache_dir
        self.dir = args.cache_dir / "pose_robust"
        self.dir.mkdir(parents=True, exist_ok=True)
        pz = np.load(self.cache / "priors.npz")
        self.prior = {"mask": pz["fill"], "illum": pz["illum"]}
        self.prior_logit = np.stack([EI.logit(pz["fill"]), EI.logit(pz["illum"])]).astype(
            np.float32
        )
        self._models: dict[str, torch.nn.Module] = {}

    def unet(self, name: str) -> torch.nn.Module:
        if name not in self._models:
            m = M.AlignedUNet(4, self.args.width, dropout=self.args.dropout)
            path = (self.dir if name != "unet_s0" else self.cache / "models") / f"{name}.pt"
            m.load_state_dict(torch.load(path))
            m.eval()
            self._models[name] = m
        return self._models[name]

    def logits(self, name: str, images: np.ndarray, batch: int = 50) -> np.ndarray:
        """U-Net logits ``(N, 2, H, W)`` for per-pose images ``(N, K, 4, H, W)``."""
        model = self.unet(name)
        out = []
        for s in range(0, len(images), batch):
            X = EM.aligned_inputs(images[s : s + batch], K)
            pl = np.broadcast_to(self.prior_logit, (len(X),) + self.prior_logit.shape).copy()
            with torch.no_grad():
                out.append(model(torch.from_numpy(X), torch.from_numpy(pl)).numpy())
        return np.concatenate(out)


def unet_prob(ctx: Ctx, name: str, images: np.ndarray) -> np.ndarray:
    return EM.sigmoid(ctx.logits(name, images)[:, 0])


# ---------------------------------------------------------------------------
# Stage: dev (method settings on validation rooms)
# ---------------------------------------------------------------------------


def focus_list(cs: list, channels: tuple[int, ...], smooth: float) -> list[np.ndarray]:
    """Autofocus features of every pose's candidates."""
    out = []
    for c in cs:
        assert c.focus is not None
        out.append(PR.focus_features(c.focus, channels, smooth))
    return out


def dev_selections(cs: list, st: float, sth: float) -> dict[str, list[int]]:
    """Every correction variant compared on validation rooms."""
    sel: dict[str, list[int]] = {"perturbed": [0] * len(cs)}
    for loss in ("l2", "huber", "quiet"):
        sel[f"rigid_{loss}"] = [PR.select_misfit(c, loss) for c in cs]
        sel[f"rigid_{loss}_map"] = [PR.select_misfit(c, loss, st, sth, 1.0) for c in cs]
    pens = [PR.prior_penalty(c.candidates.params, st, sth) for c in cs]
    for loss in ("l2", "quiet"):
        data = [PR.relative_cost(c, loss) for c in cs]
        for chn, ch in (("bp", (0,)), ("carve", (1,)), ("both", (0, 1))):
            feats = focus_list(cs, ch, 2.0)
            if loss == "l2":
                for metric in ("coherence", "entropy"):
                    for pw in (0.0, 0.05, 0.2):
                        sel[f"af_{chn}_{metric}_p{pw:g}"] = PR.autofocus_select(
                            feats, pens, None, metric, pw
                        )
            for dw in (0.3, 1.0, 3.0):
                sel[f"af+rigid_{loss}_{chn}_d{dw:g}"] = PR.autofocus_select(
                    feats, pens, data, "coherence", 0.05, dw
                )
    return sel


def stage_dev(ctx: Ctx) -> dict:
    path = ctx.dir / "dev.json"
    if path.exists():
        return json.loads(path.read_text())
    a = ctx.args
    val_mask = np.load(EM.train_chunk_path(ctx.cache, 0))["mask"]
    tau = EI.tune_tau(EM.sigmoid(np.load(ctx.cache / "logits" / "unet_s0_val.npy")[:, 0]), val_mask)
    out: dict[str, Any] = {"n_rooms": a.n_dev, "tau": tau, "conditions": {}}
    for cname in ("rigid_1_2", "rigid_2_5", "indep_1"):
        model, s, sth = CONDITIONS[cname]
        radius, thetas, st_decl, sth_decl = search_setup(model, s, sth)
        rng = np.random.default_rng(4242)
        scores: dict[str, list] = {}
        dec = None
        t0 = time.perf_counter()
        with h5py.File(TRAIN_ARCHIVE, "r") as hf:
            for r in range(a.n_dev):
                room = ArchiveRoom.from_group(hf[f"sample_{r:04d}"], K)
                if dec is None:
                    dec = TikhonovDeconvolver(room.drive, room.recordings.shape[-1], lam=1e-2)
                pr = perturb(room, model, s, sth, rng)
                cs = PR.analyse_room(pr, radius, thetas, dec)
                sel = dev_selections(cs, st_decl, sth_decl)
                done: dict[tuple, tuple[float, float]] = {}
                for name, idx in sel.items():
                    key = tuple(idx)
                    if key not in done:
                        rm = PR.apply_selection(pr, cs, idx)
                        im = compute_pose_images(rm, dec).images[None]
                        p = unet_prob(ctx, "unet_s0", im)[0]
                        err = float(PR.device_error(rm, room).mean())
                        done[key] = (EI.iou(p >= tau, room.mask), err)
                    scores.setdefault(name, []).append(done[key])
                if (r + 1) % 10 == 0:
                    LOG(f"  dev {cname}: {r + 1}/{a.n_dev} ({time.perf_counter() - t0:.0f} s)")
        out["conditions"][cname] = {
            name: {
                "iou": float(np.mean([v[0] for v in vals])),
                "iou_se": float(mean_se(np.array([v[0] for v in vals]))[1]),
                "pose_error": float(np.mean([v[1] for v in vals])),
            }
            for name, vals in scores.items()
        }
        best = sorted(out["conditions"][cname].items(), key=lambda kv: -kv[1]["iou"])[:8]
        for name, v in best:
            LOG(f"  dev {cname:10s} {name:32s} IoU {v['iou']:.4f} err {v['pose_error']:.2f}")
    path.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# Stage: augdata (perturbed-pose training images)
# ---------------------------------------------------------------------------


def aug_path(ctx: Ctx, copy: int, start: int) -> pathlib.Path:
    return ctx.dir / f"aug_c{copy}_{start:05d}.npz"


def aug_chunk(ctx: Ctx, copy: int, start: int) -> None:
    """Images of rooms ``[start, start + CHUNK)`` at poses perturbed by a random mixture member."""
    path = aug_path(ctx, copy, start)
    if path.exists():
        return
    imgs, which = [], []
    dec = None
    t0 = time.perf_counter()
    with h5py.File(TRAIN_ARCHIVE, "r") as hf:
        for r in range(start, start + CHUNK):
            rng = np.random.default_rng([ctx.args.seed, copy, r])
            room = ArchiveRoom.from_group(hf[f"sample_{r:04d}"], K)
            if dec is None:
                dec = TikhonovDeconvolver(room.drive, room.recordings.shape[-1], lam=1e-2)
            j = int(rng.integers(len(AUG_MIXTURE)))
            pr = perturb(room, *AUG_MIXTURE[j], rng)
            imgs.append(compute_pose_images(pr, dec).images.astype(np.float32))
            which.append(j)
            if (r - start + 1) % 100 == 0:
                LOG(
                    f"  aug copy {copy} rooms {start}+{r - start + 1}: {time.perf_counter() - t0:.0f} s"
                )
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, images=np.stack(imgs), which=np.array(which))
    tmp.rename(path)


def stage_augdata(ctx: Ctx) -> None:
    """Perturbed copies of training rooms [500, n_train) and one of validation rooms [0, 500)."""
    a = ctx.args
    jobs = [(c, s) for c in range(a.n_aug) for s in range(CHUNK, a.n_train_rooms, CHUNK)]
    jobs.append((99, 0))  # validation copy (threshold of the augmented network)
    chunks = range(len(jobs)) if a.chunks is None else [int(x) for x in a.chunks.split(",")]
    for i in chunks:
        if i < len(jobs):
            aug_chunk(ctx, *jobs[i])
            LOG(f"aug job {i} {jobs[i]} done")


# ---------------------------------------------------------------------------
# Stage: augtrain
# ---------------------------------------------------------------------------


def stage_augtrain(ctx: Ctx) -> None:
    a = ctx.args
    path = ctx.dir / "unet_aug.pt"
    if path.exists():
        return
    starts = range(CHUNK, a.n_train_rooms, CHUNK)
    tr = EM.load_npz(
        [EM.train_chunk_path(ctx.cache, s) for s in starts], ("images", "mask", "illum_pose")
    )
    Y1 = EM.targets(tr, K)
    Xs = [EM.aligned_inputs(tr["images"], K)]
    Ys = [Y1]
    del tr
    for c in range(a.n_aug):
        for s in starts:
            with np.load(aug_path(ctx, c, s)) as z:
                Xs.append(EM.aligned_inputs(z["images"], K))
        Ys.append(Y1)
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    del Xs
    LOG(f"augtrain: {len(X)} samples ({a.n_aug} perturbed copies + exact), {a.aug_epochs} epochs")
    torch.manual_seed(a.seed)
    model = M.AlignedUNet(4, a.width, dropout=a.dropout)

    class _C:  # the batching helper only needs prior_logit
        prior_logit = ctx.prior_logit

    t0 = time.perf_counter()
    hist = M.fit(
        model, EM.aligned_batches(_C, X, Y, a.batch, a.seed), a.aug_epochs, lr=a.lr, log=LOG
    )
    torch.save(model.state_dict(), path)
    meta = {"history": hist, "train_s": time.perf_counter() - t0, "n_samples": int(len(X))}
    (ctx.dir / "unet_aug.json").write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# Stage: held (held-out variants per condition)
# ---------------------------------------------------------------------------


def held_path(ctx: Ctx, cname: str) -> pathlib.Path:
    return ctx.dir / f"held_{cname}_n{ctx.args.n_held}.npz"


def condition_rng(cname: str) -> np.random.Generator:
    """The held-out perturbation stream of a condition (``indep``: the 6.6.1 seeds)."""
    model, s, sth = CONDITIONS[cname]
    if model == "indep":
        return np.random.default_rng(10_000 + int(round(10 * s)))
    return np.random.default_rng(20_000 + int(round(10 * s)) * 100 + int(round(10 * sth)))


def polish_path(ctx: Ctx, cname: str) -> pathlib.Path:
    return ctx.dir / f"held_{cname}_polish_n{ctx.args.n_held}.npz"


def stage_polish(ctx: Ctx, cname: str) -> None:
    """Rigid refinement plus a per-device polish (radius 1), same draws as :func:`stage_held`."""
    a = ctx.args
    path = polish_path(ctx, cname)
    if path.exists():
        return
    model, s, sth = CONDITIONS[cname]
    radius, thetas, _, _ = search_setup(model, s, sth)
    with np.load(held_path(ctx, cname)) as z:
        pert_dev = z["perturbed_devices"]
    rng = condition_rng(cname)
    out: dict[str, list] = {}
    dec = None
    t0 = time.perf_counter()
    with h5py.File(HELD_ARCHIVE, "r") as hf:
        for r in range(a.n_held):
            room = ArchiveRoom.from_group(hf[f"sample_{r:04d}"], K)
            if dec is None:
                dec = TikhonovDeconvolver(room.drive, room.recordings.shape[-1], lam=1e-2)
            pr = perturb(room, model, s, sth, rng)
            assert np.array_equal(PR._devices(pr), pert_dev[r]), "perturbation stream mismatch"
            rm, info = PR.refine_room_rigid_polish(
                pr, radius, thetas, a.settings["rigid_loss"], a.polish_radius
            )
            im = compute_pose_images(rm, dec, incident=[i.incident for i in info]).images
            out.setdefault("rigid+polish_images", []).append(im.astype(np.float32))
            out.setdefault("rigid+polish_devices", []).append(PR._devices(rm))
            if (r + 1) % 50 == 0:
                LOG(f"  polish {cname}: {r + 1}/{a.n_held} ({time.perf_counter() - t0:.0f} s)")
    arr = {k: np.stack(v) for k, v in out.items()}
    arr["wall"] = np.array(time.perf_counter() - t0)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, allow_pickle=False, **arr)
    tmp.rename(path)
    LOG(f"polish {cname} done ({time.perf_counter() - t0:.0f} s)")


def stage_held(ctx: Ctx, cname: str) -> None:
    """Perturbed poses and every correction for held-out rooms under one condition."""
    a = ctx.args
    path = held_path(ctx, cname)
    if path.exists():
        return
    model, s, sth = CONDITIONS[cname]
    radius, thetas, st_decl, sth_decl = search_setup(model, s, sth)
    r_joint = max(1, int(np.ceil(2 * sigma_dev(model, s, sth))))
    old = None
    if model == "indep":  # the 6.6.1 draws: same seed, same order
        p_old = ctx.cache / f"pose_s{s:g}_n{a.n_held}.npz"
        if p_old.exists():  # load once: indexing an NpzFile re-reads the whole array
            with np.load(p_old) as zo:
                old = {
                    k: zo[k]
                    for k in (
                        "perturbed_sources",
                        "perturbed_images",
                        "joint_images",
                        "joint_sources",
                        "joint_mics",
                    )
                }
    rng = condition_rng(cname)
    st = a.settings
    out: dict[str, list] = {}
    times: dict[str, float] = {}
    dec = None
    t0 = time.perf_counter()
    with h5py.File(HELD_ARCHIVE, "r") as hf:
        for r in range(a.n_held):
            room = ArchiveRoom.from_group(hf[f"sample_{r:04d}"], K)
            if dec is None:
                dec = TikhonovDeconvolver(room.drive, room.recordings.shape[-1], lam=1e-2)
            pr = perturb(room, model, s, sth, rng)
            rooms: dict[str, tuple[ArchiveRoom, Any]] = {"perturbed": (pr, None)}
            t1 = time.perf_counter()
            cs = PR.analyse_room(pr, radius, thetas, dec)
            times["analyse"] = times.get("analyse", 0.0) + time.perf_counter() - t1
            rooms["rigid"] = (
                PR.apply_selection(pr, cs, [PR.select_misfit(c, st["rigid_loss"]) for c in cs]),
                None,
            )
            feats = focus_list(cs, st["af_channels"], st["af_smooth"])
            pens = [PR.prior_penalty(c.candidates.params, st_decl, sth_decl) for c in cs]
            idx = PR.autofocus_select(feats, pens, None, st["af_metric"], st["af_prior"])
            rooms["autofocus"] = (PR.apply_selection(pr, cs, idx), None)
            data = [PR.relative_cost(c, st["afr_loss"]) for c in cs]
            feats2 = focus_list(cs, st["afr_channels"], st["af_smooth"])
            idx = PR.autofocus_select(
                feats2, pens, data, "coherence", st["afr_prior"], st["afr_data"]
            )
            rooms["autofocus+rigid"] = (PR.apply_selection(pr, cs, idx), None)
            reuse = old is not None and np.array_equal(old["perturbed_sources"][r], pr.sources)
            if not reuse:
                t1 = time.perf_counter()
                rj, info = refine_room(pr, r_joint, method="joint")
                times["joint"] = times.get("joint", 0.0) + time.perf_counter() - t1
                rooms["joint"] = (rj, [i.incident for i in info])
            for name, (rm, inc) in rooms.items():
                if reuse and name == "perturbed":
                    im = old["perturbed_images"][r].copy()
                else:
                    im = compute_pose_images(rm, dec, incident=inc).images.astype(np.float32)
                out.setdefault(f"{name}_images", []).append(im)
                out.setdefault(f"{name}_devices", []).append(PR._devices(rm))
            if reuse:
                assert old is not None
                out.setdefault("joint_images", []).append(old["joint_images"][r].copy())
                dj = np.concatenate([old["joint_sources"][r][:, None], old["joint_mics"][r]], 1)
                out.setdefault("joint_devices", []).append(dj)
            out.setdefault("true_devices", []).append(PR._devices(room))
            out.setdefault("n_candidates", []).append([len(c.candidates) for c in cs])
            if (r + 1) % 25 == 0:
                LOG(f"  held {cname}: {r + 1}/{a.n_held} ({time.perf_counter() - t0:.0f} s)")
    arr = {k: np.stack(v) for k, v in out.items()}
    arr["times"] = np.array([times.get("analyse", 0.0), times.get("joint", np.nan)])
    arr["wall"] = np.array(time.perf_counter() - t0)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, allow_pickle=False, **arr)
    tmp.rename(path)
    LOG(f"held {cname} done ({time.perf_counter() - t0:.0f} s)")


# ---------------------------------------------------------------------------
# Stage: score
# ---------------------------------------------------------------------------

VARIANTS = ("perturbed", "joint", "rigid", "autofocus", "autofocus+rigid", "rigid+polish")
#: Categorical slots 1-5 of the reference palette (validated for CVD separation).
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
MARKERS = ("o", "s", "^", "D", "v", "P")
NETS = ("unet_s0", "unet_aug")


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    m, se = mean_se(np.asarray(a) - np.asarray(b))
    return {"delta": m, "se": se, "z": m / se if se > 0 else float("nan")}


def stage_score(ctx: Ctx) -> dict:
    a = ctx.args
    n = a.n_held
    held = EM.load_npz([EM.held_path(ctx.cache, 500)], ("images", "mask"))
    truth = held["mask"][:n]
    exact_images = held["images"][:n, :K]
    prior = ctx.prior["mask"]
    val = np.load(EM.train_chunk_path(ctx.cache, 0))
    val_mask = val["mask"]
    taus = {
        "unet_s0": EI.tune_tau(unet_prob(ctx, "unet_s0", val["images"][:, :K]), val_mask),
    }
    with np.load(aug_path(ctx, 99, 0)) as z:
        val_pert = z["images"]
    taus["unet_aug"] = EI.tune_tau(unet_prob(ctx, "unet_aug", val_pert), val_mask)
    taus["unet_s0_mixtau"] = EI.tune_tau(unet_prob(ctx, "unet_s0", val_pert), val_mask)
    LOG(f"thresholds {taus}")
    prior_tau = EI.tune_tau(np.broadcast_to(prior, val_mask.shape).copy(), val_mask)
    prior_sc = EI.score(np.broadcast_to(prior, truth.shape).copy(), truth, prior_tau, prior)
    res: dict[str, Any] = {"n_rooms": n, "taus": taus, "prior": summarise(prior_sc)}
    exact = {}
    for net in NETS:
        exact[net] = EI.score(unet_prob(ctx, net, exact_images), truth, taus[net], prior)
    res["exact"] = {
        net: summarise(exact[net])
        | {
            "vs_prior": paired(exact[net]["iou"], prior_sc["iou"]),
            "vs_unet_exact": paired(exact[net]["iou"], exact["unet_s0"]["iou"]),
        }
        for net in NETS
    }
    per_room: dict[str, np.ndarray] = {f"exact/{net}": exact[net]["iou"] for net in NETS}
    res["conditions"] = {}
    for cname in CONDITIONS:
        path = held_path(ctx, cname)
        if not path.exists():
            LOG(f"  (skip {cname}: not computed)")
            continue
        z = load_condition(ctx, cname)
        true_dev = z["true_devices"]
        co: dict[str, Any] = {
            "model": CONDITIONS[cname],
            "wall_s": float(z["wall"]),
            "analyse_s": float(z["times"][0]),
            "joint_s": float(z["times"][1]),
            "mean_candidates": float(z["n_candidates"].mean()),
        }
        variants = [v for v in VARIANTS if f"{v}_images" in z]
        co["polish_s"] = float(z.get("polish_wall", np.nan))
        for v in variants:
            err = np.linalg.norm((z[f"{v}_devices"] - true_dev).astype(float), axis=-1)
            vo: dict[str, Any] = {
                "pose_error_mean": float(err.mean()),
                "pose_exact_frac": float((err == 0).mean()),
            }
            for net in NETS:
                sc = EI.score(unet_prob(ctx, net, z[f"{v}_images"]), truth, taus[net], prior)
                per_room[f"{cname}/{v}/{net}"] = sc["iou"]
                vo[net] = summarise(sc) | {
                    "vs_exact": paired(sc["iou"], exact["unet_s0"]["iou"]),
                    "vs_prior": paired(sc["iou"], prior_sc["iou"]),
                }
                if not (v == "perturbed" and net == "unet_s0"):
                    vo[net]["vs_baseline"] = paired(
                        sc["iou"], per_room[f"{cname}/perturbed/unet_s0"]
                    )
                LOG(
                    f"  {cname:12s} {v:16s} {net:9s} IoU {sc['iou'].mean():.4f} "
                    f"vs exact {vo[net]['vs_exact']['delta']:+.4f} (z {vo[net]['vs_exact']['z']:.1f}) "
                    f"err {vo['pose_error_mean']:.2f}"
                )
            if v == "perturbed":
                sc = EI.score(
                    unet_prob(ctx, "unet_s0", z[f"{v}_images"]),
                    truth,
                    taus["unet_s0_mixtau"],
                    prior,
                )
                vo["unet_s0_mixtau"] = summarise(sc) | {
                    "vs_exact": paired(sc["iou"], exact["unet_s0"]["iou"])
                }
            co[v] = vo
        pert = co["perturbed"]["unet_s0"]["iou"]["mean"]
        ex = res["exact"]["unet_s0"]["iou"]["mean"]
        for v in variants:
            for net in NETS:
                co[v][net]["recovered_frac"] = (co[v][net]["iou"]["mean"] - pert) / (ex - pert)
        res["conditions"][cname] = co
    pairs: dict[str, dict] = {}
    for cname in res["conditions"]:
        for a_, b_ in (
            ("rigid+polish", "joint"),
            ("rigid", "joint"),
            ("autofocus+rigid", "rigid"),
            ("rigid+polish", "rigid"),
        ):
            ka, kb = f"{cname}/{a_}/unet_s0", f"{cname}/{b_}/unet_s0"
            if ka in per_room and kb in per_room:
                pairs[f"{cname}: {a_} - {b_} (U-Net)"] = paired(per_room[ka], per_room[kb])
        for v in VARIANTS:
            ka, kb = f"{cname}/{v}/unet_aug", f"{cname}/{v}/unet_s0"
            if ka in per_room:
                pairs[f"{cname}: jitter U-Net - U-Net ({v})"] = paired(per_room[ka], per_room[kb])
    res["pairs"] = pairs
    res["meta_aug"] = json.loads((ctx.dir / "unet_aug.json").read_text())
    np.savez(ctx.dir / "per_room_iou.npz", allow_pickle=False, **per_room)
    return res


def markdown_tables(res: dict) -> str:
    """The report's main table (one row per condition and correction), as Markdown."""

    def f(e: dict) -> str:
        return f"{e['delta']:+.3f} ± {e['se']:.3f} ({e['z']:.1f})"

    lines = [
        "| condition | correction | device error (cells) | on exact cell | U-Net IoU | Δ vs exact (z) "
        "| Δ vs prior (z) | jitter U-Net IoU | Δ vs exact (z) | Δ vs no correction (z) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c, co in res["conditions"].items():
        for v in VARIANTS:
            if v not in co:
                continue
            u, a = co[v]["unet_s0"], co[v]["unet_aug"]
            base = f(u["vs_baseline"]) if "vs_baseline" in u else "—"
            lines.append(
                f"| {c} | {v} | {co[v]['pose_error_mean']:.2f} | {co[v]['pose_exact_frac']:.0%} "
                f"| {u['iou']['mean']:.3f} ± {u['iou']['se']:.3f} | {f(u['vs_exact'])} "
                f"| {f(u['vs_prior'])} | {a['iou']['mean']:.3f} | {f(a['vs_exact'])} "
                f"| {f(a['vs_baseline'])} (U-Net: {base}) |"
            )
    return "\n".join(lines)


def load_condition(ctx: Ctx, cname: str) -> dict[str, np.ndarray]:
    """The held-out variants of one condition, with the polish variant when computed."""
    with np.load(held_path(ctx, cname)) as z:
        out = {k: z[k] for k in z.files}
    pp = polish_path(ctx, cname)
    if pp.exists():
        with np.load(pp) as z:
            out.update({k: z[k] for k in z.files if k != "wall"})
            out["polish_wall"] = z["wall"]
    return out


def summarise(sc: dict) -> dict:
    return {k: dict(zip(("mean", "se"), mean_se(v))) for k, v in sc.items()}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def cond_label(cname: str) -> str:
    model, s, sth = CONDITIONS[cname]
    if model == "indep":
        return f"indep. σ={s:g}"
    return f"rigid σt={s:g}, σθ={sth:g}°"


def figures(ctx: Ctx, res: dict, out_dir: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [c for c in CONDITIONS if c in res["conditions"]]
    labels = {
        "perturbed": "no correction",
        "joint": "joint LS (per device)",
        "rigid": "rigid LS (3 DOF)",
        "autofocus": "autofocus",
        "autofocus+rigid": "autofocus + rigid",
        "rigid+polish": "rigid LS + per-device polish",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    x = np.arange(len(conds))
    for ai, net in enumerate(NETS):
        ax = axes[ai]
        present = [v for v in VARIANTS if all(v in res["conditions"][c] for c in conds)]
        w = 0.8 / len(present)
        for i, v in enumerate(present):
            m = [res["conditions"][c][v][net]["iou"]["mean"] for c in conds]
            e = [res["conditions"][c][v][net]["iou"]["se"] for c in conds]
            ax.bar(
                x + (i - (len(present) - 1) / 2) * w,
                m,
                w,
                yerr=e,
                label=labels[v],
                capsize=2,
                color=COLORS[i],
                edgecolor="white",
                linewidth=1,
                error_kw={"elinewidth": 0.8, "ecolor": "#555555"},
            )
        ax.axhline(
            res["exact"]["unet_s0"]["iou"]["mean"],
            color="k",
            ls="--",
            lw=1,
            label="U-Net, exact poses",
        )
        ax.axhline(res["prior"]["iou"]["mean"], color="grey", ls=":", lw=1, label="no-audio prior")
        ax.set_xticks(x, [cond_label(c) for c in conds], rotation=30, ha="right")
        ax.set_title(
            "U-Net trained on exact poses" if net == "unet_s0" else "U-Net trained with pose jitter"
        )
        ax.set_ylabel("held-out IoU (K = 4)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_dir / "pose_robust_iou.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, v in enumerate(VARIANTS):
        if not all(v in res["conditions"][c] for c in conds):
            continue
        xs = [res["conditions"][c][v]["pose_error_mean"] for c in conds]
        ys = [res["conditions"][c][v]["unet_aug"]["iou"]["mean"] for c in conds]
        ax.scatter(xs, ys, label=labels[v], color=COLORS[i], marker=MARKERS[i], s=45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("mean device error after correction (cells; 1 cell = 2.5 cm)")
    ax.set_ylabel("IoU, jitter-trained U-Net")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pose_error_vs_iou.png", dpi=130)
    plt.close(fig)


def fig_examples(
    ctx: Ctx, out_dir: pathlib.Path, cname: str = "rigid_1_2", rooms=(0, 1, 2)
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = load_condition(ctx, cname)
    held = EM.load_npz([EM.held_path(ctx.cache, 500)], ("images", "mask"))
    cols = [
        ("truth", None, ""),
        ("exact, U-Net", "exact", "unet_s0"),
        ("no correction, U-Net", "perturbed", "unet_s0"),
        ("no correction, jitter U-Net", "perturbed", "unet_aug"),
        ("rigid LS + polish, U-Net", "rigid+polish", "unet_s0"),
    ]
    fig, axes = plt.subplots(len(rooms), len(cols), figsize=(2.3 * len(cols), 2.3 * len(rooms)))
    for i, r in enumerate(rooms):
        for j, (title, v, net) in enumerate(cols):
            ax = axes[i, j]
            if v is None:
                ax.imshow(held["mask"][r], cmap="gray_r")
            else:
                im = held["images"][r : r + 1, :K] if v == "exact" else z[f"{v}_images"][r : r + 1]
                ax.imshow(unet_prob(ctx, net, im)[0], cmap="magma", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / f"examples_{cname}.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

#: Correction settings, the best mean U-Net IoU over the three ``dev`` conditions
#: (validation rooms 0-39; ``dev.json``). The grid is in :func:`dev_selections`.
DEFAULT_SETTINGS = {
    "rigid_loss": "l2",
    "af_channels": (1,),
    "af_smooth": 2.0,
    "af_metric": "coherence",
    "af_prior": 0.2,
    "afr_loss": "l2",
    "afr_channels": (0, 1),
    "afr_prior": 0.05,
    "afr_data": 3.0,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path(tempfile.gettempdir()) / "acoustic_imaging_models_cache",
    )
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument(
        "--stages", default="score", help="comma-separated: dev,augdata,augtrain,held,score"
    )
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--chunks", default=None, help="augdata job indices for this process")
    ap.add_argument("--n-train-rooms", type=int, default=4000)
    ap.add_argument("--n-held", type=int, default=500)
    ap.add_argument("--n-dev", type=int, default=40)
    ap.add_argument("--n-aug", type=int, default=4)
    ap.add_argument("--aug-epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--polish-radius", type=int, default=1)
    ap.add_argument("--settings", default=None, help="JSON overriding the correction settings")
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    settings: dict[str, Any] = dict(DEFAULT_SETTINGS)
    if args.settings:
        settings.update(json.loads(args.settings))
    settings["af_channels"] = tuple(settings["af_channels"])
    settings["afr_channels"] = tuple(settings["afr_channels"])
    args.settings = settings
    ctx = Ctx(args)
    stages = args.stages.split(",")
    t0 = time.perf_counter()
    if "dev" in stages:
        stage_dev(ctx)
    if "augdata" in stages:
        stage_augdata(ctx)
    if "augtrain" in stages:
        stage_augtrain(ctx)
    if "held" in stages:
        for c in args.conditions.split(","):
            stage_held(ctx, c)
    if "polish" in stages:
        for c in args.conditions.split(","):
            stage_polish(ctx, c)
    if "score" in stages:
        res = stage_score(ctx)
        res["settings"] = {k: list(v) if isinstance(v, tuple) else v for k, v in settings.items()}
        dev = ctx.dir / "dev.json"
        if dev.exists():
            res["dev"] = json.loads(dev.read_text())
        res["runtime_score_s"] = time.perf_counter() - t0
        out = args.out_dir or ctx.dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(res, indent=1))
        figures(ctx, res, out)
        fig_examples(ctx, out)
        LOG(markdown_tables(res))
        LOG(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
