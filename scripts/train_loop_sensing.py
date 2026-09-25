"""Train the closed loop's learned room estimate (loop sensing study, 2026-09-25).

The loop (``web/src/loop/closedLoop.ts``) senses the room by coherent
delay-and-sum migration of the 8-speaker bar's ping residuals and keeps the
largest blob above 0.7 x max. This script trains a compact U-Net
(``imaging.models.CompactUNet``) that maps the same grid-aligned migration
images to an obstacle probability, on random loop scenes generated with the
loop's own TypeScript code (``web/scripts/loop_sensing_data.ts``):

    uv run python scripts/train_loop_sensing.py train  --data data/loop_sensing --out checkpoints/loop_unet
    uv run python scripts/train_loop_sensing.py export checkpoints/loop_unet/best.pt \\
        --out web/public/models/loop_unet --fixtures

``features`` mirrors ``loopFeatures`` in ``web/src/loop/learnedSensing.ts``
line by line; the web parity test checks the two against each other. The
probability threshold is chosen on the validation scenes only; the test
split is scored once, at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.imaging.models import CompactUNet  # noqa: E402

N_RAW = 5  # coherent, left, right, energy image, incoherent
EXCLUDE = 6.0
CLIP = 2.0
N_CH = 11


def demo_array(n: int = 100) -> NDArray[np.float64]:
    """The demo's 8-speaker bar (``scenarios.demoScenario``)."""
    s = n / 100
    return np.array([[round(86 * s), round(31 * s) + round(k * 4 * s)] for k in range(8)], float)


def crop_of(n: int) -> tuple[int, int]:
    """Model side m (a multiple of 8) and its offset in the n x n grid."""
    m = 8 * (n // 8)
    return m, (n - m) // 2


def near_array(array: NDArray, n: int, exclude: float = EXCLUDE) -> NDArray[np.bool_]:
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    d = np.min([np.hypot(ii - a[0], jj - a[1]) for a in array], axis=0)
    return d < exclude


def load_split(data: pathlib.Path, split: str, n: int = 100) -> dict:
    """Raw migration images, true masks, back-projection estimates and metadata."""
    meta = [
        json.loads(line)
        for line in (data / f"{split}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    k = len(meta)
    raw = np.fromfile(data / f"{split}.images.f32", dtype=np.float32, count=k * N_RAW * n * n)
    masks = np.fromfile(data / f"{split}.masks.u8", dtype=np.uint8, count=k * n * n)
    bp = np.fromfile(data / f"{split}.bp.u8", dtype=np.uint8, count=k * n * n)
    return {
        "raw": raw.reshape(k, N_RAW, n, n),
        "mask": masks.reshape(k, n, n).astype(bool),
        "bp": bp.reshape(k, n, n).astype(bool),
        "meta": meta,
    }


def log_max(raw: NDArray, near: NDArray) -> NDArray:
    """log10 of the coherent-energy maximum outside the near field, per scene."""
    e = np.where(near[None], -np.inf, raw[:, 3].astype(np.float64))
    return np.log10(np.maximum(e.reshape(len(raw), -1).max(1), 1e-24))


def features(raw: NDArray, array: NDArray, norm: dict, prior_logit: NDArray) -> NDArray[np.float32]:
    """Model inputs ``(K, 11, m, m)``; mirrors ``loopFeatures`` in learnedSensing.ts."""
    k, _, n, _ = raw.shape
    m, o = crop_of(n)
    near = near_array(array, n, norm["exclude"])
    r = raw.astype(np.float64)
    outside = ~near[None]
    flat = lambda a: np.where(outside, a, 0.0).reshape(k, -1)  # noqa: E731
    sC = np.maximum(np.abs(flat(r[:, 0])).max(1), 1e-12)[:, None, None]
    sE = np.maximum(flat(r[:, 3]).max(1), 1e-24)[:, None, None]
    sI = np.maximum(flat(r[:, 4]).max(1), 1e-24)[:, None, None]
    clip = norm["clip"]
    sl = np.s_[:, o : o + m, o : o + m]
    x = np.empty((k, N_CH, m, m), np.float32)
    x[:, 0] = np.clip(r[:, 0] / sC, -clip, clip)[sl]
    x[:, 1] = np.clip(r[:, 1] / sC, -clip, clip)[sl]
    x[:, 2] = np.clip(r[:, 2] / sC, -clip, clip)[sl]
    x[:, 3] = np.clip(np.sqrt(np.maximum(0, r[:, 3]) / sE), 0, clip)[sl]
    x[:, 4] = np.clip(np.sqrt(np.maximum(0, r[:, 4]) / sI), 0, clip)[sl]
    lm = (np.log10(sE[:, 0, 0]) - norm["logMaxMean"]) / norm["logMaxStd"]
    x[:, 5] = lm[:, None, None]
    gi, gj = np.meshgrid(np.arange(o, o + m), np.arange(o, o + m), indexing="ij")
    x[:, 6] = 2 * gi / (n - 1) - 1
    x[:, 7] = 2 * gj / (n - 1) - 1
    ac = array.mean(0)
    x[:, 8] = np.hypot(gi - ac[0], gj - ac[1]) / n
    x[:, 9] = near[o : o + m, o : o + m]
    x[:, 10] = np.asarray(prior_logit, np.float32).reshape(m, m) / 5
    return x


class LoopUNet(torch.nn.Module):
    """Prior logit plus a compact U-Net on the loop features (one output head)."""

    def __init__(self, prior_logit: NDArray, width: int = 12, c_in: int = N_CH):
        super().__init__()
        self.unet = CompactUNet(c_in, 1, width)
        self.prior: torch.Tensor
        self.register_buffer("prior", torch.as_tensor(prior_logit, dtype=torch.float32)[None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prior + self.unet(x)


def full_grid(prob: NDArray, n: int) -> NDArray:
    """Probabilities on the full n x n grid (zero outside the crop)."""
    m, o = crop_of(n)
    out = np.zeros((len(prob), n, n), prob.dtype)
    out[:, o : o + m, o : o + m] = prob
    return out


def iou_per_scene(pred: NDArray, truth: NDArray) -> NDArray:
    inter = (pred & truth).reshape(len(pred), -1).sum(1)
    union = (pred | truth).reshape(len(pred), -1).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def best_threshold(
    prob: NDArray, truth: NDArray, grid: NDArray | None = None
) -> tuple[float, float]:
    """Threshold on the probability maximising mean per-scene IoU."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    scores = [float(iou_per_scene(prob >= t, truth).mean()) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), scores[i]


@torch.no_grad()
def predict(model: torch.nn.Module, x: NDArray, batch: int = 32) -> NDArray:
    model.eval()
    out = [
        torch.sigmoid(model(torch.from_numpy(x[i : i + batch]))).numpy()[:, 0]
        for i in range(0, len(x), batch)
    ]
    return np.concatenate(out)


def prepare(data: pathlib.Path, n: int = 100) -> dict:
    array = demo_array(n)
    near = near_array(array, n)
    m, o = crop_of(n)
    tr = load_split(data, "train", n)
    pr = tr["mask"][:, o : o + m, o : o + m].mean(0)
    pr = np.clip(pr, 1e-3, 1 - 1e-3)
    prior_logit = np.log(pr / (1 - pr)).astype(np.float32)
    lm = log_max(tr["raw"], near)
    norm = {
        "logMaxMean": float(lm.mean()),
        "logMaxStd": float(lm.std() + 1e-6),
        "exclude": EXCLUDE,
        "clip": CLIP,
    }
    return {"array": array, "prior_logit": prior_logit, "prior": pr, "norm": norm, "train": tr}


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = pathlib.Path(args.data)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    P = prepare(data)
    tr = P["train"]
    va = load_split(data, "val")
    if args.limit:
        tr = {k: v[: args.limit] for k, v in tr.items()}
    m, o = crop_of(100)
    xt = features(tr["raw"], P["array"], P["norm"], P["prior_logit"])
    yt = tr["mask"][:, o : o + m, o : o + m].astype(np.float32)
    xv = features(va["raw"], P["array"], P["norm"], P["prior_logit"])
    print(f"train {len(xt)}  val {len(xv)}  features {xt.shape}", flush=True)
    model = LoopUNet(P["prior_logit"], args.width)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(xt) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1
    )
    best = -1.0
    hist = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(len(xt))
        tot = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch : (b + 1) * args.batch]
            xb = torch.from_numpy(xt[idx])
            yb = torch.from_numpy(yt[idx])[:, None]
            logit = model(xb)
            bce = F.binary_cross_entropy_with_logits(logit, yb)
            p = torch.sigmoid(logit)
            inter = (p * yb).sum((1, 2, 3))
            dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + yb.sum((1, 2, 3)) + 1)
            loss = bce + args.dice * dice.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        pv = full_grid(predict(model, xv), 100)
        tau, iou = best_threshold(pv, va["mask"])
        hist.append(
            {
                "epoch": ep,
                "loss": tot / len(xt),
                "val_iou": iou,
                "tau": tau,
                "sec": time.time() - t0,
            }
        )
        print(
            f"epoch {ep:3d} loss {tot / len(xt):.4f} val IoU {iou:.4f} @ {tau:.2f}  ({time.time() - t0:.0f} s)",
            flush=True,
        )
        if iou > best:
            best = iou
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "width": args.width,
                    "norm": P["norm"],
                    "prior_logit": P["prior_logit"],
                    "threshold": tau,
                    "val_iou": iou,
                    "epoch": ep,
                    "n_params": n_params,
                    "n_train": len(xt),
                    "args": vars(args),
                },
                out / "best.pt",
            )
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"best val IoU {best:.4f}; params {n_params}", flush=True)


def load_model(ckpt: pathlib.Path) -> tuple[LoopUNet, dict]:
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = LoopUNet(c["prior_logit"], c["width"])
    model.load_state_dict(c["state_dict"])
    model.eval()
    return model, c


def fold(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold eval-mode BatchNorm into the preceding (bias-free) convolution."""
    assert bn.running_var is not None and bn.running_mean is not None
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * s[:, None, None, None]
    b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    return w, (b0 - bn.running_mean) * s + bn.bias


def folded_tensors(model: LoopUNet) -> list[tuple[str, torch.Tensor]]:
    u = model.unet
    out: list[tuple[str, torch.Tensor]] = []
    for name in ("e1", "e2", "e3", "bott", "d3", "d2", "d1"):
        blk = getattr(u, name)
        for j, (ci, bi) in enumerate(((0, 1), (3, 4))):
            w, b = fold(blk[ci], blk[bi])
            out += [(f"{name}.{j}.weight", w), (f"{name}.{j}.bias", b)]
    assert u.out.bias is not None
    out += [("out.weight", u.out.weight), ("out.bias", u.out.bias)]
    return out


def folded_forward(
    tensors: dict[str, torch.Tensor], prior: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """The exported network (folded convolutions, as web/src/loop/learnedSensing.ts runs it)."""

    def block(h: torch.Tensor, name: str) -> torch.Tensor:
        for j in (0, 1):
            h = F.relu(
                F.conv2d(h, tensors[f"{name}.{j}.weight"], tensors[f"{name}.{j}.bias"], padding=1)
            )
        return h

    up = lambda h: F.interpolate(h, scale_factor=2.0, mode="nearest")  # noqa: E731
    e1 = block(x, "e1")
    e2 = block(F.max_pool2d(e1, 2), "e2")
    e3 = block(F.max_pool2d(e2, 2), "e3")
    b = block(F.max_pool2d(e3, 2), "bott")
    d3 = block(torch.cat([up(b), e3], 1), "d3")
    d2 = block(torch.cat([up(d3), e2], 1), "d2")
    d1 = block(torch.cat([up(d2), e1], 1), "d1")
    return prior + F.conv2d(d1, tensors["out.weight"], tensors["out.bias"])


def export(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    model, c = load_model(pathlib.Path(args.ckpt))
    dst = pathlib.Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tensors, table, off = [], [], 0
    with torch.no_grad():
        for name, t in folded_tensors(model):
            a = t.detach().numpy().astype("<f4").ravel()
            table.append({"name": name, "shape": list(t.shape), "offset": off})
            tensors.append(a)
            off += a.size
    np.concatenate(tensors).tofile(dst.with_suffix(".bin"))
    # Self-check: the folded network reproduces the BatchNorm model.
    with torch.no_grad():
        x = torch.randn(2, N_CH, *model.prior.shape[-2:])
        ref = model(x)
        got = folded_forward(dict(folded_tensors(model)), model.prior, x)
        err = float((ref - got).abs().max())
    assert err < 1e-3, f"folded network differs from the model by {err}"
    print(f"folded-weights check: max |logit difference| {err:.2e}")
    m, o = crop_of(100)
    manifest = {
        "format": "loop-unet-v1",
        "grid": 100,
        "crop": m,
        "offset": o,
        "width": c["width"],
        "channels": N_CH,
        "norm": c["norm"],
        "threshold": c["threshold"],
        "prior_logit": [round(float(v), 5) for v in np.asarray(c["prior_logit"]).ravel()],
        "tensors": table,
        "training": {
            "checkpoint": str(args.ckpt),
            "epoch": c["epoch"],
            "val_iou": c["val_iou"],
            "n_params": c["n_params"],
            "n_train": c["n_train"],
            "data": "web/scripts/loop_sensing_data.ts (random loop scenes, seeds 1e6 + i)",
        },
    }
    dst.with_suffix(".json").write_text(json.dumps(manifest))
    print(f"wrote {dst}.bin ({off * 4 / 1024:.0f} KiB) and .json")
    if args.fixtures:
        write_fixtures(model, c, pathlib.Path(args.data))


def write_fixtures(model: LoopUNet, c: dict, data: pathlib.Path) -> None:
    """A few test scenes (seeds only: the web test re-simulates them) with
    Python features (strided sample) and PyTorch logits for the parity test."""
    array = demo_array(100)
    rooms = []
    for split, idx in (("test", 0), ("test", 1), ("demo", 3)):
        d = load_split(data, split)
        raw = d["raw"][idx : idx + 1]
        x = features(raw, array, c["norm"], c["prior_logit"])
        with torch.no_grad():
            logit = model(torch.from_numpy(x)).numpy()[0, 0]
        rooms.append(
            {
                "split": split,
                "index": idx,
                "seed": d["meta"][idx]["seed"],
                "raw_sum": [float(v) for v in raw[0].reshape(N_RAW, -1).astype(np.float64).sum(1)],
                "raw_absmax": [float(v) for v in np.abs(raw[0]).reshape(N_RAW, -1).max(1)],
                "feature_sample": [round(float(v), 6) for v in x[0].ravel()[::97]],
                "logits": [round(float(v), 4) for v in logit.ravel()],
            }
        )
    p = ROOT / "web/tests/fixtures/loop_sensing_parity.json"
    p.write_text(json.dumps({"stride": 97, "rooms": rooms}))
    print(f"wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/loop_sensing")
    t.add_argument("--out", default="checkpoints/loop_unet")
    t.add_argument("--width", type=int, default=12)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--dice", type=float, default=1.0)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=2)
    e = sub.add_parser("export")
    e.add_argument("ckpt")
    e.add_argument("--out", default="web/public/models/loop_unet")
    e.add_argument("--data", default="data/loop_sensing")
    e.add_argument("--fixtures", action="store_true")
    e.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else export(a)


if __name__ == "__main__":
    main()
