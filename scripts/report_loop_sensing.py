"""Statistics and figures for the loop sensing study (2026-09-25).

Reads the paired closed-loop evaluation written by
``web/scripts/loop_sensing_eval.ts`` (one JSON line per scene) plus the
sensing-only scores on the held-out test split, and writes
``tests/reports/loop_sensing_2026_09_25_artifacts/`` (results.json and
figures):

    uv run python scripts/report_loop_sensing.py \\
        --eval data/loop_sensing/eval.jsonl --ckpt checkpoints/loop_unet/best.pt

All comparisons are paired per scene: mean +- SE of each estimator, and of
the per-scene difference, with z = mean(delta) / SE(delta).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_loop_sensing import (  # noqa: E402
    demo_array,
    features,
    full_grid,
    iou_per_scene,
    load_model,
    load_split,
    predict,
)

OUT = ROOT / "tests/reports/loop_sensing_2026_09_25_artifacts"
# Reference categorical palette (dataviz skill), slots 1-3, plus a neutral for the oracle.
COL = {"bp": "#2a78d6", "learned": "#eb6834", "empty": "#1baf7a", "oracle": "#8a8986"}
INK = "#52514e"


def mse(a: np.ndarray | list) -> dict:
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), "se": float(a.std(ddof=1) / np.sqrt(len(a))), "n": len(a)}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired difference a - b: mean, SE, z, Wilcoxon p, win rate."""
    d = np.asarray(a, float) - np.asarray(b, float)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {
        "delta": float(d.mean()),
        "se": se,
        "z": float(d.mean() / se) if se > 0 else float("inf"),
        "wilcoxon_p": p,
        "win_rate": float(np.mean(d > 0)),
    }


def sensing_scores(ckpt: pathlib.Path, data: pathlib.Path) -> dict:
    """IoU on the whole held-out test split (sensing only), with the no-audio prior baseline."""
    torch.set_num_threads(2)
    model, c = load_model(ckpt)
    te = load_split(data, "test")
    tr_masks = load_split(data, "train")["mask"]
    x = features(te["raw"], demo_array(), c["norm"], c["prior_logit"])
    prob = full_grid(predict(model, x), 100)
    learned = iou_per_scene(prob >= c["threshold"], te["mask"])
    bp = iou_per_scene(te["bp"], te["mask"])
    # No-audio baseline: the training prior map at the threshold that is best on validation.
    pmap = tr_masks.mean(0)
    va = load_split(data, "val")["mask"]
    taus = np.linspace(0.02, 0.6, 59)
    tau = float(
        taus[
            int(
                np.argmax(
                    [iou_per_scene(np.broadcast_to(pmap >= t, va.shape), va).mean() for t in taus]
                )
            )
        ]
    )
    prior = iou_per_scene(np.broadcast_to(pmap >= tau, te["mask"].shape), te["mask"])
    return {
        "n": len(learned),
        "threshold": c["threshold"],
        "val_iou": c["val_iou"],
        "iou": {"bp": mse(bp), "learned": mse(learned), "prior_only": mse(prior)},
        "paired": {
            "learned_minus_bp": paired(learned, bp),
            "learned_minus_prior": paired(learned, prior),
        },
        "prior_tau": tau,
        "_prob": prob,
        "_te": te,
    }


def robustness(ckpt: pathlib.Path, data: pathlib.Path) -> dict:
    """Sensing IoU on noisy copies of the test scenes and on out-of-family shapes."""
    model, c = load_model(ckpt)
    out = {}
    for name in ("test_snr30", "test_snr20", "test_snr10", "ood"):
        if not (data / f"{name}.jsonl").exists():
            continue
        d = load_split(data, name)
        x = features(d["raw"], demo_array(), c["norm"], c["prior_logit"])
        prob = full_grid(predict(model, x), 100)
        learned = iou_per_scene(prob >= c["threshold"], d["mask"])
        bp = iou_per_scene(d["bp"], d["mask"])
        out[name] = {
            "iou": {"bp": mse(bp), "learned": mse(learned)},
            "paired": paired(learned, bp),
        }
        if name == "ood":
            out[name]["_prob"] = prob >= c["threshold"]
            out[name]["_d"] = d
    return out


def figure_ood(rob: dict, path: pathlib.Path) -> None:
    """Out-of-family shapes: truth, back-projection and learned estimates."""
    if "ood" not in rob:
        return
    d = rob["ood"]["_d"]
    prob = rob["ood"]["_prob"]
    k = min(6, len(d["mask"]))
    fig, axes = plt.subplots(3, k, figsize=(2.1 * k, 6.4))
    for i in range(k):
        for row, (lab, est) in enumerate(
            (("truth", None), ("back-projection", d["bp"][i]), ("learned", prob[i]))
        ):
            ax = axes[row, i]
            img = np.full((100, 100, 3), 0.97)
            img[d["mask"][i]] = [0.55, 0.55, 0.55]
            if est is not None:
                colr = np.array(matplotlib.colors.to_rgb(COL["bp" if row == 1 else "learned"]))
                img[est & ~d["mask"][i]] = colr
                img[est & d["mask"][i]] = 0.5 * colr + 0.15
                inter = (est & d["mask"][i]).sum()
                uni = (est | d["mask"][i]).sum()
                ax.set_title(f"IoU {inter / max(uni, 1):.2f}", fontsize=8, color=INK)
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(lab, fontsize=9, color=INK)
    fig.suptitle("Out-of-family shapes (never trained on). Grey: truth.", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_examples(
    sens: dict, demo_rows: list[dict], rand_rows: list[dict], path: pathlib.Path
) -> None:
    """Truth, back-projection and learned estimates for the demo epochs and a few test scenes."""
    n = 100
    te = sens["_te"]
    prob = sens["_prob"]
    shown = []
    for r in demo_rows:
        shown.append((r["label"], r))
    by_seed = {m["seed"]: i for i, m in enumerate(te["meta"])}
    for r in rand_rows[:4]:
        shown.append((f"test seed {r['seed'] - 3_000_000}", r))
    fig, axes = plt.subplots(3, len(shown), figsize=(2.1 * len(shown), 6.6))
    for k, (title, r) in enumerate(shown):
        truth = np.zeros(n * n, bool)
        if r["kind"] == "demo":
            d = load_split(pathlib.Path("data/loop_sensing"), "demo")
            truth = d["mask"][-1 - r["seed"]]
            p = None
        else:
            i = by_seed[r["seed"]]
            truth = te["mask"][i]
            p = prob[i]
        for row, (name, cells) in enumerate(
            (
                ("truth", None),
                ("back-projection", r["estimates"]["bp"]),
                ("learned", r["estimates"]["learned"]),
            )
        ):
            ax = axes[row, k]
            img = np.full((n, n, 3), 0.97)
            if row == 2 and p is not None:
                img = img - 0.35 * p[..., None] * np.array([0.0, 0.45, 0.8])
            img[truth] = [0.55, 0.55, 0.55]
            if cells is not None:
                est = np.zeros(n * n, bool)
                est[np.asarray(cells, int)] = True
                est = est.reshape(n, n)
                colr = np.array(matplotlib.colors.to_rgb(COL["bp" if row == 1 else "learned"]))
                img[est & ~truth] = colr
                img[est & truth] = 0.5 * colr + 0.5 * np.array([0.3, 0.3, 0.3])
            ax.imshow(img, interpolation="nearest")
            for z, cz in ((r["bright"], "#eda100"), (r["dark"], "#4a3aa7")):
                ax.add_patch(
                    plt.Rectangle(
                        (z[1] - 0.5, z[0] - 0.5),
                        z[3] - z[1] + 1,
                        z[2] - z[0] + 1,
                        fill=False,
                        ec=cz,
                        lw=1,
                    )
                )
            a = demo_array()
            ax.plot(a[:, 1], a[:, 0], "s", ms=2, color="#e34948")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=8, color=INK)
            else:
                key = "bp" if row == 1 else "learned"
                ax.set_title(
                    f"IoU {r['iou'][key]:.2f} · {r['contrast'][key]:.1f} dB",
                    fontsize=7.5,
                    color=INK,
                )
            if k == 0:
                ax.set_ylabel(name, fontsize=9, color=INK)
    fig.suptitle(
        "Grey: true obstacles. Blue: back-projection estimate. Orange: learned estimate. Yellow/violet boxes: loud/quiet zone.",
        fontsize=8.5,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_contrast(rows: list[dict], path: pathlib.Path) -> None:
    """Mean steady-state contrast per design (random test scenes) and paired scatter."""
    keys = [
        ("empty", "empty room"),
        ("bp", "BP twin"),
        ("learned", "learned twin"),
        ("guarded_bp", "guarded, BP"),
        ("guarded_learned", "guarded, learned"),
        ("oracle", "oracle"),
    ]
    colors = [COL["empty"], COL["bp"], COL["learned"], COL["bp"], COL["learned"], COL["oracle"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    vals = [np.array([r["contrast"][k] for r in rows]) for k, _ in keys]
    m = [v.mean() for v in vals]
    se = [v.std(ddof=1) / np.sqrt(len(v)) for v in vals]
    x = np.arange(len(keys))
    bars = a1.bar(
        x, m, width=0.6, color=colors, yerr=se, capsize=3, error_kw={"ecolor": INK, "lw": 1}
    )
    for b_, k in zip(bars, keys, strict=True):
        if k[0].startswith("guarded"):
            b_.set_hatch("///")
            b_.set_edgecolor("white")
    for xi, mi in zip(x, m, strict=True):
        a1.text(xi, mi + 1.2, f"{mi:.1f}", ha="center", fontsize=8, color=INK)
    a1.set_xticks(x, [k[1] for k in keys], fontsize=8, rotation=15)
    a1.set_ylabel("contrast in the true room (dB)", color=INK)
    a1.set_title(f"Mean ± SE over {len(rows)} held-out scenes", fontsize=9, color=INK)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.grid(axis="y", color="#e6e5e1", lw=0.6)
    a1.set_axisbelow(True)
    bp = vals[1]
    le = vals[2]
    lo, hi = min(bp.min(), le.min()) - 2, max(bp.max(), le.max()) + 2
    a2.plot([lo, hi], [lo, hi], color="#b8b7b1", lw=1)
    a2.scatter(bp, le, s=16, color=COL["learned"], edgecolor="white", linewidth=0.6)
    a2.set_xlabel("ACC on back-projection twin (dB)", color=INK)
    a2.set_ylabel("ACC on learned twin (dB)", color=INK)
    a2.set_title(
        f"Paired per scene: learned above the line in {np.mean(le > bp):.0%}", fontsize=9, color=INK
    )
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_iou(rows: list[dict], path: pathlib.Path) -> None:
    """Twin contrast shortfall to the oracle against sensing IoU, for both estimators."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key, lab in (("bp", "back-projection"), ("learned", "learned")):
        iou = np.array([r["iou"][key] for r in rows])
        gap = np.array([r["contrast"]["oracle"] - r["contrast"][key] for r in rows])
        ax.scatter(iou, gap, s=16, color=COL[key], edgecolor="white", linewidth=0.6, label=lab)
    ax.axhline(0, color="#b8b7b1", lw=1)
    ax.set_xlabel("sensing IoU", color=INK)
    ax.set_ylabel("oracle − twin contrast (dB)", color=INK)
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--eval", default="data/loop_sensing/eval.jsonl")
    ap.add_argument("--ckpt", default="checkpoints/loop_unet/best.pt")
    ap.add_argument("--data", default="data/loop_sensing")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line) for line in pathlib.Path(a.eval).read_text().splitlines() if line.strip()
    ]
    demo = sorted((r for r in rows if r["kind"] == "demo"), key=lambda r: -r["seed"])
    rand = [r for r in rows if r["kind"] == "random"]
    C = lambda k: np.array([r["contrast"][k] for r in rand])  # noqa: E731
    iou_of = lambda k: np.array([r["iou"][k] for r in rand])  # noqa: E731
    designs = ["bp", "learned", "empty", "guarded_bp", "guarded_learned", "oracle"]
    res: dict = {
        "n_scenes": len(rand),
        "iou": {k: mse(iou_of(k)) for k in ("bp", "learned")},
        "contrast": {k: mse(C(k)) for k in designs},
        "paired": {
            "iou_learned_minus_bp": paired(iou_of("learned"), iou_of("bp")),
            "twin_learned_minus_bp": paired(C("learned"), C("bp")),
            "twin_learned_minus_empty": paired(C("learned"), C("empty")),
            "twin_bp_minus_empty": paired(C("bp"), C("empty")),
            "guarded_learned_minus_guarded_bp": paired(C("guarded_learned"), C("guarded_bp")),
            "oracle_minus_twin_bp": paired(C("oracle"), C("bp")),
            "oracle_minus_twin_learned": paired(C("oracle"), C("learned")),
            "oracle_minus_guarded_bp": paired(C("oracle"), C("guarded_bp")),
            "oracle_minus_guarded_learned": paired(C("oracle"), C("guarded_learned")),
            "oracle_minus_empty": paired(C("oracle"), C("empty")),
        },
        "guard_twin_rate": {
            k: float(np.mean([r["guard_choice"][k] == "twin" for r in rand]))
            for k in ("bp", "learned")
        },
        "cells": {k: mse([r["cells"][k] for r in rand]) for k in ("truth", "bp", "learned")},
        "latency_ms": {
            "learned_inference": mse([r["ms"]["learned_inference"] for r in rows]),
            "sense_total": mse([r["ms"]["sense_total"] for r in rows]),
        },
        "demo": [
            {
                "label": r["label"],
                "iou": r["iou"],
                "contrast": r["contrast"],
                "guard_choice": r["guard_choice"],
            }
            for r in demo
        ],
    }
    gap_bp = C("oracle") - C("bp")
    res["gap_closed"] = {
        "twin": float((C("learned") - C("bp")).mean() / gap_bp.mean()),
        "guarded": float(
            (C("guarded_learned") - C("guarded_bp")).mean() / (C("oracle") - C("guarded_bp")).mean()
        ),
    }
    sens = sensing_scores(pathlib.Path(a.ckpt), pathlib.Path(a.data))
    res["sensing_test_split"] = {k: v for k, v in sens.items() if not k.startswith("_")}
    ood_path = pathlib.Path(a.eval).with_name("eval_ood.jsonl")
    if ood_path.exists():
        ood = [json.loads(line) for line in ood_path.read_text().splitlines() if line.strip()]
        oc = lambda k: np.array([r["contrast"][k] for r in ood])  # noqa: E731
        oi = lambda k: np.array([r["iou"][k] for r in ood])  # noqa: E731
        ood_res: dict = {
            "n_scenes": len(ood),
            "iou": {k: mse(oi(k)) for k in ("bp", "learned")},
            "contrast": {k: mse(oc(k)) for k in designs},
            "paired": {
                "iou_learned_minus_bp": paired(oi("learned"), oi("bp")),
                "twin_learned_minus_bp": paired(oc("learned"), oc("bp")),
                "twin_learned_minus_empty": paired(oc("learned"), oc("empty")),
                "guarded_learned_minus_guarded_bp": paired(oc("guarded_learned"), oc("guarded_bp")),
                "oracle_minus_guarded_learned": paired(oc("oracle"), oc("guarded_learned")),
            },
            "guard_twin_rate": {
                k: float(np.mean([r["guard_choice"][k] == "twin" for r in ood]))
                for k in ("bp", "learned")
            },
        }
        res["loop_out_of_family"] = ood_res
    rob = robustness(pathlib.Path(a.ckpt), pathlib.Path(a.data))
    res["robustness"] = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in rob.items()
    }
    figure_ood(rob, OUT / "ood.png")
    hist = pathlib.Path(a.ckpt).with_name("history.json")
    if hist.exists():
        res["training_history"] = json.loads(hist.read_text())
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    figure_contrast(rand, OUT / "contrast.png")
    figure_iou(rand, OUT / "iou_vs_gap.png")
    figure_examples(sens, demo, rand, OUT / "examples.png")
    p = res["paired"]
    print(f"scenes {len(rand)}")
    for k in ("bp", "learned"):
        print(f"IoU {k}: {res['iou'][k]['mean']:.3f} ± {res['iou'][k]['se']:.3f}")
    for k in designs:
        print(f"contrast {k}: {res['contrast'][k]['mean']:.2f} ± {res['contrast'][k]['se']:.2f}")
    for k, v in p.items():
        print(
            f"{k}: {v['delta']:+.3f} ± {v['se']:.3f}  z {v['z']:.2f}  p {v['wilcoxon_p']:.2g}  win {v['win_rate']:.2f}"
        )
    print("gap closed", res["gap_closed"], "guard twin rate", res["guard_twin_rate"])
    print(
        "sensing test split",
        json.dumps(res["sensing_test_split"]["iou"]),
        json.dumps(res["sensing_test_split"]["paired"]),
    )
    for d in res["demo"]:
        print(d)
    if "loop_out_of_family" in res:
        o: dict = res["loop_out_of_family"]
        print(
            "OOD loop",
            o["n_scenes"],
            {k: f"{v['mean']:.2f}±{v['se']:.2f}" for k, v in o["contrast"].items()},
        )
        print("OOD iou", {k: f"{v['mean']:.3f}±{v['se']:.3f}" for k, v in o["iou"].items()})
        for k, v in o["paired"].items():
            print(
                f"  OOD {k}: {v['delta']:+.2f} ± {v['se']:.2f} z {v['z']:.2f} win {v['win_rate']:.2f}"
            )
        print("  OOD guard twin rate", o["guard_twin_rate"])
    for k, v in res["robustness"].items():
        print(
            k,
            {e: f"{x['mean']:.3f}±{x['se']:.3f}" for e, x in v["iou"].items()},
            f"z {v['paired']['z']:.1f}",
        )


if __name__ == "__main__":
    main()
