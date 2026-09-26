"""Evaluate the two-speaker study (tests/reports/two_speaker_2026_09_26.md).

Scores every trained scheme (``scripts/train_loop_sensing.py train --scheme``)
on the 100 held-out loop scenes (test seeds 3e6 + i), with the threshold
each model chose on validation scenes, and writes
``tests/reports/two_speaker_2026_09_26_artifacts/`` (``results.json`` and
figures):

    uv run python scripts/two_speaker_report.py

Per scheme: learned IoU and back-projection IoU (mean +- SE), the paired
difference against the sequential two-speaker device and against the bar
(mean +- SE, z, Wilcoxon p), and the measurement time. Test-only schemes
(beams, short codes, chirps, the naive code separation) are scored with the
model of the scheme whose images they imitate, and the noisy test splits
with the clean models (never trained on noise).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_loop_sensing as T  # noqa: E402

N = 100
LISTEN = 622
DX_M = 0.05  # physical scale assumed for "seconds of sound": 5 cm cells (a 5 m grid, f0 = 549 Hz)
DT_S = 0.5 * DX_M / 343.0

LABELS = {
    "bar8": "8-element bar (baseline)",
    "seq": "2 speakers, in turn",
    "wide_seq": "2 speakers 28 apart, in turn",
    "sum": "2 at once, same pulse",
    "band": "2 at once, disjoint bands",
    "code": "2 at once, noise codes (LS)",
    "seq_k2": "2 speakers, K = 2 placements",
    "seq_k4": "2 speakers, K = 4 placements",
    "seq_rigid": "2 speakers, rigid walls",
    "seq_rigid_img": "2 speakers, rigid walls + image sources",
}
TRAINED = list(LABELS)
# test-only schemes: (data dir, model)
EXTRA = {
    "beams": ("beams", "seq"),
    "code622": ("code622", "code"),
    "code622_seqmodel": ("code622", "seq"),
    "chirp": ("chirp", "code"),
    "code_mf": ("code_mf", "code"),
    "code_seqmodel": ("code", "seq"),
    "bar8_full2400": ("bar8", "bar8_full"),
}
NOISY = {
    "test_snr20": ["seq", "band", "code", "seq_k4", "bar8", "beams"],
    "test_snr10": ["seq", "band", "code", "bar8"],
}


def data_dir(scheme: str) -> pathlib.Path:
    return ROOT / "data/two_speaker" / scheme


def steps_of(scheme: str) -> int:
    if scheme == "bar8":
        return 8 * LISTEN
    return int(json.loads((data_dir(scheme) / "device.json").read_text())["steps"])


def load_ckpt(name: str, ckdir: pathlib.Path) -> tuple[T.LoopUNet, dict]:
    p = (
        ROOT / "checkpoints/loop_unet_w12/best.pt"
        if name == "bar8_full"
        else ckdir / name / "best.pt"
    )
    model, c = T.load_model(p)
    if "array" not in c:
        c["array"] = T.demo_array(N).tolist()
    return model, c


def evaluate(model: T.LoopUNet, c: dict, d: dict) -> np.ndarray:
    x = T.features(d["raw"], np.asarray(c["array"], float), c["norm"], c["prior_logit"])
    return T.full_grid(T.predict(model, x), N)


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    nz = d[np.abs(d) > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) > 5 else None
    return {
        "delta": float(d.mean()),
        "se": se,
        "z": float(d.mean() / se) if se > 0 else None,
        "wilcoxon_p": p,
        "first_better": float((d > 1e-12).mean()),
        "ties": float((np.abs(d) <= 1e-12).mean()),
    }


def ms(v: np.ndarray) -> dict:
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", default="checkpoints/two_speaker")
    ap.add_argument("--out", default="tests/reports/two_speaker_2026_09_26_artifacts")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ckdir = ROOT / a.ckpts
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    models = {}
    for name in [*TRAINED, "bar8_full"]:
        if name == "bar8_full" or (ckdir / name / "best.pt").exists():
            models[name] = load_ckpt(name, ckdir)
    test = {s: T.load_split(data_dir(s), "test") for s in TRAINED if s != "bar8" and s in models}
    test["bar8"] = T.load_split(ROOT / "data/loop_sensing", "test")
    truth = test["bar8"]["mask"]
    seeds = [m["seed"] for m in test["bar8"]["meta"]]
    for s, d in test.items():
        assert [m["seed"] for m in d["meta"]] == seeds, s

    res: dict = {"n_test": len(truth), "schemes": {}, "extra": {}, "noise": {}, "prior": {}}
    iou: dict[str, np.ndarray] = {}
    prob: dict[str, np.ndarray] = {}
    for s in TRAINED:
        if s not in models:
            continue
        model, c = models[s]
        p = evaluate(model, c, test[s])
        prob[s] = p
        iou[s] = T.iou_per_scene(p >= c["threshold"], truth)
        bp = T.iou_per_scene(test[s]["bp"], truth)
        st = steps_of(s)
        res["schemes"][s] = {
            "label": LABELS[s],
            "iou": ms(iou[s]),
            "iou_median": float(np.median(iou[s])),
            "bp_iou": ms(bp),
            "learned_minus_bp": paired(iou[s], bp),
            "threshold": c["threshold"],
            "val_iou": c["val_iou"],
            "epoch": c["epoch"],
            "n_train": c["n_train"],
            "elements": len(c["array"]),
            "steps": st,
            "seconds_at_5cm": st * DT_S,
        }
        print(
            f"{s:14s} IoU {iou[s].mean():.3f} ± {iou[s].std(ddof=1) / 10:.3f}  bp {bp.mean():.3f}  steps {st}"
        )
    for s in TRAINED:
        if s in iou:
            if "seq" in iou and s != "seq":
                res["schemes"][s]["vs_seq"] = paired(iou[s], iou["seq"])
            if "bar8" in iou and s != "bar8":
                res["schemes"][s]["vs_bar8"] = paired(iou[s], iou["bar8"])

    # No-audio prior: the training prior map, threshold chosen on validation.
    _, cb = models["bar8"]
    pr = 1 / (1 + np.exp(-np.asarray(cb["prior_logit"])))
    va = T.limit_split(T.load_split(ROOT / "data/loop_sensing", "val"), 200)
    pv = T.full_grid(np.repeat(pr[None], len(va["mask"]), 0), N)
    tau, _ = T.best_threshold(
        pv, va["mask"], np.linspace(0.02, 0.6, 59)
    )  # as report_loop_sensing.py
    pt = T.full_grid(np.repeat(pr[None], len(truth), 0), N)
    iou_prior = T.iou_per_scene(pt >= tau, truth)
    res["prior"] = {"iou": ms(iou_prior), "threshold": tau}
    for s in iou:
        res["schemes"][s]["vs_prior"] = paired(iou[s], iou_prior)
    print(f"prior IoU {iou_prior.mean():.3f}")

    # Test-only schemes.
    for name, (dd, mname) in EXTRA.items():
        if mname not in models:
            continue
        dpath = ROOT / "data/loop_sensing" if dd == "bar8" else data_dir(dd)
        if not (dpath / "test.jsonl").exists():
            continue
        d = T.load_split(dpath, "test")
        model, c = models[mname]
        v = T.iou_per_scene(evaluate(model, c, d) >= c["threshold"], truth)
        meta = d["meta"]
        sdr = [m.get("sdr_db") for m in meta if m.get("sdr_db") is not None]
        r = {
            "data": dd,
            "model": mname,
            "iou": ms(v),
            "bp_iou": ms(T.iou_per_scene(d["bp"], truth)),
            "steps": steps_of(dd) if dd != "bar8" else 8 * LISTEN,
            "sdr_db": float(np.mean(sdr)) if sdr else None,
        }
        if mname in iou:
            r["vs_model_scheme"] = paired(v, iou[mname])
        if "seq" in iou:
            r["vs_seq"] = paired(v, iou["seq"])
        res["extra"][name] = r
        print(f"extra {name:18s} IoU {v.mean():.3f} (model {mname})  sdr {r['sdr_db']}")

    # Noise (models trained without noise).
    for split, schemes in NOISY.items():
        for s in schemes:
            dpath = data_dir(s)
            mname = "seq" if s == "beams" else s
            if mname not in models or not (dpath / f"{split}.jsonl").exists():
                continue
            d = T.load_split(dpath, split)
            model, c = models[mname]
            v = T.iou_per_scene(evaluate(model, c, d) >= c["threshold"], truth)
            bp = T.iou_per_scene(d["bp"], truth)
            r = {
                "model": mname,
                "iou": ms(v),
                "bp_iou": ms(bp),
                "sigma_mean": float(np.mean([m["sigma"] for m in d["meta"]])),
            }
            if "seq" in iou:
                r["vs_clean_seq"] = paired(v, iou["seq"])
            res["noise"].setdefault(split, {})[s] = r
            print(f"{split} {s:8s} IoU {v.mean():.3f}  bp {bp.mean():.3f}")
        if "seq" in res["noise"].get(split, {}):
            # paired against the noisy sequential device
            base = None
            for s in res["noise"][split]:
                d = T.load_split(data_dir(s), split)
                model, c = models["seq" if s == "beams" else s]
                v = T.iou_per_scene(evaluate(model, c, d) >= c["threshold"], truth)
                if s == "seq":
                    base = v
                res["noise"][split][s]["_v"] = v
            for s in res["noise"][split]:
                v = res["noise"][split][s].pop("_v")
                if base is not None and s != "seq":
                    res["noise"][split][s]["vs_noisy_seq"] = paired(v, base)

    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {out / 'results.json'}")
    figures(res, prob, truth, models, out)


BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"


def figures(res: dict, prob: dict, truth: np.ndarray, models: dict, out: pathlib.Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # IoU per scheme (learned and back-projection), with the measurement time.
    order = [
        s
        for s in [
            "bar8",
            "seq",
            "wide_seq",
            "sum",
            "band",
            "code",
            "seq_k2",
            "seq_k4",
            "seq_rigid",
            "seq_rigid_img",
        ]
        if s in res["schemes"]
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(order))[::-1]
    for k, (key, col, lab) in enumerate(
        (("iou", BLUE, "learned U-Net"), ("bp_iou", ORANGE, "back-projection"))
    ):
        m = [res["schemes"][s][key]["mean"] for s in order]
        e = [res["schemes"][s][key]["se"] for s in order]
        ax.barh(
            y + (0.2 if k == 0 else -0.2),
            m,
            0.38,
            xerr=e,
            color=col,
            label=lab,
            error_kw={"lw": 1, "ecolor": "#52514e"},
        )
    pr = res["prior"]["iou"]["mean"]
    ax.axvline(pr, color=GRAY, lw=1, ls="--")
    ax.text(pr + 0.005, y[-1] - 0.6, "no-audio prior", color="#52514e", fontsize=8)
    ax.set_yticks(
        y, [f"{LABELS[s]}\n{res['schemes'][s]['steps']} steps" for s in order], fontsize=8
    )
    ax.set_xlabel("IoU on 100 held-out rooms (mean ± SE)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "iou_by_scheme.png", dpi=150)
    plt.close(fig)

    # IoU vs placements K.
    ks = [(1, "seq"), (2, "seq_k2"), (4, "seq_k4")]
    ks = [(k, s) for k, s in ks if s in res["schemes"]]
    if ks:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        kk = [k for k, _ in ks]
        for key, col, lab in (("iou", BLUE, "learned"), ("bp_iou", ORANGE, "back-projection")):
            m = np.array([res["schemes"][s][key]["mean"] for _, s in ks])
            e = np.array([res["schemes"][s][key]["se"] for _, s in ks])
            ax.errorbar(kk, m, e, color=col, lw=2, marker="o", ms=6, capsize=3, label=lab)
        if "bar8" in res["schemes"]:
            b = res["schemes"]["bar8"]["iou"]["mean"]
            ax.axhline(b, color=GRAY, ls="--", lw=1)
            ax.text(1.05, b + 0.015, "8-element bar (learned)", color="#52514e", fontsize=8)
        ax.set_xticks(kk, [f"K = {k}\n{res['schemes'][s]['steps']} steps" for k, s in ks])
        ax.set_ylabel("IoU (mean ± SE)")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "iou_vs_placements.png", dpi=150)
        plt.close(fig)

    # Example reconstructions.
    cols = [
        s for s in ["bar8", "seq", "sum", "band", "code", "seq_k4", "seq_rigid_img"] if s in prob
    ]
    idx = [0, 3, 7, 12]
    m_, o_ = T.crop_of(N)
    fig, axs = plt.subplots(
        len(idx), len(cols) + 1, figsize=(1.45 * (len(cols) + 1), 1.5 * len(idx))
    )
    for r, i in enumerate(idx):
        axs[r, 0].imshow(truth[i], cmap="Greys", vmin=0, vmax=1)
        axs[r, 0].set_title("truth" if r == 0 else "", fontsize=8)
        for c, s in enumerate(cols):
            _, ck = models[s]
            est = prob[s][i] >= ck["threshold"]
            ax = axs[r, c + 1]
            ax.imshow(prob[s][i], cmap="Blues", vmin=0, vmax=1)
            ax.contour(truth[i], levels=[0.5], colors=[ORANGE], linewidths=0.7)
            iou_i = T.iou_per_scene(est[None], truth[i : i + 1])[0]
            ax.set_title((s + "\n" if r == 0 else "") + f"IoU {iou_i:.2f}", fontsize=7)
            el = np.asarray(ck["array"])
            ax.plot(el[:, 1], el[:, 0], "s", color="#0b0b0b", ms=1.5)
        for ax in axs[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=150)
    plt.close(fig)
    print("wrote figures")


if __name__ == "__main__":
    main()
