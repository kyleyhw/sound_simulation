"""Run eval.py on both the training dataset and a held-out set,
collect metrics, and emit a one-line summary suitable for pasting into
a training report.

Usage::

    python scripts/eval_and_report.py \\
        --checkpoint checkpoints/long_baseline/best_iou.pt \\
        --indist data/training_data/active_sensing_train_10k.hdf5 \\
        --heldout data/training_data/active_sensing_heldout_500.hdf5 \\
        --output-dir tests/reports/training_2026_05_14_artifacts

Keep datasets and checkpoints under ``data/training_data/`` and
``checkpoints/`` (gitignored), never ``/tmp`` — temp cleanups have
already destroyed one set of trained checkpoints (2026-05-14 runs).

Writes preds_indist.png and preds_heldout.png to the output dir, and
copies the checkpoint's loss.png alongside. Prints the summary table
to stdout (mean IoU per split per checkpoint).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--indist", required=True, help="Training dataset HDF5 (in-distribution eval).")
    p.add_argument("--heldout", required=True, help="Separately-generated held-out HDF5.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-samples", type=int, default=8)
    return p.parse_args()


def run_eval(checkpoint: str, dataset: str, output: str, n_samples: int) -> float:
    """Invoke eval.py as a subprocess; parse the printed mean IoU."""
    cmd = [
        sys.executable,
        "-m",
        "acoustic_system.learning.eval",
        "--checkpoint",
        checkpoint,
        "--dataset",
        dataset,
        "--n-samples",
        str(n_samples),
        "--output",
        output,
    ]
    print(f"[eval-and-report] running: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"eval.py returned {completed.returncode}")
    # Parse the "mean IoU @ threshold=0.5: <FLOAT>" line.
    iou: float | None = None
    for line in completed.stdout.splitlines():
        if "mean IoU" in line:
            try:
                iou = float(line.split(":")[1].split()[0])
            except (IndexError, ValueError):
                continue
    if iou is None:
        raise RuntimeError("could not parse mean IoU from eval output")
    return iou


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indist_iou = run_eval(
        args.checkpoint,
        args.indist,
        str(out_dir / "preds_indist.png"),
        args.n_samples,
    )
    heldout_iou = run_eval(
        args.checkpoint,
        args.heldout,
        str(out_dir / "preds_heldout.png"),
        args.n_samples,
    )

    # Copy the checkpoint's loss curve into the report directory if present.
    ckpt_dir = Path(args.checkpoint).parent
    loss_png = ckpt_dir / "loss.png"
    if loss_png.exists():
        shutil.copy(loss_png, out_dir / "loss.png")

    print()
    print("=== summary ===========================================")
    print(f"checkpoint:      {args.checkpoint}")
    print(f"in-dist IoU:     {indist_iou:.4f}  ({args.indist})")
    print(f"held-out IoU:    {heldout_iou:.4f}  ({args.heldout})")
    print(f"plots:           {out_dir}/")
    print("========================================================")


if __name__ == "__main__":
    main()
