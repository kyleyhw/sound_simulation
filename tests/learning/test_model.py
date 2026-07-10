"""Shape-correctness gates for the active-sensing CNN.

Run as a plain script (no pytest dependency). Each check exits non-zero
on failure so a CI run can chain them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_system.learning.losses import bce_dice_loss, iou_score  # noqa: E402
from acoustic_system.learning.model import DualInputCNN, PassiveCNN, build_model  # noqa: E402


def main() -> None:
    torch.manual_seed(0)

    # Realistic input shapes: 2-mic stereo @ 200 sim-time-steps,
    # synthetic chirp @ 200 Hz native rate over ~100 sim-time-units.
    batch, n_mics, t_rec = 4, 2, 200
    t_audio = 20000
    target_size = 64

    model = DualInputCNN(n_mics=n_mics)
    sensor = torch.randn(batch, n_mics, t_rec)
    source = torch.randn(batch, 1, t_audio)
    target = (torch.rand(batch, target_size, target_size) > 0.7).float()

    logits = model(sensor, source)
    expected = (batch, 1, target_size, target_size)
    if tuple(logits.shape) != expected:
        print(f"FAIL: expected logits {expected}, got {tuple(logits.shape)}")
        sys.exit(1)
    print(
        f"OK: model forward {tuple(sensor.shape)} + {tuple(source.shape)} -> {tuple(logits.shape)}"
    )

    # Loss + grad
    loss = bce_dice_loss(logits, target)
    if not loss.requires_grad:
        print("FAIL: loss is not differentiable")
        sys.exit(1)
    loss.backward()
    print(f"OK: bce+dice loss = {loss.item():.4f} backward succeeded")

    # IoU on logits = perfect prediction case
    perfect_logits = torch.where(target.unsqueeze(1) > 0.5, torch.tensor(10.0), torch.tensor(-10.0))
    perfect_iou = iou_score(perfect_logits, target).item()
    if perfect_iou < 0.999:
        print(f"FAIL: perfect-prediction IoU = {perfect_iou:.4f}, expected ~1.0")
        sys.exit(1)
    print(f"OK: perfect-prediction IoU = {perfect_iou:.6f}")

    # Param count sanity (warn if huge)
    n_params = model.num_params
    if n_params > 5_000_000:
        print(f"WARN: model has {n_params:,} params — larger than the CPU-laptop budget")
    print(f"OK: model has {n_params:,} params")

    # Single-mic fallback
    model_mono = DualInputCNN(n_mics=1)
    sensor_mono = torch.randn(batch, 1, t_rec)
    logits_mono = model_mono(sensor_mono, source)
    if tuple(logits_mono.shape) != expected:
        print(f"FAIL: mono path expected {expected}, got {tuple(logits_mono.shape)}")
        sys.exit(1)
    print(f"OK: mono (n_mics=1) forward returns {tuple(logits_mono.shape)}")

    # Passive model (Task 2.2): sensor-only path, with and without the
    # ignored source argument (the plumbing-compatibility contract).
    model_passive = PassiveCNN(n_mics=n_mics)
    logits_p1 = model_passive(sensor)
    logits_p2 = model_passive(sensor, source)
    if tuple(logits_p1.shape) != expected or tuple(logits_p2.shape) != expected:
        print(
            f"FAIL: passive expected {expected}, got "
            f"{tuple(logits_p1.shape)} / {tuple(logits_p2.shape)}"
        )
        sys.exit(1)
    if not torch.equal(logits_p1, logits_p2):
        print("FAIL: passive forward depends on the ignored source argument")
        sys.exit(1)
    loss_p = bce_dice_loss(logits_p1, target)
    loss_p.backward()
    print(
        f"OK: passive forward {tuple(sensor.shape)} -> {tuple(logits_p1.shape)}, "
        f"source ignored, loss={loss_p.item():.4f} backward succeeded "
        f"({model_passive.num_params:,} params)"
    )

    # build_model dispatch
    if not isinstance(build_model("dual"), DualInputCNN) or not isinstance(
        build_model("passive"), PassiveCNN
    ):
        print("FAIL: build_model dispatch broken")
        sys.exit(1)
    print("OK: build_model('dual'/'passive') dispatch")

    print()
    print("all CNN shape-correctness checks passed")


if __name__ == "__main__":
    main()
