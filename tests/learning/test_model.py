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
from acoustic_system.learning.model import (  # noqa: E402
    DualInputCNN,
    JointPoseCNN,
    PassiveCNN,
    SkipSensingCNN,
    build_model,
)


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

    # Joint-pose model (Task 2.1.4c): K-pose input, permutation
    # invariance, K-agnosticism, single-pose (3D input) degradation.
    model_joint = JointPoseCNN(n_mics=n_mics)
    k = 4
    sensor_k = torch.randn(batch, k, n_mics, t_rec)
    model_joint.eval()  # deterministic forward for the invariance checks
    with torch.no_grad():
        logits_j = model_joint(sensor_k, source)
        # Permutation invariance: shuffling poses must not change output.
        perm = torch.randperm(k)
        logits_perm = model_joint(sensor_k[:, perm], source)
        # K-agnostic: a K=2 subset must run through the same weights.
        logits_k2 = model_joint(sensor_k[:, :2], source)
        # Single-pose (B, M, T) treated as K=1.
        logits_k1 = model_joint(sensor, source)
    if tuple(logits_j.shape) != expected or tuple(logits_k2.shape) != expected:
        print(f"FAIL: joint expected {expected}, got {tuple(logits_j.shape)}")
        sys.exit(1)
    if not torch.allclose(logits_j, logits_perm, atol=1e-5):
        print("FAIL: joint model is not permutation-invariant over poses")
        sys.exit(1)
    if tuple(logits_k1.shape) != expected:
        print(f"FAIL: joint K=1 fallback expected {expected}, got {tuple(logits_k1.shape)}")
        sys.exit(1)
    loss_j = bce_dice_loss(model_joint(sensor_k, source), target)
    loss_j.backward()
    print(
        f"OK: joint forward (B={batch}, K={k}) -> {tuple(logits_j.shape)}, "
        f"pose-permutation invariant, K-agnostic (K=2, K=1 OK), "
        f"loss={loss_j.item():.4f} backward succeeded ({model_joint.num_params:,} params)"
    )
    if model_joint.num_params != DualInputCNN(n_mics=n_mics).num_params:
        print("FAIL: joint and dual param counts differ — comparison no longer controlled")
        sys.exit(1)
    print("OK: joint param count == dual param count (controlled comparison)")

    # Skip model (Task 2.3a+d): phase channels + multi-scale skips.
    # Longer T_rec (400) matching the v2 acquisition protocol.
    model_skip = SkipSensingCNN()
    t_rec_v2 = 400
    sensor_v2 = torch.randn(batch, k, 2, t_rec_v2)
    model_skip.eval()
    with torch.no_grad():
        logits_s = model_skip(sensor_v2, source)
        logits_s_perm = model_skip(sensor_v2[:, torch.randperm(k)], source)
        logits_s_k1 = model_skip(sensor_v2[:, 0], source)  # (B, 2, T) fallback
    if tuple(logits_s.shape) != expected or tuple(logits_s_k1.shape) != expected:
        print(f"FAIL: skip expected {expected}, got {tuple(logits_s.shape)}")
        sys.exit(1)
    if not torch.allclose(logits_s, logits_s_perm, atol=1e-5):
        print("FAIL: skip model is not permutation-invariant over poses")
        sys.exit(1)
    # Phase-channel sensitivity: swapping the two mics conjugates the
    # cross-spectrum (phi -> -phi), so the output MUST change — this is
    # the direct check that inter-channel phase reaches the network
    # (magnitude channels alone are mic-swap invariant up to reorder).
    with torch.no_grad():
        logits_swap = model_skip(sensor_v2.flip(dims=[2]), source)
    if torch.allclose(logits_s, logits_swap, atol=1e-6):
        print("FAIL: mic swap left output unchanged — phase channels not informative")
        sys.exit(1)
    loss_s = bce_dice_loss(model_skip(sensor_v2, source), target)
    loss_s.backward()
    n_skip = model_skip.num_params
    if n_skip > 2_000_000:
        print(f"FAIL: skip model {n_skip:,} params exceeds the CPU budget")
        sys.exit(1)
    try:
        model_skip(torch.randn(batch, k, 1, t_rec_v2), source)
        print("FAIL: skip model accepted mono input")
        sys.exit(1)
    except ValueError:
        pass
    print(
        f"OK: skip forward (B={batch}, K={k}, T={t_rec_v2}) -> {tuple(logits_s.shape)}, "
        f"pose-permutation invariant, K=1 fallback OK, mic-swap sensitive (phase used), "
        f"mono rejected, loss={loss_s.item():.4f} backward succeeded ({n_skip:,} params)"
    )

    # build_model dispatch
    if (
        not isinstance(build_model("dual"), DualInputCNN)
        or not isinstance(build_model("passive"), PassiveCNN)
        or not isinstance(build_model("joint"), JointPoseCNN)
        or not isinstance(build_model("skip"), SkipSensingCNN)
    ):
        print("FAIL: build_model dispatch broken")
        sys.exit(1)
    print("OK: build_model('dual'/'passive'/'joint'/'skip') dispatch")

    print()
    print("all CNN shape-correctness checks passed")


if __name__ == "__main__":
    main()
