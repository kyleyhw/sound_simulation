# Project Status as of 2026-05-14

## 1. Resolution

Phase 2 active-sensing pipeline is in place end-to-end on the
laptop-only hardware target. Stereo dataset generator, dual-input
spectrogram CNN with mask decoder, training loop with AdamW + cosine
LR, and per-sample IoU eval all wired together. A first long training
run (10k samples, 100 epochs, AdamW + cosine schedule, weight decay
1e-4) is running against the in-distribution training set; a separate
held-out validation set is available for clean eval after.

The 3D volumetric simulation + Three.js volume renderer landed earlier
this session and remains usable end-to-end (200×200×64 grids stream as
~256 KB / frame uint8 over the socket, ray-marched in WebGL2 at >100
FPS). Both 2D and 3D paths share the same `Simulate` engine; the 2D
hot path uses the original numba-fused 5-point kernel, 3D the new
fused 7-point kernel.

## 2. What changed this session

### Engine + visualisation
- `fused_leapfrog_step_3d` 7-point stencil (numba @njit, prange) — 3D
  sims now run at ~0.04 ms/step at 32³, ~7.0 ms/step at 200³, ~50×
  faster than the previous scipy fallback.
- Backend binary uint8 streaming for 3D pressure fields via socket.io
  binary attachments. 2D path unchanged (JSON nested-list grid).
- Three.js volumetric renderer with the standard front-to-back alpha-
  compositing volume integral, opacity-corrected for step-count
  invariance. UI sliders for γ (alpha exponent), αscale, obstacle
  alpha, ray-step count.
- React 19 dev-mode profiler workaround: typed-array props are passed
  via refs + a frame-counter rather than directly, so the profiler's
  internal `performance.measure()` can't choke on deep-cloning them.

### Phase 2 / active sensing
- `AudioFileWaveform` + `AudioFileWaveform.from_samples` for synthetic-
  chirp-as-source, registered in the existing waveform_registry.
- `dataset.py` helpers: `generate_random_obstacles`,
  `random_free_position`, `pick_mic_positions` (random-orientation
  stereo pair), `run_with_sensors` (streaming sensor recording),
  `synthetic_chirp` (linear-sweep fallback source).
- `scripts/generate_active_sensing.py`: writes `(stereo recording,
  source audio, obstacle mask)` triplets to HDF5. Channel-last
  `sensor` dataset of shape `(T_rec, n_mics)`, plus per-sample attrs
  for reproduction.
- `src/acoustic_system/learning/`:
  - `losses.py`: BCE + soft-Dice + IoU score.
  - `dataset.py`: PyTorch Dataset wrapping the HDF5 archive.
  - `model.py`: `DualInputCNN` (~232k params, designed for CPU).
  - `train.py`: AdamW + cosine LR, train + val IoU tracking, two
    checkpoints (best by val_loss, best by val_iou).
  - `eval.py`: whole-dataset IoU + per-sample prediction grid.
- `tests/learning/test_model.py`: forward shape, loss + backward,
  perfect-prediction IoU sanity, mono fallback path.

### Hygiene + infra
- `frontend/node_modules/` untracked (was tracked from before the
  hygiene pass).
- `pyproject.toml` `ml` extra: `torch>=2.4`, `torchaudio>=2.4`.
  `uv sync --extra dev --extra ml` bootstraps; CPU wheels are the
  default.

### Project memory (in `~/.claude/projects/.../memory/`)
- `end-goal-laptop-only.md`: stock laptop hardware constraint —
  built-in stereo mic + speaker, optionally a webcam mic. 2-3
  channels max. CNN architectures and dataset generators must
  default to 2-channel input.

## 3. Open follow-ups

- **2.1.3 (in progress)**: long training run completing today. After
  it lands: write up the IoU achieved on the held-out set and decide
  whether to push for more epochs / more data / regularisation tweaks
  before moving to 2.1.4.
- **2.1.4 (next)**: temporal aggregation across multiple chirps as
  the laptop is moved (active sensing pose). Single-shot 2-mic
  inference is severely under-determined; the lever for higher
  resolution is multi-pose, not more sensors.
- The 3D web UI does not yet support 3D obstacle drawing — only the
  generator script can author 3D scenes. Worth adding once the ML
  side stabilises.
