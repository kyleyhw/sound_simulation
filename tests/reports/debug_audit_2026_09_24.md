# Debug audit, 2026-09-24 (plan Phase 3)

This audit reviewed every Python module and script, and the retired
backend server. Each fixed finding is pinned by a regression test that
failed on the pre-audit code. The fixes landed in:

- `8c97129`: engine;
- `648e4f4`: learning;
- `30075b8`: the backend, retired and replaced by the browser app.

The plan-level findings (units, the no-audio baseline, plan structure)
are in `docs/plan_audit.md`.

| Area | Major | Minor | Fixed | Open |
|---|---|---|---|---|
| Simulation engine (`simulation/`, 3.2) | 5 | 13 | 18 | GPU path unverified (no GPU) |
| Backend (`app/main.py`, 3.3) | 11 (3 critical) | 1 | retired | – |
| Learning code and scripts (3.4) | 5 | 5 | 10 | re-score after retraining (3.4.7) |

## 1. Simulation engine

Tests: `tests/simulation/test_engine_audit.py` (one test per finding
unless noted), plus the gate scripts wrapped as pytest tests.

### Major

| # | Finding | Failure | Fix |
|---|---|---|---|
| E1 | The single-driver fast-path cache went stale after `sim.drivers.append(...)` / `sim.drivers = []`. | A removed driver kept emitting, and an appended one was silently ignored. | `drivers` is a read-only tuple and its setter routes through `set_drivers`. `Driver` is frozen. The cache is refreshed on every mutation. |
| E2 | A driver position with fewer than `dims` coordinates was accepted. | `Driver((5,), …)` on a 2D grid broadcast into a whole row of sources. | `_validate_driver` requires `dims` integer coordinates and rounds floats. |
| E3 | The dataset generators produced wrong geometry: thin walls one cell thicker than drawn, an asymmetric rectangle margin, and mics allowed on the source cell. | The training rooms did not match the documented protocol. Some poses recorded the source directly. | Protocol **v3** (the new default) fixes all three. **v2** reproduces the published archives exactly: 0 mismatches over 400 seeds against the pre-fix code, and a content-digest match on the regenerated held-out archive (`data/MANIFEST.json`). |
| E4 | `AudioFileWaveform` never played its final sample and did not check aliasing. | Audio sources at a coarse `dt` injected aliased energy with no warning. | The last sample is played. `aliased_energy_fraction` plus a `Simulate` warning report aliasing, and `resampled_for(dt)` gives an anti-aliased `resample_poly`. |
| E5 | `random_free_position` was 2D-only. | 3D mic pairs could not be placed. | It is now N-D (test: `test_random_free_position_and_mics_work_in_3d`). |

### Minor

| # | Finding | Fix |
|---|---|---|
| E6 | `set_obstacle_mask` aliased the caller's array, so later edits changed the simulation. | Copies the array. |
| E7 | `timestep` was a NumPy float64, which promoted the 1D and N-D fields to float64. | Stored as a Python float, so the fields stay float32. |
| E8 | No validation of `wavespeed`, `gridstep`, `courant` or `timestep`. | The constructor raises on non-positive values. |
| E9 | The CFL warning missed σ = 1/√d, which is only marginally stable. | Strict inequality. |
| E10 | Stereo files were not downmixed in `from_samples`. | Downmixed. |
| E11 | The diverse obstacle generator crashed on tiny grids. | Returns an empty room. |
| E12 | `run_with_sensors` accepted invalid durations and sensors. | Validates its inputs. |
| E13 | `data_io` dropped driver positions and crashed on `None` parameters. | Both saved. |
| E14 | `visualize` forced the TkAgg backend and created `./plots` at import. | Neither happens at import. |
| E15 | `main.py` used out-of-band sources and had no headless mode. | In-band sources, `add_driver`, and `--save`. |
| E16 | Dead modules: `boundary.py`, `interactive_setup.py`, `reconstruct.py`, `generate.py`, `GenerateDriver`/`GenerateSensor`, `LocationGenerator`, `get_edge_values`. | Removed with their docs. |
| E17 | Docs said the 3D path was unfused and called p = 0 walls "rigid". | Corrected: 3D is fused, and p = 0 is pressure-release (Γ = −1) (3.2.8). |
| E18 | The 1D fallback was untested. | Tested against a hand-written leap-frog. |

Later physics work (Phase 5, `docs/physics.md`) also found and fixed three
**verification-method** bugs (not engine bugs) in `scripts/verify_physics.py`:

- a mode-frequency search confused near-degenerate modes;
- the rigid-box energy counted faces into rigid cells;
- the 3D Green's-function window included the wall echo.

It also found and fixed a real engine bug in new code: the CPML diverged
with walls painted inside the layer until no-flux faces were masked.

**Open (3.2.3):** `calculate_gpu.py` (fused kernels, the general kernel,
the CPML term) has not been run. It needs an NVIDIA GPU
(`tests/perf/check_simulate_gpu.py` skips without one).

## 2. Backend (`app/main.py`, retired)

Twelve findings, three critical. Line numbers refer to the last version
of the file (`git show 30075b8^:src/acoustic_system/app/main.py`).
Items 1–7 match the plan's audit record. Items 8–12 were found when the
retired file was re-read for this report.

| # | Finding | Severity | Lines | Failure |
|---|---|---|---|---|
| B1 | Client-chosen file paths through `AudioFileWaveform`, and `build_waveform` catches only `TypeError` | critical | 77–84, 186–187, 251–252 | `add_driver` with `{"type":"AudioFileWaveform","path":"/any/file.wav"}` reads a server file into the field that every client receives. A missing or non-WAV path raises an uncaught error that kills the handler. |
| B2 | `update_config` is not validated | critical | 184–190, 597–599 | `grid_shape` [50000, 50000] tries to allocate about 30 GB. A non-dict payload raises after the sim has stopped. A string `downsample` breaks every later emit. |
| B3 | One global session broadcast to every client, with no auth and `*` CORS | critical | 416, 432, 445, 535–541, 566–567 | Any client, or any web page, can start, stop, reset or reconfigure the shared simulation. |
| B4 | `add_driver` checks the grid outside the lock | major | 237–255 | If a concurrent `configure` changes the grid shape, the driver is validated against the old shape. The result is an uncaught `ValueError` or an out-of-bounds `IndexError` in `step()`. |
| B5 | Loop setup runs outside the `try`, so `is_running` gets stuck | major | 340–347, 376–378 | With `broadcast_hz` = 0 the task dies with `ZeroDivisionError`, and every later start returns early. |
| B6 | NaN/Infinity accepted, and `OverflowError` not caught | major | 81, 210–211, 240–241, 260–261, 669–671 | A NaN amplitude fills the field with NaN. `int(Infinity)` escapes every `(TypeError, ValueError)` handler. |
| B7 | Sensing errors never reach the client | major | 316–317, 333, 674–675 | A bad checkpoint raises outside any `try`, so no `sense_result` is sent and the UI waits forever. |
| B8 | Config is committed before the rebuild succeeds | major | 184–190 | `wavespeed` = −1 is stored while the old engine stays. Every later valid update fails until the value is fixed. |
| B9 | The waveform merge keeps the old type's kwargs | major | 80–84, 186–187 | Switching Ricker → Cosine carries `delay` over. The result is a `TypeError`, then a silent fallback to default parameters. |
| B10 | Private waveform fields, including an ndarray, sent on the wire | major | 499–502 | Any audio driver makes every `status` emit fail JSON serialisation. |
| B11 | `steps_per_frame` mixes wall-clock and simulation time, and steps run on the event loop | major | 378, 383–385 | Default units give 1 step per frame. SI units give about 5,500 synchronous steps per frame, which block every client. |
| B12 | An unstable `timestep` is accepted (warning only), and NaN frames are emitted | minor | 116–121, 159, 406, 437 | The browser's `JSON.parse` rejects the literal `NaN` tokens. |

The server was retired in plan 4.5.4. The web app runs the engine in the
browser (`web/src/engine/`), so there is no server state, lock, socket or
file path left to get wrong. These findings are recorded for the
history, not fixed.

## 3. Learning code and scripts

Tests: `tests/learning/test_split.py`, `test_calibration.py` (sibling
checkpoints), `test_metrics.py`.

### Major

| # | Finding | Failure | Fix |
|---|---|---|---|
| L1 | Train/val split over the flattened (room, pose) index. | Sibling poses of one room landed on both sides: P(leak) ≈ 0.999 at K = 4. Validation IoU, `best_iou` selection and calibration were scored on rooms the model trained on. | `room_level_split`, then expand to poses. Checkpoints record `val_rooms`. |
| L2 | Epoch means averaged per batch. | The last partial batch was over-weighted, which biased `best_iou` selection. | Sample-weighted means in `train.py` and `eval.py`. |
| L3 | One `calibration.json` per directory. | `best.pt` and `final.pt` silently used `best_iou.pt`'s temperature and threshold. | Per-checkpoint sidecar `<stem>.calibration.json`. A legacy file is honoured only for the checkpoint it names. |
| L4 | `eval_multipose.py` scored at τ = 0.5 by default. | The documented command could not reproduce the 0.100 @ τ = 0.12 headline. | It scores at the stored, validation-selected operating point by default. |
| L5 | The Bayes-fusion prior came from the evaluated (held-out) archive. | Held-out information leaked into the fused prediction. | The prior comes from training rooms (`train_prior` in the checkpoint). |

### Minor

| # | Finding | Fix |
|---|---|---|
| L6 | `StepLR` divided by zero for very short runs. | Step size ≥ 1. |
| L7 | `fit_calibration.py` accepted any dataset, and legacy flattened multi-pose checkpoints (with no leak-free validation set). | Refuses both. Reuses the checkpoint's stored validation rooms. |
| L8 | `eval_multipose.py --target-size` was broken. | Flag removed. |
| L9 | `eval.py` scored skip checkpoints per pose, but they were trained per room. | Scores per room. |
| L10 | `sensing.py` silently accepted acquisition settings it could not reproduce (`record_step`, `wavespeed`/`gridstep`, `n_mics`, randomised sources). | Raises. Honours v3 exclusion. Prefers `train_prior`. |

### Results impact (3.4.5 and 3.4.7)

The no-audio baseline (`scripts/eval_no_audio_baseline.py`) takes the
per-pixel training prior map, thresholded at a τ chosen on training
rooms. It scores **0.1015 ± 0.0031** IoU on the exact v2 held-out
archive. That equals the published audio models: skip_v2 0.100 at K = 4,
joint_v2 0.101 at K = 8. The Phase 2 IoU headline therefore shows no
measurable use of the audio.

**Re-scoring (3.4.7): in progress.** skip_v2 is being retrained with the
leak-free room split (L1) and the corrected selection (L2). It will be
re-scored with `scripts/eval_sensing.py` against the prior map on the
same rooms, using per-room paired differences. This section will be
updated with the numbers.

## 4. Repository hygiene (3.5)

- `.DS_Store` and `.idea/` are untracked. The README was rewritten.
- `data/MANIFEST.json` (`scripts/manifest.py`) records, for each dataset,
  its seed, protocol, regeneration command and a content digest that
  ignores HDF5 timestamps. It records a file hash and the training
  arguments for each checkpoint. Regenerating the held-out v2 archive
  from its recorded command reproduces the digest exactly.
- **Open:** off-repo hosting for the 1.6 GB training archive and the
  checkpoints needs a storage target the owner has not yet authorised
  (Hugging Face or Releases). Datasets regenerate exactly from the
  manifest in the meantime.
