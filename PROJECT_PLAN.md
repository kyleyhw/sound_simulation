# Project Plan: Real-Time Acoustic Control System

## 1. Project Vision

The ultimate goal is to develop a closed-loop acoustic control system capable of sensing an environment's acoustic properties and actively manipulating its sound field. This includes "virtual headphones" (directed audio) and directed noise cancellation.

## 2. Guiding Principles

*   **Intermediate Deliverables:** We will prioritize the creation of tangible, usable tools at each stage of the project. This allows for continuous validation and experimentation.
*   **Cross-Platform Compatibility:** All user-facing components, such as interactive UIs, will be browser-based to ensure they run on various operating systems.
*   **Consumer Hardware Constraint:** The final system must be computationally efficient enough to run in real-time on standard consumer hardware (laptops, phones).

## 3. Phased Development Plan

Work on each task or phase will only commence with explicit user permission.

---

### Phase 1: Interactive Browser-Based Simulation UI
**Objective:** Create the first major deliverable: a web-based user interface to interactively define, run, and visualize simulations in real-time.

*   `[completed]` **Task 1.1: Technology Stack Selection & Setup**
    *   `[completed]` 1.1.1: **Backend:** FastAPI + uvicorn server hosts the simulation manager (`src/acoustic_system/app/main.py`).
    *   `[completed]` 1.1.2: **Communication:** Socket.IO async server provides bidirectional WebSocket transport with HTTP-polling fallback.
    *   `[completed]` 1.1.3: **Frontend:** Vite + React + TypeScript dev server (`frontend/`).
*   `[completed]` **Task 1.2: Backend Development for Simulation Control**
    *   `[completed]` 1.2.1: Socket.IO event handlers: `start_simulation`, `stop_simulation`, `reset_simulation`, `update_config`, `set_obstacle`, `clear_obstacles`, `add_driver`, `remove_driver`, `clear_drivers`, `request_status`.
    *   `[completed]` 1.2.2: `Simulate.step()` runs from a single asyncio task owned by `SimulationManager`; mutation handlers acquire `_lock` to coordinate with start/stop/configure.
    *   `[completed]` 1.2.3: Pressure field broadcast at the configured `broadcast_hz` cadence; per-step downsampling keeps the wire cheap.
*   `[completed]` **Task 1.3: Frontend Development for Interactive UI**
    *   `[completed]` 1.3.1: Canvas renders the downsampled pressure field via an `ImageData` blit through an offscreen canvas; rAF render loop is decoupled from wire cadence.
    *   `[completed]` 1.3.2: Mouse-drag obstacle drawing (with brush radius), click-to-place / click-to-remove drivers, batched `set_obstacle` emit.
    *   `[completed]` 1.3.3: Start / Stop / Reset buttons + mode selector + parameter panel (waveform, grid_shape, courant, downsample, broadcast_hz) emit `update_config`.
*   `[completed]` **Task 1.4: Enable Live Interaction**
    *   `[completed]` 1.4.1: `Simulate` exposes `set_obstacle`, `clear_obstacles`, `add_driver`, `remove_driver`, `set_drivers`; the fast-path driver cache is refreshed on every mutation; obstacle scrub is gated by `_has_obstacles` so the no-obstacle path is bit-identical to the pre-feature numerics.
    *   `[completed]` 1.4.2: Socket events + `status` payload (`obstacles`, `drivers`) keep the UI synchronised with backend geometry. Smoke-tested with Playwright on 2026-05-13.
*   `[pending]` **Task 1.5: Implement GPU Acceleration**
    *   `[pending]` 1.5.1: Integrate CuPy into the core simulation logic, replacing NumPy operations in `calculate.py` with their GPU-accelerated counterparts.
    *   `[pending]` 1.5.2: Ensure efficient data transfer between the CPU and GPU.

---

### Phase 2: Advanced Sensing (Developing the "Ear")
**Objective:** Achieve high-fidelity, real-time room mapping within the simulation environment, using the UI from Phase 1 for testing.

*   `[in-progress]` **Task 2.1 (Active Sensing): Integrate & Model Complex Audio Sources**
    *   `[completed]` 2.1.1: `AudioFileWaveform` reads `.wav` sources via linear interpolation; `dataset.py` provides random rectangular-room generation, free-cell sampling, mic-pair placement, and a streaming sensor-recording runner; `scripts/generate_active_sensing.py` writes `(stereo sensor, source, obstacle_mask)` triplets to HDF5 with a synthetic-chirp fallback for users without an audio corpus. Updated for the laptop-only constraint (memory: end-goal-laptop-only) — sensor is a 2-mic stereo pair at random orientation, not a single mic.
    *   `[completed]` 2.1.2: `DualInputCNN` in `src/acoustic_system/learning/model.py` — two parallel spectrogram-encoder branches (sensor + source) feeding a transposed-conv mask decoder. ~232k params, designed for CPU laptop training. PyTorch Dataset, BCE+Dice loss, AdamW + cosine LR training loop, IoU eval, shape-correctness gate. Smoke training (200 samples / 50 epochs) reached val IoU 0.04 (well above random); next step is to push training further on a larger dataset.
    *   `[completed]` 2.1.3: Train and validate the active-sensing model. Two runs completed 2026-05-14 with a decisive **negative result**: baseline (10k samples, 100 epochs) reached held-out IoU 0.037 vs in-dist 0.134 — the model learned a marginal prior over obstacle locations, not the conditional map; the dropout + augmentation retry made every metric worse (held-out 0.030), confirming the failure is data under-determination, not model capacity. Single-pose 2-mic recordings do not carry enough information to constrain the mask. See `tests/reports/training_2026_05_14.md` and `training_2026_05_14_aug.md`, and `docs/learning.md`.
    *   `[pending]` 2.1.4: Multi-pose aggregation — K (driver, mic-pair) poses per room, exploiting laptop movement instead of adding channels (fits the laptop-only hardware constraint).
        - `[pending]` Extend `scripts/generate_active_sensing.py` to write K poses per room (shared obstacle mask, K sensor recordings).
        - `[pending]` Quick signal: Bayesian aggregation at inference — geometric-mean the single-pose model's probability maps over K poses and compare held-out IoU against the 0.037 baseline. Requires retraining the baseline checkpoint first (~2 h CPU; the 2026-05-14 checkpoints were lost to a temp-dir cleanup — exact command in `docs/learning.md`).
        - `[pending]` If aggregation lifts IoU: joint-pose model (shared encoder over K poses + pose-aggregation block). If not: revisit the chirp's spectral design before touching architecture.
*   `[pending]` **Task 2.2 (Passive Sensing): Research & Model Blind Deconvolution**
    *   `[pending]` 2.2.1: Research and implement a model architecture suitable for blind deconvolution (e.g., Autoencoder, RNN).
    *   `[pending]` 2.2.2: Train and validate the passive model, comparing its performance to the active model.

---

### Phase 3: Beamforming Simulation (Developing the "Mouth")
**Objective:** Create the algorithms and simulation environment for directed audio using a speaker array.

*   `[pending]` **Task 3.1: Implement Speaker Array and Control Algorithm**
    *   `[pending]` 3.1.1: Modify the simulation to support multiple, independently controlled drivers.
    *   `[pending]` 3.1.2: Develop a `control/beamformer.py` module with a foundational beamforming algorithm.
*   `[pending]` **Task 3.2: Validate Beamforming in Simulation**
    *   `[pending]` 3.2.1: Create test scenarios using the interactive UI to place a speaker array and target zones.
    *   `[pending]` 3.2.2: Verify the creation of localized sound and silence zones.

---

### Phase 4: Closed-Loop System Integration
**Objective:** Connect the "Ear" and "Mouth" in a real-time feedback loop within the simulation.

*   `[pending]` **Task 4.1: Develop Integrated Simulation Environment**
    *   `[pending]` 4.1.1: Create a master script or enhance the web backend to manage the full sense-model-act loop.
*   `[pending]` **Task 4.2: Connect Inference to Control**
    *   `[pending]` 4.2.1: Feed the microphone data into the trained inference model (from Phase 2).
    *   `[pending]` 4.2.2: Use the model's output (inferred RIR) as the input for the beamformer (from Phase 3).
*   `[pending]` **Task 4.3: Validate Real-Time Adaptation**
    *   `[pending]` 4.3.1: Use the UI to create dynamic scenarios (e.g., move a target or an obstacle).
    *   `[pending]` 4.3.2: Verify that the system adapts the directed audio beam in response to the changes.
