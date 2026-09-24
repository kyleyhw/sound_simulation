# Project Plan: Real-Time Acoustic Control System

_Last revised 2026-09-24. Phases are numbered in the order they will be
worked on. The reasoning behind this revision is in
[`docs/plan_audit.md`](docs/plan_audit.md). Details of completed work are
in `tests/reports/`._

## 1. Project Vision

Build a closed-loop acoustic system that senses a room and then shapes
the sound field in it. The goals are directed audio ("virtual
headphones") and directed noise cancellation.

The primary hardware target is a stock laptop or phone: 2 speakers,
2–4 mics, a webcam, and a browser. Speaker arrays are a secondary
research target. The simulator is a deliverable in its own right: a
fast, verified, browser-based acoustic sandbox.

## 2. Guiding Principles

*   **Intermediate deliverables:** Every phase ships something usable.
*   **Browser-first:** User-facing tools run in the browser.
*   **Consumer hardware:** The final system runs in real time on stock
    laptops and phones, and that is measured.
*   **Baselines always:** Report every result next to a no-information
    baseline, such as the average-room guess or no control at all.
*   **Verified physics:** Check the simulator against analytic
    solutions and use SI units at every interface.
*   **Autonomous execution:** After the user's go-ahead, phases run in
    order without approval for each phase (waived 2026-09-24). See
    §4 for the rules and stop points.

## 3. Phases

### Phase 1: Interactive Simulation UI — `[completed]`

*   `[completed]` 1.1: FastAPI + Socket.IO backend, Vite + React +
    TypeScript frontend.
*   `[completed]` 1.2: Step-at-a-time simulation loop with start, stop,
    reset, and config events.
*   `[completed]` 1.3: Canvas field view, obstacle brush, driver
    placement, and parameter panel.
*   `[completed]` 1.4: Live geometry and driver edits while the
    simulation runs.
*   `[completed]` 1.5: CUDA GPU backend (13× at 2048² in 2D, 17× at
    200³ in 3D).
*   `[completed]` 1.6: 3D mode with a Three.js volume renderer and
    binary streaming.

### Phase 2: Room Sensing, first attempt — `[completed]`, results under review

*   `[completed]` 2.1: Dataset generator: random rooms, a stereo mic
    pair, a chirp source, and multiple poses per room.
*   `[completed]` 2.2: Single-pose CNN (held-out IoU 0.037).
*   `[completed]` 2.3: Multi-pose Bayes fusion and the joint-pose CNN.
*   `[completed]` 2.4: Passive (unknown-source) CNN (held-out IoU
    0.030).
*   `[completed]` 2.5: Sensing v2: phase channels, wider band, varied
    room shapes, skip decoder, and calibration (held-out IoU 0.100).
*   `[completed]` 2.6: Room-mapping demo script and a *Sense room*
    panel in the web UI.
*   **Caveat:** A guess that ignores the audio scores 0.104 IoU on the
    same rooms, so none of these results beats that baseline yet. Task
    3.4 re-checks all of them.

### Phase 3: Full Codebase Debug Audit — `[in-progress]`
**Objective:** Find and fix bugs across the whole codebase before any
new work builds on it.
**Done when:** Every module is reviewed, found bugs are fixed or
logged, all tests run in CI, and the Phase 2 results are re-scored.

*   `[completed]` **Task 3.1: Test infrastructure**
    *   `[completed]` 3.1.1: Build a clean environment from `uv.lock`
        (dev + ml extras) and record which existing gates pass.
    *   `[completed]` 3.1.2: Add pytest and convert the script-style
        checks in `tests/` into pytest tests, keeping the scripts
        runnable.
    *   `[completed]` 3.1.3: Add `build` and `typecheck` scripts to
        `frontend/package.json`.
    *   `[completed]` 3.1.4: Set up GitHub Actions CI: ruff, ty, pytest,
        the CPU kernel checks, and the frontend typecheck and build.
*   `[pending]` **Task 3.2: Simulation engine (`simulation/`)**
    *   `[pending]` 3.2.1: `simulate.py`: step ordering, buffer
        rotation, `reset()`, the driver cache, obstacle masking, and
        the CFL logic.
    *   `[pending]` 3.2.2: `calculate.py`: kernel indexing, edge
        handling, and the thread-cap logic.
    *   `[pending]` 3.2.3: `calculate_gpu.py`: agreement with the CPU
        kernels in 2D and 3D, including obstacles.
    *   `[pending]` 3.2.4: The 1D fallback path: test it or remove it.
    *   `[pending]` 3.2.5: `waveforms.py` and `setup.py`: sampling
        limits and `AudioFileWaveform` edge cases (start, end,
        resampling).
    *   `[pending]` 3.2.6: `dataset.py`: obstacle generators, mic and
        source placement (never inside obstacles), and seed
        reproducibility.
    *   `[pending]` 3.2.7: Legacy modules (`data_io`, `visualize`,
        `reconstruct`, `interactive_setup`, `boundary`, `generate`,
        `utils`): fix, or delete if dead.
    *   `[completed]` 3.2.8: Correct the boundary wording everywhere: p = 0
        is a pressure-release wall, not a rigid one. This covers code
        comments, `docs/simulate.md`, and `CLAUDE.md`.
*   `[pending]` **Task 3.3: Backend (`app/main.py`)**
    *   `[pending]` 3.3.1: Check that every state-changing handler holds
        the lock, and look for start/stop/reset races.
    *   `[pending]` 3.3.2: Validate inputs on every socket event
        (types, bounds, malformed payloads).
    *   `[pending]` 3.3.3: Multiple clients share one simulation;
        decide whether that is intended and handle it.
    *   `[pending]` 3.3.4: Error handling: exceptions in the step loop,
        failed emits, and a client disconnecting mid-sense.
    *   `[pending]` 3.3.5: Add a test for every socket event (extend
        `tests/app/`).
*   `[pending]` **Task 3.4: Learning code and results (`learning/`,
    `scripts/`)**
    *   `[pending]` 3.4.1: Dataset loader: normalisation, pose
        flattening, and leakage between the train and validation
        splits.
    *   `[pending]` 3.4.2: `model.py` and `train.py`: shapes, pooling,
        seeding, and checkpoint selection (validation set only).
    *   `[pending]` 3.4.3: `eval.py`, `eval_multipose.py`, and
        `calibration.py`: metric correctness, prior computation, and
        threshold-selection leakage.
    *   `[pending]` 3.4.4: `sensing.py`: live inference must preprocess
        exactly like the offline evaluation.
    *   `[pending]` 3.4.5: Run `eval_no_audio_baseline.py` on the exact
        training and held-out archives.
    *   `[pending]` 3.4.6: Add metrics that don't depend on a threshold:
        information gained over the prior (bits), average precision,
        and boundary F-score.
    *   `[pending]` 3.4.7: Re-score every checkpoint and update the
        Phase 2 conclusions in the docs and reports.
*   `[pending]` **Task 3.5: Repo hygiene**
    *   `[completed]` 3.5.1: Stop tracking `.DS_Store` and `.idea/`.
    *   `[completed]` 3.5.2: Fix `README.md`: broken image paths, stale
        feature list, and the out-of-date layout.
    *   `[pending]` 3.5.3: Store datasets and checkpoints remotely,
        with checksums and seed manifests.
*   `[pending]` **Task 3.6: Report**
    *   `[pending]` 3.6.1: Write `tests/reports/debug_audit_<date>.md`
        listing the bugs found, the fixes, and the open issues.

### Phase 4: UI Overhaul — `[pending]`
**Objective:** Replace the current UI with a well-designed,
well-structured app. Today it is a single ~1.1k-line `App.tsx` with 40+
pieces of local state and an ad-hoc layout.
**Done when:** The new UI matches every current feature, adds the core
features below, passes end-to-end tests, and the old UI is deleted.

*   `[pending]` **Task 4.1: Design**
    *   `[pending]` 4.1.1: List the current features and their pain
        points.
    *   `[pending]` 4.1.2: Write a UX spec: layout (canvas, tool
        palette, inspector, bottom plot/timeline dock) and the main
        user flows.
    *   `[pending]` 4.1.3: Visual design: design tokens, typography,
        colour maps, and light and dark themes.
    *   `[pending]` 4.1.4: Choose the architecture: state store,
        component structure, a WebGL renderer, and a typed wire
        protocol shared with the backend.
*   `[pending]` **Task 4.2: Foundation**
    *   `[pending]` 4.2.1: New app shell with the layout and component
        structure.
    *   `[pending]` 4.2.2: Typed socket client plus protocol types
        generated from one schema.
    *   `[pending]` 4.2.3: A central state store to replace the scattered
        `useState` and `useRef` state.
    *   `[pending]` 4.2.4: A 2D WebGL field renderer with colour maps
        and a dB scale.
    *   `[pending]` 4.2.5: Port the 3D volume view into the new
        structure.
*   `[pending]` **Task 4.3: Feature parity**
    *   `[pending]` 4.3.1: Run controls: start, stop, reset, and
        single-step.
    *   `[pending]` 4.3.2: Config panel with input validation.
    *   `[pending]` 4.3.3: Obstacle tools: brush, eraser, line,
        rectangle, and ellipse.
    *   `[pending]` 4.3.4: Driver placement with a per-driver editor
        (waveform, amplitude, delay).
    *   `[pending]` 4.3.5: Sensing panel.
*   `[pending]` **Task 4.4: New core features**
    *   `[pending]` 4.4.1: Probes: virtual mics with a live waveform
        and spectrogram.
    *   `[pending]` 4.4.2: Listen at a probe through Web Audio.
    *   `[pending]` 4.4.3: Field views: dB scale, time-averaged loudness
        map, and intensity arrows.
    *   `[pending]` 4.4.4: Undo and redo.
    *   `[pending]` 4.4.5: Scene save and load (JSON) and shareable
        URLs.
    *   `[pending]` 4.4.6: Pause and scrub through recent frames.
    *   `[pending]` 4.4.7: A starter set of preset scenes.
    *   `[pending]` 4.4.8: Export a screenshot, GIF, or MP4.
    *   `[pending]` 4.4.9: Keyboard shortcuts and first-run hints.
*   `[pending]` **Task 4.5: Quality**
    *   `[pending]` 4.5.1: A Playwright test for every feature, run in
        CI.
    *   `[pending]` 4.5.2: Performance: 60 fps rendering at 512²,
        measured.
    *   `[pending]` 4.5.3: Responsive layout and accessibility pass.
    *   `[pending]` 4.5.4: Delete the old UI and rewrite
        `docs/web_ui.md`.

### Phase 5: Physics Correctness and Realism — `[pending]`
**Objective:** Make the simulator physically right for real rooms.
**Done when:** Analytic checks pass (grid convergence, room
resonances within 0.5 %, energy conservation), absorbing walls reflect
less than −40 dB, and T60 matches the Sabine formula.

*   `[pending]` **Task 5.1: Rigid walls**
    *   `[pending]` 5.1.1: Neumann (rigid) walls and obstacles in the
        2D and 3D CPU kernels.
    *   `[pending]` 5.1.2: The same in the GPU kernels.
    *   `[pending]` 5.1.3: A per-cell wall-type option, with the
        current p = 0 path unchanged so `reference.npz` still passes.
*   `[pending]` **Task 5.2: Physical units**
    *   `[pending]` 5.2.1: An SI scene layer (metres, seconds, Hz,
        c = 343 m/s) mapped to grid units.
    *   `[pending]` 5.2.2: Pick and document a realistic physical scale
        for future datasets.
*   `[pending]` **Task 5.3: Verification suite**
    *   `[pending]` 5.3.1: Room resonances vs the analytic
        eigenfrequencies, for both wall types.
    *   `[pending]` 5.3.2: Grid convergence (expect second order).
    *   `[pending]` 5.3.3: Energy conservation in lossless runs.
    *   `[pending]` 5.3.4: Numerical dispersion vs theory.
    *   `[pending]` 5.3.5: Point source vs the analytic 2D and 3D
        Green's functions.
    *   `[pending]` 5.3.6: Report with figures.
*   `[pending]` **Task 5.4: Absorbing boundaries**
    *   `[pending]` 5.4.1: Simple absorbing edges (Mur).
    *   `[pending]` 5.4.2: A perfectly matched layer (CPML).
    *   `[pending]` 5.4.3: Measure reflection against angle and
        frequency.
*   `[pending]` **Task 5.5: Wall materials**
    *   `[pending]` 5.5.1: Frequency-independent absorbing walls
        (absorption coefficient per cell).
    *   `[pending]` 5.5.2: Frequency-dependent absorption.
    *   `[pending]` 5.5.3: Check T60 against the Sabine and Eyring
        formulas.
*   `[pending]` **Task 5.6: Media, sources, and receivers**
    *   `[pending]` 5.6.1: Spatially varying sound speed c(x).
    *   `[pending]` 5.6.2: Soft, band-limited source injection.
    *   `[pending]` 5.6.3: Sub-cell source and mic positions.
    *   `[pending]` 5.6.4: Speaker and mic directivity patterns.
    *   `[pending]` 5.6.5: Mic noise model.
*   `[pending]` **Task 5.7: Engine capabilities**
    *   `[pending]` 5.7.1: A low-dispersion scheme (more usable
        bandwidth per cell).
    *   `[pending]` 5.7.2: A differentiable PyTorch copy of the kernels,
        checked against the numba ones.
    *   `[pending]` 5.7.3: Batched multi-room simulation for faster
        dataset generation.
*   `[pending]` **Task 5.8: UI**
    *   `[pending]` 5.8.1: Material painting, per-edge absorbing
        toggles, and sound-speed painting.

### Phase 6: Room Sensing, second attempt — `[pending]`
**Objective:** Show that sound actually reveals room geometry, then
push how much.
**Done when:** A method beats the no-audio baseline on held-out rooms
with statistical significance.

*   `[pending]` **Task 6.1: Better targets**
    *   `[pending]` 6.1.1: Label only obstacle surfaces the sound
        actually reaches, not the hidden interiors.
    *   `[pending]` 6.1.2: Room-outline (polygon) targets.
    *   `[pending]` 6.1.3: Distance-field targets.
*   `[pending]` **Task 6.2: Physics baselines (no ML)**
    *   `[pending]` 6.2.1: Recover the room impulse response by
        deconvolving the known chirp.
    *   `[pending]` 6.2.2: Locate walls from echo times (image-source
        method).
    *   `[pending]` 6.2.3: Synthetic-aperture imaging: combine the K
        laptop positions into one large virtual array.
    *   `[pending]` 6.2.4: Time-reversal imaging.
    *   `[pending]` 6.2.5: Full-waveform inversion using the
        differentiable engine (5.7.2).
*   `[pending]` **Task 6.3: Improved models**
    *   `[pending]` 6.3.1: Feed the synthetic-aperture images to the
        network as spatially aligned inputs.
    *   `[pending]` 6.3.2: Use the recovered impulse response as the
        input features.
    *   `[pending]` 6.3.3: A pose-aware set transformer for multi-pose
        fusion.
    *   `[pending]` 6.3.4: A generative model that outputs uncertainty
        maps.
*   `[pending]` **Task 6.4: What can a laptop hear?**
    *   `[pending]` 6.4.1: Theoretical accuracy limits (Cramér–Rao
        bound) against bandwidth, mic count, spacing, noise, and pose
        count.
    *   `[pending]` 6.4.2: A design chart of which hardware setups can
        resolve what.
*   `[pending]` **Task 6.5: Room acoustic parameters**
    *   `[pending]` 6.5.1: Estimate T60, direct-to-reverberant ratio,
        and absorption from recordings.
*   `[pending]` **Task 6.6: Real-world conditions**
    *   `[pending]` 6.6.1: Robustness when laptop positions are only
        roughly known.
    *   `[pending]` 6.6.2: Estimate laptop position and map jointly.
    *   `[pending]` 6.6.3: Suggest the next best place to move the
        laptop.
    *   `[pending]` 6.6.4: Passive sensing with phase features and
        multiple poses.
    *   `[pending]` 6.6.5: 3D rooms and 3–4 mic laptops.
*   `[pending]` **Task 6.7: UI**
    *   `[pending]` 6.7.1: Uncertainty display and a "move here next"
        hint in the sensing panel.

### Phase 7: Beamforming and Sound-Field Control — `[pending]`
**Objective:** Create localised sound and silence in simulated rooms
with realistic walls.
**Done when:** At least 10 dB contrast between the target zone and the
quiet zone with an array, and at least 15 dB crosstalk cancellation
with 2 laptop speakers, in an absorbing room.

*   `[completed]` 7.0: Multiple independently driven sources (the
    engine and UI already support this).
*   `[pending]` **Task 7.1: Tooling**
    *   `[pending]` 7.1.1: Compute speaker-to-point transfer functions
        from the simulator (batched, on GPU).
    *   `[pending]` 7.1.2: Metrics: zone contrast (dB), reproduction
        error, and array effort.
*   `[pending]` **Task 7.2: Classical controllers (`control/`)**
    *   `[pending]` 7.2.1: Delay-and-sum beam steering.
    *   `[pending]` 7.2.2: Pressure matching.
    *   `[pending]` 7.2.3: Acoustic contrast control.
    *   `[pending]` 7.2.4: Time-reversal focusing.
    *   `[pending]` 7.2.5: Broadband FIR filter design.
*   `[pending]` **Task 7.3: Laptop virtual headphones**
    *   `[pending]` 7.3.1: Crosstalk cancellation with 2 speakers.
    *   `[pending]` 7.3.2: Channel separation against head position and
        frequency.
    *   `[pending]` 7.3.3: Filters that adapt to a tracked head
        position.
*   `[pending]` **Task 7.4: Noise cancellation**
    *   `[pending]` 7.4.1: Adaptive (FxLMS) noise cancellation at a
        quiet zone.
    *   `[pending]` 7.4.2: Measure the size limits of the quiet zone.
*   `[pending]` **Task 7.5: Sensing requirements**
    *   `[pending]` 7.5.1: Design control from the *estimated* room and
        test it in the *true* room.
    *   `[pending]` 7.5.2: A curve of control quality against sensing
        error, which tells Phase 6 how accurate sensing must be.
*   `[pending]` **Task 7.6: Differentiable control**
    *   `[pending]` 7.6.1: Optimise speaker signals directly through the
        differentiable simulator.
*   `[pending]` **Task 7.7: UI**
    *   `[pending]` 7.7.1: Speaker-array tool.
    *   `[pending]` 7.7.2: Paint loud and quiet zones.
    *   `[pending]` 7.7.3: Live contrast readout and before/after
        loudness maps.

### Phase 8: Closed Loop in Simulation — `[pending]`
**Objective:** Connect sensing and control in one live loop.
**Done when:** Control quality holds while a listener or obstacle
moves, within a measured laptop latency budget.

*   `[pending]` 8.1: A "digital twin" bridge: sensing estimates the
    room, the twin simulates the transfer functions, and the controller
    is designed from them.
*   `[pending]` 8.2: Real-time loop in the backend: sense, update the
    twin, redesign, act.
*   `[pending]` 8.3: Measure the latency of each stage on a laptop CPU.
*   `[pending]` 8.4: Dynamic scenes: a moving listener, a moving
    obstacle, an opening door.
*   `[pending]` 8.5: Compare against a controller that knows the true
    room.
*   `[pending]` 8.6: UI dashboard: sensing map, twin vs truth,
    controller state, and zone metrics over time.

### Phase 9: Real Hardware — `[pending]`
**Objective:** Check the simulated results against real laptops and
real rooms.
**Done when:** Distance to a real wall is measured within a few cm,
real T60 is within 10 % of the simulated room, and the laptop
virtual-headphones demo works.

*   `[pending]` 9.1: Browser recording tool: play a chirp through the
    laptop speakers, record the mics, with auto gain and echo
    cancellation turned off.
*   `[pending]` 9.2: Recover real room impulse responses.
*   `[pending]` 9.3: Measure distance to a wall or desk from echoes.
*   `[pending]` 9.4: Measure real-room T60.
*   `[pending]` 9.5: Rebuild the same room in the simulator and compare.
*   `[pending]` 9.6: Calibrate speaker and mic responses and latency,
    per device.
*   `[pending]` 9.7: Capture a small, measured real-room test set.
*   `[pending]` 9.8: Test the sim-trained sensing models on it, and add
    randomisation to close the gap.
*   `[pending]` 9.9: Head tracking from the webcam in the browser.
*   `[pending]` 9.10: Live virtual-headphones demo on laptop speakers.
*   `[pending]` 9.11: Use a phone as a moving sensor (its IMU gives the
    pose).

### Phase 10: Showcase — `[pending]`
**Objective:** Make the work easy to try, understand, and cite.
**Done when:** A public zero-install site hosts the sandbox, gallery,
and explainers, and the write-ups are published.

*   `[pending]` 10.1: 2D simulator in WebGPU, running entirely in the
    browser.
*   `[pending]` 10.2: 3D simulator in WebGPU.
*   `[pending]` 10.3: Sensing models in the browser (ONNX).
*   `[pending]` 10.4: Deploy to GitHub Pages.
*   `[pending]` 10.5: Scene gallery: diffraction, whispering gallery,
    acoustic lens, room modes, time-reversal, beam steering, quiet
    zone, and others.
*   `[pending]` 10.6: Interactive explainer articles with live embedded
    simulations.
*   `[pending]` 10.7: Automated video and GIF rendering from scene
    files.
*   `[pending]` 10.8: README rewrite with an animated hero and a "try
    it" link.
*   `[pending]` 10.9: Docs website built from `docs/`.
*   `[pending]` 10.10: Public dataset and benchmark, with the baselines
    included.
*   `[pending]` 10.11: Write-up: "What can a laptop hear?" (the sensing
    results, negative results included).
*   `[pending]` 10.12: Write-ups on control and on the real-hardware
    results.

## 4. Execution Rules (agreed 2026-09-24)

*   **Start:** Nothing runs until the user gives an explicit go-ahead.
    After that, phases and tasks run in plan order.
*   **Commits:** One commit per subtask, pushed to `main`. Every commit
    passes ruff, ruff-format, ty, and the kernel checks.
*   **Tracking:** Mark each subtask's status here as work lands.
    `CURRENT_STATE.md` is updated at the end of every task.
*   **Phase reports:** Every phase ends with a report in
    `tests/reports/`.
*   **Checkpoints:** Datasets and checkpoints are regenerated or
    retrained on CPU from the documented seeds. There are no uploads
    from the user's machine.
*   **UI design (Phase 4):** Claude uses its own judgement. The user
    may ask for revisions later.
*   **Stop and report to the user** when:
    *   a result changes the plan (for example, Phase 6 finds no usable
        sensing signal), or
    *   a step needs the user's hardware: the GPU checks (3.2.3, 5.1.2,
        GPU parts of 5.7) and the real-room measurements (9.2–9.11), or
    *   a step publishes something under the user's name: remote
        artefact storage (3.5.3), GitHub Pages (10.4), the public
        benchmark (10.10), and the write-ups (10.11–10.12).
*   **After a stop:** Work continues on any tasks that don't depend on
    the blocked one. The GPU code itself is still written, and marked
    "untested on GPU".
