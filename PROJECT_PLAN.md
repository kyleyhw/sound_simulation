# Project Plan: Real-Time Acoustic Control System

_Revised 2026-09-24 after a full audit ([`docs/plan_audit.md`](docs/plan_audit.md)).
Audit finding ids (A1, B3, …) are cited in brackets. The history of
completed tasks is compressed here; the details live in `tests/reports/`._

## 1. Project Vision

Build a closed-loop acoustic system. It senses a space's geometry and
acoustics, then shapes the sound field there. The applications are
"virtual headphones" (directed audio) and directed noise cancellation.

The work targets two hardware tiers [B4]:

- **Tier L — laptop / phone (primary).** Two speakers, 2–4 mics, a
  webcam, and a browser. Realistic goals:
  - sense room geometry and reverberation
  - *transaural* (crosstalk-cancelled) virtual headphones that track
    the listener's head
  - low-frequency local quiet
- **Tier A — speaker arrays (research / stretch).** Personal sound
  zones, steerable beams, and zone-based active noise cancellation.

The simulator is itself a first-class deliverable: a fast, verified,
browser-native acoustic sandbox. It is also the project's showcase.

## 2. Guiding Principles

*   **Intermediate deliverables:** Each stage ships a usable tool.
*   **Cross-platform:** User-facing components are browser-based.
*   **Consumer hardware:** The final system runs in real time on stock
    laptops and phones. Latency and compute are *measured*, not
    assumed.
*   **Beat the null model [A1]:** Report every sensing or control
    result next to a no-information baseline (prior map, no control)
    and with threshold-free metrics. A number without its baseline is
    not a result.
*   **Verified physics [A4–A6]:** Check numerics against analytic
    solutions, not only against regression snapshots. Use SI units at
    every interface.
*   **Sim-to-real is the finish line [B5]:** Wherever possible, every
    phase ends with something measured on real hardware.
*   **Showcase as you go [B7]:** Every phase ships a standalone demo:
    a scene, a figure, a video, or a write-up.
*   Work on each task or phase starts only with explicit user
    permission.

## 3. Status Snapshot

| phase | status | headline |
| --- | --- | --- |
| 0 Audit remediation | in-progress | no-audio baseline measured; it matches the Phase 2 headline |
| 1 Interactive simulation UI | completed | 2D/3D live UI, numba CPU + CUDA GPU backends |
| 1B Engine v2: physical fidelity | pending | PML, materials, rigid BCs, SI units, differentiable twin |
| 1C UI v2: the acoustic sandbox | pending | probes, auralisation, materials, arrays, sharing |
| 2 Sensing ("Ear") | reopened as 2R | v1/v2 pipeline built; the gain over the prior is not yet shown |
| 3 Beamforming ("Mouth") | pending (rewritten) | classical controllers, transaural, ANC, robustness |
| 4 Closed loop (sim) | pending (revised) | digital-twin bridge from sensing to control |
| 5 Reality: sim-to-real | pending (new) | browser acoustic lab, real rooms, real transaural demo |
| 6 Showcase & dissemination | pending (new) | WebGPU sandbox, gallery, explainers, papers, benchmark |

Decision gates:

| gate | criterion | unblocks |
| --- | --- | --- |
| G0 | Baselines and threshold-free metrics in every eval; all checkpoints re-scored | any further sensing claim |
| G1 | Engine verified: 2nd-order convergence, cavity modes within 0.5 %, energy conserved in lossless runs, PML reflection < −40 dB | Phase 3; the physics claims in any write-up |
| G2 | Sensing beats the prior-map baseline significantly on held-out rooms (positive information gain, p < 0.01) | using sensing output in Phase 4 |
| G3 | Sim control: ≥ 10 dB bright/dark contrast (Tier A) or ≥ 15 dB crosstalk separation (Tier L) in an absorbing, reverberant room | Phase 4 |
| G4 | Real echo time-of-flight → wall distance within a few cm; real T60 within 10 % of the simulated twin | Phase 5 transaural and real-data sensing |
| G5 | Closed loop keeps G3-level performance under a moving listener or obstacle, within a laptop latency budget | "system works" claim |

---

## 4. Phases

### Phase 0: Audit Remediation
**Objective:** Make every existing claim defensible and put the base in
order before new work builds on it.

*   `[in-progress]` **Task 0.1: Baselines and metrics [A1, A2]**
    *   `[completed]` 0.1.1: `scripts/eval_no_audio_baseline.py`.
        Prior map 0.104 ± 0.003 and a fixed interior box 0.103 match
        the v2 headline of 0.100. On v1 rooms, the prior (0.090) is at
        or above every published result.
    *   `[pending]` 0.1.2: Run the baseline on the exact archives.
        Add threshold-free metrics to `eval_multipose.py`:
        - information gain over the prior (bits/room)
        - AP
        - boundary F-score and Chamfer distance
        - skill vs the prior map

        Re-score every checkpoint and every value of K.
    *   `[pending]` 0.1.3: Restate the Phase 2 conclusions in
        `CURRENT_STATE.md`, `docs/learning.md` and the reports.
        Annotate superseded claims in place rather than deleting them.
*   `[pending]` **Task 0.2: Physics correctness [A4–A6]**
    *   `[pending]` 0.2.1: Rigid (Neumann, Γ = +1) walls and obstacles
        as a per-cell boundary type in the fused 2D/3D CPU and GPU
        kernels. Keep today's Dirichlet path bit-identical so
        `reference.npz` still holds. Correct the "rigid Dirichlet"
        wording in the code, `docs/simulate.md`, and `CLAUDE.md`.
    *   `[pending]` 0.2.2: SI scene layer: metres, seconds, Hz, and
        c = 343 m/s, mapped to grid units. Add a physical-plausibility
        report for datasets (mic baseline vs room size vs band) and
        pick a documented physical scale for future archives.
    *   `[pending]` 0.2.3: Verification suite with a report and figures:
        - analytic cavity eigenfrequencies (Dirichlet and Neumann)
        - grid-convergence order
        - discrete-energy conservation
        - dispersion vs von Neumann analysis
        - 2D/3D Green's-function comparison
*   `[pending]` **Task 0.3: Engineering hygiene [C1–C6]**
    *   `[pending]` 0.3.1: GitHub Actions CI:
        - ruff and ty
        - the 2D/3D check gates
        - the learning gates on CPU torch
        - `tsc` and `vite build` for the frontend

        Adopt pytest for `tests/`.
    *   `[pending]` 0.3.2: README rewrite: current feature set, fixed
        image paths, animated hero, "try it" link. Untrack `.DS_Store`
        and `.idea/`. Fix the section numbering in `CURRENT_STATE.md`.
    *   `[pending]` 0.3.3: Persist datasets and checkpoints to a remote
        store (Hugging Face Hub or release assets) with checksums and
        seed manifests.
    *   `[pending]` 0.3.4: Frontend: `build` and `typecheck` scripts.
        Split `App.tsx` into components. Add Playwright E2E tests to CI.

---

### Phase 1: Interactive Browser-Based Simulation UI — `[completed]`

*   `[completed]` 1.1 Stack: FastAPI + Socket.IO backend, Vite + React
    + TypeScript frontend.
*   `[completed]` 1.2 Backend control: a single-task step loop under an
    asyncio lock, a full lifecycle and geometry event set, and
    downsampled broadcast.
*   `[completed]` 1.3 Frontend:
    - `ImageData` rendering on a requestAnimationFrame loop
    - obstacle brush
    - drivers
    - parameter panel
    - Three.js volumetric 3D view with binary uint8 streaming
*   `[completed]` 1.4 Live interaction: the engine mutates geometry and
    drivers between steps. Playwright-verified.
*   `[completed]` 1.5 GPU: CuPy `RawKernel` twins, device-resident
    buffers. 13.2× at 2048² and 16.7× at 200³. See `docs/gpu.md`.

### Phase 1B: Engine v2 — Physical Fidelity
**Objective:** Make the simulator trustworthy for realistic rooms. This
is a prerequisite for Phase 3 [A4] and for any sim-to-real claim.

*   `[pending]` 1.6: Absorbing outer boundaries. Mur/Engquist–Majda as
    the quick win, then CPML. Enables open-field scenes and anechoic
    references.
*   `[pending]` 1.7: Materials: frequency-independent, then
    frequency-dependent, locally reacting impedance boundaries. Map
    absorption coefficient α to impedance Z per cell. Validate T60
    against the Sabine and Eyring formulas.
*   `[pending]` 1.8: Heterogeneous media with c(x) and ρ(x): lenses,
    temperature gradients, layered media. Decide between the
    variable-coefficient scalar form and a staggered p–v formulation.
*   `[pending]` 1.9: Low-dispersion schemes: compact explicit
    interpolated wideband (IWB) scheme and higher-order stencils. These
    buy usable bandwidth per cell, which directly relaxes the sensing
    resolution limit.
*   `[pending]` 1.10: Sources and receivers:
    - band-limited soft-source injection
    - sub-cell positions
    - directivity: dipole, cardioid, measured laptop-speaker patterns
    - mic self-noise and ADC model
*   `[pending]` 1.11: Differentiable engine: a PyTorch twin of the 2D
    and 3D kernels with autograd and gradient checkpointing, gated
    against the numba truth chain. It unlocks full-waveform inversion
    (2.5) and gradient-based control design (3.7).
*   `[pending]` 1.12: Throughput: batched multi-room kernels (many
    scenes per launch), an fp16-storage study, and 3D room-scale
    datasets on GPU.
*   `[pending]` 1.13 (stretch): A neural-operator surrogate (FNO) for
    millisecond forward prediction. Use it in control loops, with FDTD
    as the ground truth.

### Phase 1C: UI v2 — The Acoustic Sandbox
**Objective:** Turn the viewer into an instrument people want to play
with. It doubles as the lab bench for Phases 2–4 [B8].

*   `[pending]` 1.14 Probes: drop virtual mics anywhere. Each shows a
    live oscilloscope and a spectrogram. **Auralisation** lets you
    listen at any probe through Web Audio, with the sim-to-audible
    frequency mapping made explicit.
*   `[pending]` 1.15 Field views:
    - dB / log colour scale and diverging colormaps
    - time-averaged SPL (RMS) map
    - energy-decay curve and T60 readout
    - intensity vectors
    - wavefront isochrones
*   `[pending]` 1.16 Scene tools:
    - paint materials (absorption) and c(x)
    - per-face PML toggles
    - primitives: line, rectangle, ellipse, polygon
    - undo and redo
    - floor-plan image → obstacles
    - 3D: OBJ/STL mesh import with voxelisation, and slice editing
      (retires the deferred 3D-drawing item)
*   `[pending]` 1.17 Source tools: a per-driver editor for waveform,
    amplitude, phase, and delay. An **array tool** builds line, arc,
    or circular arrays with N elements and steering-angle and
    focus-point handles.
*   `[pending]` 1.18 Timeline: pause and scrub through a ring buffer of
    frames, single-step, and slow motion.
*   `[pending]` 1.19 Measurement mode:
    - impulse and frequency response between any two points
    - click a resonance to see its mode shape (harmonic drive)
    - reverberation metrics: T60, C50, DRR
*   `[pending]` 1.20 Persistence and sharing:
    - scene JSON save and load
    - URL-encoded shareable scenes
    - record to MP4 or GIF
    - export the current scene as a dataset sample

---

### Phase 2: Advanced Sensing ("Ear") — v1/v2 built, reopened as 2R

**History (compressed; details in `docs/learning.md` §1–8 and the
reports):**

*   `[completed]` 2.1: Active sensing:
    - `AudioFileWaveform`, the dataset generator, and 2-mic stereo
      recordings
    - `DualInputCNN` (single-pose held-out IoU 0.037)
    - multi-pose archives, Bayes fusion, and `JointPoseCNN`
*   `[completed]` 2.2: Passive sensing: `PassiveCNN` with randomised
    sources (held-out 0.030).
*   `[completed]` 2.3: Sensing v2:
    - phase channels
    - band at the spatial Nyquist limit and a longer recording
    - shape-diverse rooms
    - `SkipSensingCNN`
    - Platt calibration and the operating point (held-out 0.100 @ K=4)
*   **Superseded claims.** Task 2.3 showed that the "2.4× from fusion"
    of 2.1.4 was a fixed-threshold artefact. The audit [A1] shows that
    the 2.3 "information ceiling ≈ 0.10" equals the no-audio prior
    baseline (0.104 ± 0.003). No recipe has yet been shown to beat the
    prior on the IoU metric. Task 2.3's scientific findings still
    stand: the calibration mathematics, the threshold artefact, and
    that passive is close to active.

**2R research programme.** It is ordered: first establish whether and
where there is signal, then exploit it.

*   `[pending]` 2.4 Well-posed targets [A2]: replace the filled-mask
    IoU with quantities the recording can determine:
    - (a) illuminated-boundary maps, labelled from the simulated
      energy flux
    - (b) the room-boundary polygon
    - (c) signed-distance fields
    - (d) discrete object detection

    Keep IoU only as a legacy column.
*   `[pending]` 2.5 Physics baselines, no learning:
    - matched-filter or Wiener deconvolution to recover the RIR
    - echo labelling and image-source wall localisation (Dokmanić et
      al., "Acoustic echoes reveal room shape", PNAS 2013)
    - **synthetic-aperture backprojection** across K poses, the
      aperture-synthesis analogue of radio interferometry
    - time-reversal imaging
    - **full-waveform inversion** through the differentiable engine
      (1.11) as the high-effort reference

    This is the fastest way to learn whether the acquisition carries
    geometry at all.
*   `[pending]` 2.6 Physics-informed learning:
    - feed backprojection images as *spatially aligned* input
      channels, which fixes the "no geometric correspondence between
      time-frequency and room pixels" bottleneck
    - a deconvolved-RIR front-end
    - a pose-aware set transformer
    - unrolled or learned FWI
    - a diffusion prior with posterior sampling, which gives
      uncertainty maps
*   `[pending]` 2.7 Acquisition design and information theory: compute
    the Fisher information and CRLB for wall and obstacle positions as
    functions of bandwidth, mic count, baseline, SNR, and pose count.
    Estimate the mutual information. Deliverable: the "what can a
    laptop hear?" design chart. It fixes the physical scale (0.2.2)
    and the hardware tier.
*   `[pending]` 2.8 Active sensing: choose the next-best pose by
    expected information gain. The UI tells the user where to move the
    laptop next.
*   `[pending]` 2.9 Unknown poses [A3]:
    - pose-noise robustness curves
    - acoustic SLAM (joint pose + map)
    - echo-based self-localisation against walls
*   `[pending]` 2.10 Passive extensions:
    - GCC-PHAT and passive multi-pose (carried over)
    - ambient-noise interferometry: cross-correlating diffuse noise
      recovers inter-mic Green's functions (the seismology technique)
*   `[pending]` 2.11 Room-acoustic parameter estimation: T60, DRR,
    volume, and per-wall absorption, blind from speech or music. This
    is well-posed, useful on its own, and exactly what control needs
    (3.5).
*   `[pending]` 2.12 Scale-up: 3D rooms (GPU), finer grids and higher
    relative bandwidth, and 3–4 mic arrays (common on modern laptops),
    driven by the conclusions of 2.7.

---

### Phase 3: Beamforming and Sound-Field Control ("Mouth") — rewritten
**Objective:** Algorithms that create localised sound and silence,
first in simulated rooms with realistic absorption. Requires 1.6–1.7
and G1.

*   `[completed]` 3.0: Multiple independently driven sources. The
    engine and UI support N drivers with independent waveforms. This
    was 3.1.1 [B2].
*   `[pending]` 3.1 Transfer-function tooling: batched GPU computation
    of speaker → control-point impulse responses and ATFs; control-point
    grids; frequency- and time-domain views.
*   `[pending]` 3.2 Classical controllers in `control/`:
    - delay-and-sum and superdirective steering
    - regularised pressure matching
    - acoustic contrast control (a generalised eigenproblem)
    - ACC–PM hybrids
    - MVDR
    - time-reversal focusing
    - broadband FIR design

    Metrics: acoustic contrast (dB), reproduction error, array effort,
    and white-noise gain.
*   `[pending]` 3.3 Tier L — **transaural virtual headphones**:
    crosstalk cancellation with 2 laptop speakers. Measure channel
    separation against head position and frequency, sweet-spot size,
    and regularisation, including head-tracked adaptive filters.
*   `[pending]` 3.4 Active noise control: FxLMS feedforward and feedback
    ANC at a quiet zone. Show the physics limits: quiet-zone size is
    about λ/10, and causality constrains feedforward control.
*   `[pending]` 3.5 **Control-oriented sensing requirement:** design
    controllers on the *inferred* room (Phase 2 output or estimated
    parameters) and evaluate them on the *true* room. Deliverable: the
    curve of contrast against sensing error. It tells Phase 2 how good
    sensing must be, which is the key scientific bridge between Ear and
    Mouth.
*   `[pending]` 3.6 UI:
    - array placement
    - bright and dark zone painting
    - a live contrast (dB) readout
    - a beam-steering slider
    - before and after SPL maps
*   `[pending]` 3.7 Differentiable control (needs 1.11): optimise driver
    signals end to end through the simulator for broadband zone
    contrast in reverberant rooms. Compare with 3.2.

---

### Phase 4: Closed-Loop System Integration (sim) — revised
**Objective:** Connect Ear and Mouth in a live loop inside the
simulator.

*   `[pending]` 4.1 Digital-twin bridge [B3]: sensing yields a
    geometry, material, or parameter estimate. The twin re-simulates
    ATFs to the targets, and the controller is designed from them. The
    alternative path estimates ATFs at control points directly. Choose
    using 3.5.
*   `[pending]` 4.2 Real-time loop in the backend: sense → update the
    twin → redesign filters → act. Add a per-stage latency budget
    measured on laptop CPU and incremental twin updates (warm-started
    FWI or surrogate).
*   `[pending]` 4.3 Dynamic scenarios: a moving listener, obstacle, or
    door. Measure adaptation speed and stability, and compare against
    a controller given the true twin (the oracle).
*   `[pending]` 4.4 UI closed-loop dashboard:
    - live sensing map with uncertainty
    - twin vs truth
    - controller state
    - zone metrics over time

---

### Phase 5: Reality — Sim-to-Real (new)
**Objective:** Ground every simulated claim in real measurements on
consumer hardware [B5]. Start early, because the first items are cheap.

*   `[pending]` 5.1 Browser acoustic lab: a Web Audio app
    (getUserMedia + AudioWorklet) that turns off AGC, echo
    cancellation, and noise suppression. It plays a chirp through the
    laptop speakers, records the mics, deconvolves the RIR, and
    displays it. It lives in the same frontend.
*   `[pending]` 5.2 First real numbers:
    - echo time-of-flight → distance to the nearest wall or desk
    - real-room T60
    - the same room rebuilt as a simulated scene and compared (G4)
*   `[pending]` 5.3 Calibration: speaker and mic frequency responses,
    I/O latency, and inter-channel sync, stored in a device-profile
    database.
*   `[pending]` 5.4 Real dataset:
    - a capture protocol with tape-measured rooms and poses
    - a small real test set
    - evaluation of sim-trained models on it
    - domain randomisation (1.10 noise, 1.7 materials, 3D) to close
      the gap
    - cross-checks against public measured-RIR datasets
*   `[pending]` 5.5 Real transaural demo: webcam head tracking
    (MediaPipe face landmarks, in-browser) drives the 3.3 crosstalk
    cancellation filters. The result: **virtual headphones from your
    laptop speakers**.
*   `[pending]` 5.6 Phones as moving sensors: the IMU supplies pose,
    which answers A3. Phone + laptop multi-device apertures.

---

### Phase 6: Showcase and Dissemination (new)
**Objective:** Make the work easy to see, try, trust, and cite [B7].

*   `[pending]` 6.1 **Zero-install sandbox:** a WebGPU compute-shader
    FDTD for 2D and 3D, with a WebGL2 fallback. Sensing models are
    exported to ONNX and run in onnxruntime-web. The whole sandbox runs
    client-side on GitHub Pages; the Python backend remains for
    datasets and training.
*   `[pending]` 6.2 Scene gallery: one-click presets, each with a short
    physics caption:
    - double slit and diffraction
    - whispering-gallery ellipse
    - parabolic focusing
    - gradient-index acoustic lens
    - Helmholtz resonator
    - room modes
    - anechoic vs reverberant
    - time-reversal refocusing through a chaotic cavity
    - sonic-crystal band gap
    - steered array
    - quiet zone
    - bat echolocation
*   `[pending]` 6.3 Interactive explainer series (scrollytelling with
    live embedded simulations):
    - FDTD from scratch
    - CFL and dispersion
    - boundaries and PML
    - room acoustics
    - echolocation and its limits (including the baseline lesson)
    - beamforming
    - closing the loop
*   `[pending]` 6.4 Media pipeline: a deterministic headless renderer
    (scene JSON → MP4/GIF) for the README hero, the reports, and social
    posts.
*   `[pending]` 6.5 Write-ups and preprints:
    - (i) *"What can a laptop hear? Baselines and information limits
      for few-microphone acoustic room mapping"*, which turns Phase 2,
      its negative results included, into a rigorous contribution
    - (ii) the control-oriented sensing requirement (3.5)
    - (iii) sim-to-real laptop transaural audio (5.5)
*   `[pending]` 6.6 Open benchmark: datasets, generators, and baselines,
    with the no-audio baseline built in, plus a leaderboard on the
    Hugging Face Hub.
*   `[pending]` 6.7 Docs site (mkdocs-material from `docs/`), a
    per-commit performance dashboard from CI benchmarks, and badges.

## 5. Suggested Sequencing

1. **Now:** Phase 0 (0.1.2–0.1.3, 0.2.1, 0.3.1–0.3.2). This is days of
   work and makes the base honest.
2. **Unblockers:** 1.6 PML and 1.7 materials together with 0.2.3
   verification, which reaches G1.
3. **In parallel:**
   - *Science:* 2.5 physics baselines and 2.4 targets answer "is there
     signal?"
   - *Showcase:* 1.14 probes and auralisation, then a 6.1 WebGPU
     prototype.
   - *Reality:* 5.1 browser acoustic lab, which is cheap and grounds
     everything that follows.
4. Phase 3 (3.1–3.3, then 3.5), gated by G1.
5. Phase 4 behind G2 and G3, alongside 5.2–5.5.
6. The 6.5 write-ups, as each result lands.
