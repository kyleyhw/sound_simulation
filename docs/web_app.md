# Web app (`web/`): the Acoustic Sandbox

The browser application replaces the previous Socket.IO UI (`frontend/` +
`app/main.py`). The simulator runs **entirely in the browser**, so the app
is a static site hosted on GitHub Pages (`.github/workflows/pages.yml`) with
no server.

## 1. Why the UI was rebuilt (plan 4.1.1)

Inventory of the previous UI:
- A single ~1.1k-line `App.tsx` with 40+ `useState`/`useRef` variables.
- A fixed 600 px canvas.
- JSON nested-list frames sent over a socket.
- The Python server was required, so there was no static hosting.
- The backend audit (`tests/reports/debug_audit_2026_09_24.md`) found 12
  bugs. Three were critical:
  - one malformed `update_config` blocked every new client
  - one config change could freeze the event loop for 61 s
  - clients could make the server open arbitrary file paths

Features the old UI had, all kept:
- 2D and 3D views
- obstacle brush and eraser
- driver placement
- waveform, grid and cadence settings
- the sensing panel

Features it lacked:
- recordings, spectra and listening
- dB, RMS and energy-flow views
- materials and absorbing boundaries
- undo and redo
- saving, loading and sharing scenes
- a history scrubber
- presets
- export
- keyboard shortcuts
- a responsive layout

## 2. UX specification (4.1.2)

Layout. The sandbox route is a four-region grid:

| region | content |
| --- | --- |
| left rail | tools: select, brush, eraser, line, rectangle, ellipse, source, microphone |
| centre stage | run controls, speed, undo/redo, share and export toolbar above the field canvas, with a HUD (step, time, fps, cursor cell) and a colour legend |
| bottom dock | recording of the selected microphone (scope + spectrogram or spectrum), a Listen button, and the history scrubber |
| right inspector | tabs: Scene (presets, grid, units, boundaries, materials), Sources, Mics, View, Sensing, Control |

Below 980 px the regions stack vertically (rail as a horizontal strip).

Main flows:
1. Open the app, press Space: the default scene runs. First-run hints
   explain the basics.
2. Gallery → pick an experiment → it opens in the sandbox.
3. Draw walls → place a source → place a microphone → run → listen.
4. Share: the scene is compressed into the URL (`#/sandbox?s=…`).

Other routes:
- Learn: interactive explainers.
- Research: results, benchmark and write-ups.
- Lab: real-hardware measurement with the laptop's speakers and mics.

## 3. Visual design (4.1.3)

- **Themes:** dark ("instrument") and light, as CSS custom properties on
  `:root` / `[data-theme=light]`. The default follows
  `prefers-color-scheme`, and the choice is remembered.
- **Pressure colormaps** (signed data, diverging): `icefire` is
  dark-centred for the dark theme, `balance` is light-centred for the
  light theme.
- **Magnitude colormaps** (RMS, dB, spectrograms, sequential): `magma`,
  `viridis`.
- **Walls** are drawn in material-specific neutral colours, so they read
  as geometry rather than data.
- **Marker colours:** sources are rose circles, microphones amber
  triangles.

## 4. Architecture (4.1.4)

```
web/src/
  engine/   simulation.ts   FDTD engine (TypeScript port of Simulate)
            waveforms.ts    source waveforms (same formulas as Python)
            scene.ts        serialisable scene, RLE, share-URL codec
            presets.ts      gallery experiments
  state/    store.ts        zustand store: scene, tools, view, undo/redo
            runtime.ts      live Simulation + rAF loop + frame history
            editable.ts     in-memory scene form (byte maps)
  render/   fieldRenderer.ts  WebGL2 field renderer (Canvas2D fallback)
            colormaps.ts
  components/ Viewport, Volume3D, ToolRail, StageToolbar, Inspector, Dock, …
  pages/    Sandbox, Gallery, Learn, Research, Lab
  lib/      dsp (FFT, spectrogram), audio (Web Audio), exporters, geometry, router
```

**State.** The zustand store owns the *scene* (the single source of
truth for geometry and sources) and the UI state. The `Runtime` owns the
live `Simulation` outside React, mirrors store edits into it, and runs
the `requestAnimationFrame` loop. Stepping is bounded per frame by a step
count (the speed setting) and a 12 ms budget, so rendering stays smooth
on any grid. Every edit goes through a store action that snapshots the
scene for undo/redo, up to 60 levels.

**Rendering.** The field is uploaded each frame as an R32F texture at
native grid resolution and mapped through a 256-entry colormap in a
fragment shader. Linear and dB scaling happen in the shader. Nearest
sampling keeps cells crisp. Materials are a second texture composited in
the same pass. Markers, previews and the brush cursor are an SVG
overlay; energy-flow arrows are a Canvas2D overlay. 3D grids show either
an editable slice or a ray-marched volume (Three.js). Drawing tools paint
in the current slice plane.

**Engine.** The engine has two paths:
- **Fast path.** p = 0 walls and obstacles, uniform c. It is the exact
  port of the Python kernels, and parity is tested against Python
  fixtures (`tests/unit/parity.test.ts`, gate 1e-4).
- **General path.** Rigid and impedance walls, the absorbing layer, and
  c(x), with precomputed per-cell coefficients. The physics is
  documented in `docs/physics.md`.

## 5a. Sound-field control panel (plan 7.7)

The **Control** tab is `components/ControlPanel.tsx`, backed by
`control/soundfield.ts` and `control/complex.ts`. The workflow:

1. **Zones.** The Zone tool (Z) drags a loud (bright) and a quiet (dark)
   rectangle. They live in the store (`zones`), not in the scene file.
2. **Array.** Place N speakers on a line (count, spacing, centre,
   orientation). They are ordinary drivers with ids `arr-*`.
3. **Measure and design.** The browser engine drives each speaker in turn
   with cos(ωt). Once the room has settled, it reads the steady-state
   transfer function H by a DFT over whole periods at up to 40 points per
   zone. The walls, materials and c(x) all come from the current scene.
   The panel predicts contrast for four designs, all scaled to the same
   array effort (‖w‖² = N):
   - delay-and-sum (free-field alignment);
   - focus (time reversal, conj H);
   - pressure matching (regularised least squares);
   - acoustic contrast control (the principal generalised eigenvector of
     R_b and R_d + δI).
4. **Apply.** Each weight w_s = g_s e^{−jωd_s} becomes the driver's `gain`
   and `delay`, so the drivers play g cos(ω(t − d)).
5. **Live readout.** The measured bright/dark contrast comes from the
   time-averaged loudness map (RMS overlay). *Reset averaging* drops the
   start-up transient.

Verified in `tests/unit/soundfield.test.ts`: an 8-speaker array in a
90² CPML room at 20 cells per wavelength.

| design | contrast |
|---|---|
| delay-and-sum | 19.1 dB predicted |
| pressure matching | 21.2 dB predicted |
| ACC | 46.6 dB predicted, **44.8 dB measured** in the time domain |

`tests/e2e/control.spec.ts` runs the same flow through the UI and checks
that the live measured contrast exceeds 10 dB.

## 5b. Closed-loop dashboard (plan Phase 8)

`#/loop` (`pages/Loop.tsx`) runs `loop/closedLoop.ts` in a Web Worker
(`loop/loopWorker.ts`) on the scenario in `loop/scenarios.ts`: an
8-speaker bar in an anechoic (CPML) room, over four epochs:

1. initial room;
2. the listener (loud zone) moves;
3. the obstacle moves;
4. a partition with a door appears.

Each epoch runs:

- **Sense:** every speaker pings and every array position records. The
  residual against an empty-room reference is back-projected by
  *coherent* delay-and-sum migration (`migrationImages`). This gives the
  coherent image, the coherent images of the left and right half of the
  array, and the smoothed coherent and incoherent energy. Two estimators
  turn these images into an obstacle mask (the **Room estimate** selector):
  - **Learned U-Net** (default, 100² grid): `loop/learnedSensing.ts`. A
    compact U-Net (120 k parameters, BatchNorm folded) maps the aligned
    images, plus geometry channels and a training prior, to an obstacle
    probability. The estimate is the cells above 0.62, a threshold chosen
    on validation scenes.
  - **Back-projection:** the largest blob of the coherent energy above
    0.7 × max, dilated by one cell. An envelope sum only resolves range
    and smears whole arcs.

  The model is trained for the 100² grid only. On 60² and 80² the loop
  falls back to back-projection, and the page says so.
- **Twin:** the true outer boundary plus the estimate as rigid cells.
- **Design:** ACC on the twin.
- **Act and measure:** the steady-state contrast over the whole zones in
  the true room.
- **Guard:** five monitor mics per zone score the twin design and the
  empty-room design in the true room, and the loop keeps the better one.

Each epoch also scores three references: a **static** controller designed
once at epoch 0, an **empty-room** design (no sensing), and an **oracle**
designed on the true room.

### Learned room estimate (loop sensing study, 2026-09-25)

Report: `tests/reports/loop_sensing_2026_09_25.md`.

- **Data.** `web/scripts/loop_sensing_data.ts` (`npm run loop:data`) runs
  the loop's own sensing code in Node on random scenes from
  `scenarios.randomLoopScene`. Each scene has 1–3 rigid blocks or thin
  partitions with a door, kept at least 12 cells from the bar, and a
  bright and a dark zone. The splits are 2400 train, 300 validation and
  100 test scenes, with disjoint seeds.
- **Training.** `scripts/train_loop_sensing.py` trains on CPU (BCE +
  Dice, 40 epochs) and exports `web/public/models/loop_unet.{bin,json}`
  and the parity fixture. `tests/unit/loopSensing.test.ts` re-simulates
  held-out scenes in TypeScript and matches PyTorch's features to 2e-5
  and its logits to 2e-3.
- **Evaluation.** `web/scripts/loop_sensing_eval.ts`
  (`npm run loop:eval`) runs a paired comparison on all 100 held-out test
  scenes. The results below are mean ± SE in dB, measured in the true
  room.

| design | sensing IoU | contrast (dB) |
|---|---|---|
| empty room (no sensing) | — | 14.7 ± 1.1 |
| back-projection twin | 0.18 ± 0.01 | 16.9 ± 1.2 |
| guarded, back-projection | | 17.8 ± 1.2 |
| **learned twin** | **0.79 ± 0.02** | **26.3 ± 1.2** |
| guarded, learned | | 26.4 ± 1.2 |
| oracle (true room) | 1 | 30.6 ± 1.0 |

The paired differences are:

- Learned twin minus back-projection twin: **+9.5 ± 0.7 dB** (z = 12.6;
  better in 93 % of scenes).
- IoU: +0.61 ± 0.02 (z = 29.6).

The learned estimate closes **69 %** of the back-projection twin's
13.7 dB gap to the oracle. The remaining gap is 4.3 ± 0.6 dB, with a
median of 1.0 dB. The guard now keeps the twin in 93 % of scenes (66 %
with back-projection). Only 9 of 100 scenes stay below 10 dB (28 with
back-projection, 2 for the oracle).

Demo epochs (contrast in dB; the IoU is shown after the slash):

| epoch | back-projection twin | learned twin | empty room | oracle |
|---|---|---|---|---|
| initial | 35.8 / 0.18 | 45.4 / 1.00 | 33.0 | 45.4 |
| listener moves | 40.7 / 0.18 | 50.3 / 1.00 | 36.6 | 50.3 |
| obstacle moves | 9.6 / 0.18 | 34.6 / 1.00 | 19.4 | 34.6 |
| partition + door | 15.5 / 0.13 | 29.9 / 1.00 | 16.6 | 29.9 |

With the learned estimate, the closed loop matches the oracle on all four
demo epochs. With back-projection, the guarded loop scored
35.8/40.7/19.4/16.6 dB and fell back to the empty-room design in epochs
3 and 4.

Caveats. Training and testing use the same noiseless simulator, grid and
bar, with the same families of axis-aligned blocks and partitions, so the
network can use a strong shape prior.

- **Recording noise.** The IoU is 0.78 at an echo SNR of 30 dB, 0.71 at
  20 dB and 0.36 at 10 dB. Back-projection stays at 0.18 at every level.
- **Out-of-family shapes.** Discs, L-shapes and diagonal walls were never
  trained on. On these the IoU falls to 0.32, against 0.18 for
  back-projection, and the network draws rectangles.
- **Control on out-of-family rooms.** On 40 such rooms the control gain
  vanishes. The learned twin minus the back-projection twin is
  +0.7 ± 1.0 dB (z = 0.7), and the guarded loops are equal. The monitor
  guard is what keeps the loop safe there.

Real rooms add model mismatch that these tests do not cover.

Latency, measured single-threaded in Node on the 4-core container at 100²:

| stage | time |
|---|---|
| sense (2 × 8 pings: the room and the empty-room reference) | about 2.3 s |
| learned estimate (U-Net forward on the central 96², in the worker) | about 0.27 s |
| design (8 steady-state tones on the twin) | about 2.3 s |
| act and measure | about 0.3 s |

This is a loop period of about 5 s, which is room-change pace, not
audio-rate. Applying new weights is instant.

Tests:

- `tests/unit/loop.test.ts` checks that the back-projection estimate lies
  on the obstacle. It also checks that the guarded loop stays at or above
  10 dB, at or above the static design, and at or below the oracle.
- `tests/unit/loopSensing.test.ts` holds the PyTorch parity checks. It
  also checks that the loop uses the learned estimate at 100² (IoU above
  0.8 on the demo), and falls back to back-projection on other grids.
- `tests/e2e/loop.spec.ts` runs the dashboard end to end at 60², where it
  falls back to back-projection, and checks the learned estimate on the
  first epoch at 100².

### Echo vision page (`#/echo`)

A toy front door for the learned room estimate: "machine learning
reconstructs a room from its echoes". It sits in the nav right after the
sandbox and uses the loop's sensing setup unchanged: the 100² CPML room,
the 8-speaker bar, and the U-Net above.

1. **The room.** Pick a named example (the demo's rooms, plus a round
   pillar and a diagonal wall as hard cases), press **New random room**
   (`randomLoopScene`, so rooms match the training family), or draw.
   Drawing is a rectangle drag: **Block**, **Wall** (snaps to the dominant
   axis, 2 cells thick) or **Erase**. It is clamped to the area that
   `randomLoopScene` uses (rows 8–74, columns 8–91), away from the bar.
   **Hide the room** hides the truth so the viewer can guess along.
2. **Listen.** A worker (`echo/echoWorker.ts`) runs the 8 pings. It loads
   the model and records the empty-room reference once, when the page
   opens. It streams the live pressure field of the current ping, and the
   page shows "Ping k of 8". A Listen takes about 3 s (2.9–3.2 s in
   headless Chromium in the cloud container).
3. **What the echoes show:** the back-projection energy image, with its
   brightest-blob estimate outlined, and its IoU.
4. **What the network reconstructs:** the U-Net probability map, with the
   thresholded estimate outlined, and its IoU. Both maps sit next to the
   true room, with a toggle for the true outline and a one-line verdict.

A room outside the training family gets an honest note. This covers
non-rectangular objects, objects larger than 12 × 36 cells, and more than
six pieces. The note reads: "It was trained on boxes and walls, so it
draws boxes." Details sit in a collapsed "How it works" section, which
links to the report.

Code:

- `echo/room.ts`: the room model, drag, clamp, paint and the family check.
- `echo/sense.ts`: `recordPings` is `pingRecordings` with a per-step frame
  hook. `analyseEchoes` is the rest of `senseRoom`.
- `echo/draw.ts`: the canvas panels and the cell-edge outlines.
- `pages/EchoVision.tsx` and `echo/echo.css`: the page.

Tests:

- `tests/unit/echo.test.ts` covers drawing and clamping, the outlines and
  the family check. It checks that `recordPings` equals `pingRecordings`
  exactly, and that the network beats back-projection on the door
  example.
- `tests/e2e/echo.spec.ts` covers the listen flow and the IoU display,
  drawing, hiding and the out-of-family note, and the absence of
  horizontal overflow at 390 px.

## 5c. WebGPU engine (plan 10.1, 10.2)

`engine/gpu.ts` runs the simulation in WGSL compute shaders, and the
**Engine** selector in the stage toolbar switches between CPU and GPU.
The CPU `Simulation` still owns the model. The GPU mirrors it from
`Simulation.deviceState()`, the precomputed general-update coefficients
(C, S, Q, Q/a, 1/a, K_s, K and the active flags). The same update also
expresses the fast path, so one kernel covers:
- p = 0, rigid and impedance walls and materials;
- the sponge and c(x);
- the CPML, as two extra passes for ψ and ζ;
- Mur edges, as one pass per axis in the CPU's face order.

It works in 2D and in 3D.

Each frame encodes a batch of steps. Driver values for the batch come from
the CPU waveform code, and probe samples and RMS accumulate on the device.
After each batch, p and p_prev are read back, so rendering, probes,
history and the instability guard work unchanged. Switching back to the
CPU first reads the full state (branch and CPML memory), so a run
continues seamlessly. Intensity arrows need the CPU engine, and the app
switches automatically.

Parity (`tests/e2e/gpu.spec.ts`) runs the CPU and GPU engines on eight
scenes. The relative field error is 7e-7 to 2e-6, and 4e-5 on a
low-amplitude 3D CPML case. The scenes cover:

- the fast path;
- rigid ellipse;
- impedance materials;
- c(x) lens;
- CPML with obstacles;
- the quiet-zone array;
- mixed Mur/impedance/sponge faces;
- a 3D CPML box.

Headless Chromium in CI provides a SwiftShader (CPU-emulated) WebGPU
adapter, which checks correctness, not speed. Speed on real GPUs has not
been measured here.

## 5. Testing (4.5)

| layer | tool | what |
| --- | --- | --- |
| unit | Vitest (`npm test`) | Python parity 2D/3D, energy conservation (closed box, rigid box), absorption, scene/RLE/URL round-trips, every preset runs finite, DSP, geometry, throughput guard |
| end-to-end | Playwright (`npm run e2e`) | one test per feature: run/pause/step/reset, shortcuts, brush/eraser/undo/redo, shape tools, sources, microphones + listen, share link, scrubbing, view modes, grid/units/boundaries + input validation, 3D slice + volume, theme + help, PNG export, save, gallery, phone layout (no horizontal overflow) |

The store is exposed as `window.__app`, so end-to-end tests assert on the
engine's actual state, not just pixels.

Throughput (Node 22, one core of the cloud container):

| scene | steps/s |
| --- | --- |
| 2D 256², fast path | 3,600 |
| 2D 200×240, rigid ellipse | 3,100 |
| 2D 256², absorbing layer | 1,700 |
| 3D 64³ | 505 |

The UI renders at display rate and steps up to the speed setting per
frame.

## 6. Commands

```bash
cd web
npm install
npm run dev        # http://127.0.0.1:3000
npm test           # unit tests
npm run e2e        # end-to-end (builds + previews on :4173)
npm run build      # static site in web/dist (BASE=/repo/ for Pages)
```
