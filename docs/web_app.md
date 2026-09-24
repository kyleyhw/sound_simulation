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

- **Sense:** every speaker pings, every array position records, and the
  residual against an empty-room reference is back-projected by
  *coherent* delay-and-sum migration. The largest blob above 0.7 × max,
  dilated by one cell, is the obstacle estimate. An envelope sum only
  resolves range and smears whole arcs.
- **Twin:** the true outer boundary plus the estimate as rigid cells.
- **Design:** ACC on the twin.
- **Act and measure:** the steady-state contrast over the whole zones in
  the true room.
- **Guard:** five monitor mics per zone score the twin design and the
  empty-room design in the true room, and the loop keeps the better one.

Each epoch also scores three references: a **static** controller designed
once at epoch 0, an **empty-room** design (no sensing), and an **oracle**
designed on the true room.

Results on the 100² scene (exploration run):

| epoch | closed loop | twin | empty room | static | oracle |
|---|---|---|---|---|---|
| initial | 35.8 (twin) | 35.8 | 33.0 | 35.8 | 45.4 |
| listener moves | 40.7 (twin) | 40.7 | 36.6 | 39.8 | 50.3 |
| obstacle moves | 19.4 (empty) | 9.6 | 19.4 | 16.0 | 34.6 |
| partition + door | 16.6 (empty) | 15.5 | 16.6 | 15.4 | 29.9 |

(All values are contrast in dB; the closed-loop column names the design the guard kept.)

What this shows:

- The guarded loop holds the 10 dB target through every change, and never
  does worse than the static design.
- The 10–15 dB gap to the oracle is the sensing gap. The back-projected
  twin finds the obstacle's front face (IoU 0.13–0.32 against the full
  block), which sometimes helps and sometimes hurts compared with
  assuming an empty room.

Latency, measured single-threaded in Node on the 4-core container at 100²:

| stage | time |
|---|---|
| sense (2 × 8 pings) | about 2.3 s |
| design (8 steady-state tones on the twin) | about 2.3 s |
| act and measure | about 0.3 s |

This is a loop period of about 5 s, which is room-change pace, not
audio-rate. Applying new weights is instant.

Tests: `tests/unit/loop.test.ts` checks that the estimate lies on the
obstacle, and that the guarded loop stays at or above 10 dB, stays at or
above the static design, and stays at or below the oracle.
`tests/e2e/loop.spec.ts` runs the dashboard end to end.

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
