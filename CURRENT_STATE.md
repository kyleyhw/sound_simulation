# Project Status as of 2026-05-13

## 1. Resolution

The interactive editing batch (Phase 1, Tasks 1.3.2 / 1.3.3 / 1.4) is
functional end-to-end. The web UI now supports click-and-drag obstacle
drawing, click-to-place drivers, click-to-remove drivers, a parameter
panel that rebuilds the engine with new geometry / waveform / cadence,
and live status reflecting every mutation. The engine, backend, and
frontend changes were validated together with the perf gate
(`tests/perf/check_simulate.py`) and a Playwright smoke test that drew
obstacles, placed a second driver, and cleared obstacles, all while
the simulation continued advancing.

## 2. What changed

### Engine (`src/acoustic_system/simulation/simulate.py`)

- New `obstacle_mask: np.ndarray` (boolean, same shape as the field).
  Interior cells flagged True are scrubbed to zero inside `step()`
  after the stencil pass but before driver injection, giving them
  rigid-wall (Dirichlet) semantics consistent with the outer boundary.
- The scrub is guarded by a cached `_has_obstacles` flag, so a
  Simulate with no obstacles takes a code path that is bit-identical
  to the pre-obstacle code. `check_simulate.py` keeps passing against
  the existing `reference.npz` at `max_abs=7.75e-7`, `l2_rel=1.04e-6`.
- New methods: `set_obstacle(positions, value=True)`,
  `clear_obstacles()`, `add_driver(driver)`, `remove_driver(index)`,
  `set_drivers(drivers)`. The single-driver fast-path cache
  (`_fast_driver`, `_fast_driver_pos`) is now refreshed on every
  driver mutation rather than only at construction.

### Backend (`src/acoustic_system/app/main.py`)

- `SimulationManager` gained `set_obstacle`, `clear_obstacles`,
  `add_driver`, `remove_driver`, `clear_drivers`. Each acquires the
  asyncio lock briefly to coordinate with `configure`/`reset`/`start`,
  mutates the engine, and re-broadcasts `status`.
- `_broadcast_status` payload now includes `obstacles` (downsampled
  bool mask + shape/downsample metadata) and `drivers` (list of
  `{position, waveform}`). Geometry lives on the `status` channel,
  not on the per-frame `simulation_update`.
- Five new socket events wired: `set_obstacle`, `clear_obstacles`,
  `add_driver`, `remove_driver`, `clear_drivers`.

### Frontend (`frontend/src/App.tsx`, `frontend/src/index.css`)

- Side-panel layout: canvas on the left, controls on the right
  (collapses below on narrow viewports).
- Mode selector with five modes: view, draw-obstacle, erase-obstacle,
  place-driver, remove-driver. The cursor changes per mode.
- Brush radius input for the obstacle modes.
- Mouse-down / move / up handlers translate canvas pixels to full-grid
  indices, compute the disc of touched cells client-side, and batch
  them in a `Set<string>` with a ~33 ms flush window so a brush stroke
  emits ~30 socket events/sec regardless of brush size.
- Parameter panel: waveform (type, amplitude, frequency, delay),
  grid rows/cols, courant, downsample, broadcast Hz. Apply emits
  `update_config`.
- Driver list panel with per-driver remove buttons.
- Rendering composites the obstacle mask (gray pixels) into the
  ImageData buffer, then draws yellow disc markers for each driver
  on the canvas overlay. Both stay aligned with the field because
  the obstacle mask is downsampled by the same factor as the
  pressure field.

### Cleanup

- `numba` added to `environment.yml` (it was a required dep of the 2D
  FDTD kernel but had been missing from the env file, so fresh installs
  failed on first `Simulate.step()`).
- `README.md` "Code Structure" block rewritten to reflect the
  `src/acoustic_system/` layout. The legacy flat layout reference is
  gone.

### Docs

- `docs/simulate.md` gained a new section on interior obstacles
  covering the physical interpretation, the ordering inside `step()`,
  the hot-loop guard, and the mutation methods.
- `docs/web_ui.md` updated wire-protocol tables (new events on both
  directions, new fields in `status`) and added a section on the
  interactive editing modes and coordinate mapping.

## 3. Next steps

Phase 1.5 (GPU acceleration via CuPy) is the only outstanding Phase 1
task. Phase 2 (advanced sensing — CNN training on simulated
echolocation data) is queued behind it. The interactive UI now
provides the scene-authoring tool the rest of the project needs.

Open follow-ups noticed during this batch but not landed:

- Per-driver waveform editing in the side panel (currently every
  `place-driver` click uses the form's current waveform; you cannot
  change an existing driver's waveform without removing and replacing
  it).
- Sensor placement parity with drivers — the engine still only
  exposes sensors through the batch generator path, not through the
  UI.
- `update_config` discards drivers/obstacles on rebuild. Preserving
  them across a structural rebuild would need an explicit migration
  step (project the obstacle mask onto the new grid shape, clip
  driver positions).
