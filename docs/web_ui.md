# Documentation for the interactive web UI

## 1. Purpose

The browser UI is the project's first end-to-end deliverable: an
interactive front-end for the FDTD engine that streams the live pressure
field to the browser as the simulation runs.

## 2. Architecture

```
Browser                     Vite dev server          Uvicorn / FastAPI / Socket.IO
React + canvas  <-- ws -->  proxy /socket.io  <-->   SimulationManager  ──>  Simulate
( ImageData )                  (port 3000)              (port 8001)            (NumPy / SciPy)
```

Two processes:

- **Backend** (`scripts/run_ui_server.py`) hosts a FastAPI app and a
  Socket.IO async server on `127.0.0.1:8001`. Inside it,
  `SimulationManager` owns the `Simulate` engine and a single
  `asyncio` task that runs the step loop and broadcasts state.
- **Frontend** (`frontend/`) is a Vite + React + TypeScript dev server
  on `127.0.0.1:3000`. `vite.config.ts` proxies `/socket.io` (HTTP and
  WebSocket) to the backend.

### Wire protocol

Client → server events:

| event              | payload                          | effect |
| ------------------ | -------------------------------- | ------ |
| `start_simulation` | `{}`                             | start the step loop |
| `stop_simulation`  | `{}`                             | stop the step loop |
| `reset_simulation` | `{}`                             | stop + zero the field, time, step |
| `update_config`    | `{grid_shape, driver_position, waveform: {type, ...}, ...}` | rebuild the engine |
| `request_status`   | `{}`                             | re-emit current `status` |

Server → client events:

| event               | payload | meaning |
| ------------------- | ------- | ------- |
| `simulation_update` | `{ grid: number[][], max_val: number, step: int, time: float }` | downsampled pressure field |
| `status`            | `{ is_running: bool, engine: {...}, config: {...} }` | manager state for UI sync |

## 3. Why the previous version did not work

The prior implementation had several compounding bugs:

1. **DC source.** The default cosine had $f \Delta t = 2.0$, so
   $\cos(2 \pi f \Delta t \, n) = 1$ for every integer step $n$ — the
   "oscillator" was actually a constant. The field grew monotonically
   and no waves were visible. (See `docs/waveforms.md`.)
2. **Race on start.** `is_running` was set inside the broadcast
   coroutine after the first `await`, so two near-simultaneous
   `start_simulation` events spawned two parallel step loops on the
   same field.
3. **Auto-start on connect** + manual *Start* — duplicate sims again.
4. **Orphan emit tasks.** `asyncio.create_task(sio.emit(...))`
   discarded the handle, swallowing exceptions and breaking
   back-pressure.
5. **`numpy.float32` in the JSON payload** for `max_val`; depending on
   the encoder, this either silently drops or raises.
6. **No reset.** *Stop → Start* resumed with the previous (DC-soaked)
   field intact.
7. **Front-end draw cost.** A 50 × 50 grid rendered as 2,500 `fillRect`
   calls + per-cell `fillStyle` strings per frame; janky in practice.
8. **No status sync.** The UI's `isRunning` flag was local and
   desynchronised from the backend state.

## 4. The fix

- Ricker wavelet as default source — mean-zero, broadband,
  well-localised in time. (`waveforms.py`)
- `Simulate.reset()` zeros pressure, time, step counter.
  (`simulate.py`)
- CFL is enforced at construction; if the user does not supply
  `timestep`, the engine derives one from a target Courant number.
- `SimulationManager.start()` sets `is_running=True` *synchronously*
  inside an `asyncio.Lock` before scheduling the loop — eliminates the
  race. Auto-start on connect was removed.
- `_run_loop` `await`s every emit, so exceptions surface and
  back-pressure is honoured.
- All numpy scalars are cast to native Python types before emission.
- Backend broadcasts a `status` event after every state change; the
  frontend mirrors that into its `isRunning`, button-disable state, and
  engine-info readout. The UI is no longer the source of truth.
- The canvas renderer fills a per-pixel `ImageData` buffer at the grid
  resolution and blits it through an offscreen canvas with
  `imageSmoothingEnabled = false`. The render loop is decoupled from
  the wire cadence via `requestAnimationFrame`, so fast clients render
  at refresh rate even when the backend ticks slower.

## 5. Running it

From the project root, in two terminals:

```bash
# Terminal 1 — backend
python scripts/run_ui_server.py

# Terminal 2 — frontend
cd frontend
npm install   # first time only
npm run dev
```

Open `http://127.0.0.1:3000`. The header shows connection state, the
backend's run state (idle/running), the current step, simulated time,
the field's current peak amplitude, and the renderer's frames-per-second.
The footer shows the engine's geometry, timestep, wavespeed, gridstep,
and Courant number — useful for confirming CFL compliance at a glance.

## 6. Reading the visualisation

The canvas uses a **diverging blue → white → red** colormap normalised
per-frame to the field's current peak amplitude. Concretely:

- Deep red regions are the highest positive pressure cells in the frame,
- Deep blue regions are the most negative,
- Off-white is the rest field.

Because the colour scale is renormalised every frame, a quiet field
remains visible (not all white). The numeric `max|p|` readout in the
header gives the absolute scale.

A correctly-running default scene shows a Ricker pulse fire at the
centre of the grid at $t \approx 20$, then a circular wavefront radiates
outward at $c = 1$ cell per time unit, reflecting off the hard walls and
forming an interference pattern. Concentric red/blue ring structure with
$\langle p \rangle \approx 0$ is a sanity check that the source is
mean-zero and the integrator is stable.
