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
| `update_config`    | `{grid_shape, driver_position, waveform: {type, ...}, ...}` | structural rebuild — discards field, drivers, obstacles |
| `request_status`   | `{}`                             | re-emit current `status` |
| `set_obstacle`     | `{positions: [[i, j], ...], value: bool}` | batched obstacle paint/erase |
| `clear_obstacles`  | `{}`                             | drop every obstacle (field is preserved) |
| `add_driver`       | `{position: [i, j], waveform: {type, ...}}` | append a driver at runtime |
| `remove_driver`    | `{index: int}`                   | drop the driver at that list index |
| `clear_drivers`    | `{}`                             | empty the driver list |

The `set_obstacle` payload is batched on the frontend: a brush stroke
generates one socket event per `~33 ms` flush window with all the cells
touched during that window, rather than one event per cell. This keeps
the wire and the asyncio lock-contention bounded by the drag rate, not
by the brush size.

Server → client events:

| event               | 2D payload | 3D payload |
| ------------------- | ---------- | ---------- |
| `simulation_update` | `{ dims: 2, shape: [Nx, Ny], grid: number[][], max_val, step, time }` | `{ dims: 3, shape: [Nx, Ny, Nz], data: bytes(Nx*Ny*Nz uint8), max_val, step, time }` |
| `status`            | `{ is_running, engine: {dims:2, ...}, config, obstacles: {dims:2, shape, downsample, mask: number[][]}, drivers }` | `{ is_running, engine: {dims:3, ...}, config, obstacles: {dims:3, shape, downsample, mask: bytes}, drivers }` |

Both directions of the schema are tagged with `dims`, so the frontend can
dispatch the renderer (canvas vs Three.js) on a single field rather than
inferring it from `engine.grid_shape.length`. Geometry (`obstacles`,
`drivers`) is broadcast on the `status` channel because it changes at
human pace, not per frame.

### 3D binary encoding

For 3D the pressure field is quantised to `uint8` with the convention
$0 \mapsto -p_{\max}$, $128 \mapsto 0$, $255 \mapsto +p_{\max}$:

$$
b = \operatorname{round}\!\bigl(127.5 \cdot (\operatorname{clip}(p / p_{\max}, -1, 1) + 1)\bigr).
$$

The receiver inverts this in the volume-shader as
$p_{\text{norm}} = 2 b / 255 - 1$ before applying the diverging
colormap. 256 levels is more than the eye can resolve in a transparent
volume; the quantisation noise floor is $p_{\max}/128 \approx 0.78\%$
of peak amplitude, well below the per-frame normalisation jitter.
The bytes ship via socket.io's binary-attachment mechanism (the
`bytes` value in the payload dict is detected automatically by
python-socketio and arrives at the client as an `ArrayBuffer`), so no
JSON expansion happens — wire size is exactly `Nx * Ny * Nz` bytes
plus the small JSON envelope.

Wire-cost reference at common downsampled sizes:

| downsampled grid | bytes / frame | @ 30 FPS |
| ---------------- | ------------- | -------- |
| $32^3$ | 32 KB | 1.0 MB/s |
| $64^3$ | 262 KB | 7.7 MB/s |
| $100^3$ | 1.0 MB | 30 MB/s |
| $128^3$ | 2.1 MB | 60 MB/s |

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

## 7. Interactive editing

The frontend has five interaction modes, switched by buttons in the
side panel:

| mode             | click behaviour                                                    |
| ---------------- | ------------------------------------------------------------------ |
| `view`           | inert; no canvas mutations                                         |
| `draw-obstacle`  | press-and-drag paints a disc of cells per cursor position          |
| `erase-obstacle` | press-and-drag clears the same disc shape                          |
| `place-driver`   | single click adds a `Driver` at that grid cell using the form's waveform spec |
| `remove-driver`  | single click removes the driver whose marker is closest to the click (within ~12 px) |

The brush radius is configurable per session. The disc is computed
client-side as the set of grid cells $(i, j)$ with
$(i - i_0)^2 + (j - j_0)^2 \le r^2$, clipped to grid bounds. Cells are
de-duplicated using a `Set<string>` keyed by `"i,j"` so a slow drag
across the same area still only emits each cell once per flush.

The parameter panel sends `update_config` when *Apply* is clicked. This
is a **structural** rebuild — the backend stops the loop, replaces the
simulation with one built from the new config, and emits a fresh
`status`. The new simulation has a single driver from the form's
waveform spec at the engine's default position; any drivers or
obstacles the user had placed are discarded. This matches how the
previous version of the UI worked; if you want to change geometry
without losing state, use the obstacle / driver buttons instead.

### Coordinate mapping

The canvas is `CANVAS_PX × CANVAS_PX` (currently 600 px) displaying the
downsampled grid view. A canvas pixel `(cx, cy)` maps to full-grid
indices

$$ i = \left\lfloor \frac{c_y}{\text{CANVAS\_PX}} \cdot H \right\rfloor,
\quad
j = \left\lfloor \frac{c_x}{\text{CANVAS\_PX}} \cdot W \right\rfloor $$

where $(H, W)$ is the engine's full `grid_shape`. The same ratio runs
in reverse when drawing driver markers on top of the field, so a marker
at engine position `(i, j)` sits exactly above the cell the user
clicked on, independent of the current downsample factor.
