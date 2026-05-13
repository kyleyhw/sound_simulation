# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

N-dimensional FDTD simulation of acoustic wave propagation, written to generate acoustic datasets for ML echolocation / room-mapping models. The longer-term goal is a closed-loop acoustic control system (sense → infer → beamform); see `PROJECT_PLAN.md`. Current focus is Phase 1: the interactive browser-based simulation UI.

`CURRENT_STATE.md` is the live status note — read it for the latest "what works, what's blocked" picture before diagnosing anything.

## Layout

The Python package is installed under `src/acoustic_system/`:

- `simulation/` — the standalone FDTD engine (`Simulate`, kernel, waveforms, I/O, plotting). Dimension-agnostic, but the 2D path is hot.
- `app/main.py` — FastAPI + Socket.IO server hosting an interactive `SimulationManager` over the engine.
- `control/`, `learning/`, `utils/` — placeholders for later phases (Phase 3 beamformer, Phase 2 inference).

`frontend/` is a Vite + React + TypeScript dev server that talks to the backend over `/socket.io`. `tests/perf/` contains the correctness gate and benchmark used by the evolve harness.

Note: older docs (and the top-level `README.md` "Code Structure" block) refer to a flat `simulation/` and `app/` layout at the repo root. The code has been moved under `src/acoustic_system/`; trust the actual tree, not the legacy references.

## Commands

### Web UI (two terminals, from project root)

```
python scripts/run_ui_server.py             # uvicorn on 127.0.0.1:8001
cd frontend && npm install && npm run dev   # vite on 127.0.0.1:3000
```

Open `http://127.0.0.1:3000`. Vite proxies `/socket.io` (HTTP + WS) to 8001. The backend host/port can be overridden with `ACOUSTIC_HOST` / `ACOUSTIC_PORT`.

### Standalone batch simulation

```
cd src && python -m acoustic_system.simulation.main
```

Renders a 256×256 / 500-step run with two cosine drivers and three sensors, then pops up matplotlib windows for the field, sensor timeseries, and sensor FFT.

### Performance / correctness gates on the FDTD kernel

```
python tests/perf/check_simulate.py     # field equality vs reference.npz (atol 1e-5, rtol 1e-4); also asserts reset() and the CFL RuntimeWarning
python tests/perf/bench_simulate.py --grid 512 --steps 1000 --trials 5
python tests/perf/make_reference.py     # ONLY on the protected baseline — regenerates the snapshot candidates must match
```

`tests/perf/reference.npz` is the truth that candidate kernel implementations must reproduce. Do not regenerate it from a modified branch.

### Environment

Conda, via `environment.yml`:

```
conda env create -f environment.yml && conda activate sound_simulation
```

Python 3.9. Key deps: numpy, scipy, h5py, matplotlib, fastapi, uvicorn, python-socketio. **Numba is required by the 2D FDTD kernel but is NOT in `environment.yml`** — install with `pip install numba` (or add it to the env file) before running anything that calls `Simulate.step()` in 2D.

## Architecture

### FDTD engine (`Simulate`)

A stateful, step-at-a-time leap-frog solver of

$$ \frac{\partial^2 p}{\partial t^2} = c^2 \nabla^2 p $$

discretised as

$$ p^{n+1} = 2 p^n - p^{n-1} + (c \Delta t)^2 \, \nabla^2 p^n $$

with the standard central second-order stencil. `Simulate` owns `p`, `p_prev`, time, step counter, the driver list, and an interior `obstacle_mask`. Mutating methods: `step()`, `reset()`, `add_driver/remove_driver/set_drivers`, `set_obstacle/clear_obstacles`. The step-at-a-time design is what makes the interactive UI possible — the loop can pause, resume, mutate geometry, and reconfigure between steps.

Interior obstacles are a boolean mask of the same shape as the field. Between the stencil pass and driver injection, `step()` zeroes `p_next` at the masked cells (rigid Dirichlet wall, $\Gamma = -1$). The hot path is guarded by `_has_obstacles`, so a `Simulate` with no obstacles is bit-identical to the pre-feature numerics and `check_simulate.py` keeps matching `reference.npz`.

Two code paths:

- **2D hot path**: `fused_leapfrog_step_2d` in `simulation/calculate.py` — a numba `@njit(parallel=True, fastmath=True)` kernel that fuses the 5-point Laplacian, the leap-frog combine, and Dirichlet edge zeroing into a single pass over the interior. Roughly 65–70× faster than the legacy path at 512×512 / 1000 steps.
- **1D / 3D fallback**: `scipy.ndimage.laplace` divided by `gridstep**2`, then the leap-frog combine in NumPy, then `set_edge_values(arr, 0)` to enforce hard walls.

Both paths inject driver values **after** boundary zeroing — a driver placed on a wall intentionally overwrites the zero. Do not reorder this.

The 2D path also uses a three-buffer rotation (`p_prev`, `p`, `_p_next`) to avoid heap allocations per step, a pre-bound kernel reference / cached coefficient (`self._kernel`, `self._coeff`), and a single-driver fast path that precomputes the tuple index. The fast-path cache is refreshed on every driver mutation (not just at construction) via `_refresh_driver_cache`, so a live `add_driver` / `remove_driver` sequence that ends with exactly one driver still hits the precomputed indexed write. The rationale is documented inline in `simulate.py` and `calculate.py` — preserve it if you touch the hot loop, and re-run `check_simulate.py` and `bench_simulate.py` afterwards.

### CFL stability

The leap-frog scheme is stable iff $\sigma \equiv c \Delta t / \Delta x \le 1/\sqrt{d}$ in $d$ dimensions. The constructor enforces this in two modes:

- `timestep=None` (default): pick $\Delta t = \kappa \, \Delta x / c$ where $\kappa = \min(\texttt{courant}, 0.95 / \sqrt{d})$. The default `courant` is 0.5.
- `timestep=<value>`: caller-provided. The constructor emits a `RuntimeWarning` whose message contains `"CFL"` when violated. `check_simulate.py` asserts this warning is fired.

### Thread cap

`simulation/calculate.py` calls `numba.set_num_threads(min(cpu_count - 3, 13))` for the 5-point stencil. This is an empirically tuned sweet spot on a 16-logical-core machine; above ~13 the kernel oversubscribes the memory bus. The cap is set **twice** (before and after kernel registration) because the first `@njit(parallel=True)` decoration lazily initialises numba's threading runtime and the pre-init `set_num_threads` call can be reset. Users can override at runtime with `numba.set_num_threads(n)` after import.

### Waveforms

`simulation/waveforms.py` exposes `Cosine`, `GaussianPulse`, `RickerWavelet`, all registered in `waveform_registry`. The default UI source is `RickerWavelet` (mean-zero, broadband, finite duration) — the standard choice for FDTD.

Watch out for sampling errors on `Cosine`: the source must satisfy $f \cdot \Delta t < 0.5$ (Nyquist), ideally $< 0.1$ (≥10 samples/period). A previous bug used `Cosine(frequency=20, timestep=0.1)`, which evaluates to $\cos(4\pi n) = 1$ for every integer $n$ — the "oscillator" injected DC and broke the UI. See `docs/web_ui.md` and `docs/waveforms.md`.

`Driver.waveform` takes an instance, not a class.

### Boundary conditions

Hard wall (Dirichlet, $p = 0$) only. Absorbing boundaries (PML) are listed under Future Work but not implemented. The wall enforcement lives in the kernel (2D) and in `set_edge_values` (1D/3D); `simulation/boundary.py` is a generic scaffold that is not currently wired in.

### Web UI (`app/main.py` + `frontend/`)

Architecture, wire protocol, and the bugs that previously blocked it are written up in `docs/web_ui.md`. Read it before modifying the manager or the socket handlers.

Invariants that previously broke and must not regress:

- `SimulationManager` holds an `asyncio.Lock` and sets `is_running = True` **synchronously inside the lock** before scheduling the background coroutine. This eliminates a double-start race that previously spawned two parallel step loops on the same field.
- Every emit is `await`ed. No `asyncio.create_task(sio.emit(...))` fire-and-forget — that swallows exceptions and drops back-pressure.
- NumPy scalars are cast to native Python types before emission; socket.io's JSON encoder rejects `numpy.float32`.
- The `connect` handler must NOT auto-start (the combination of auto-start + manual *Start* click was one of the original duplicate-loop bugs).
- Grids are downsampled by `downsample` (default 2) before emission to keep the wire cheap.
- The frontend renders via an `ImageData` buffer at native grid resolution blitted through an offscreen canvas with `imageSmoothingEnabled = false`. The render loop is a `requestAnimationFrame` tick decoupled from wire cadence and reads the latest frame from a ref — so fast clients render at refresh rate even when the backend ticks slower.

Wire protocol (full table in `docs/web_ui.md`):

- Client → server lifecycle: `start_simulation`, `stop_simulation`, `reset_simulation`, `update_config`, `request_status`.
- Client → server geometry: `set_obstacle` (batched cells), `clear_obstacles`, `add_driver`, `remove_driver`, `clear_drivers`.
- Server → client: `simulation_update` (downsampled grid + `step`, `time`, `max_val`), `status` (`is_running`, `engine`, `config`, `obstacles`, `drivers`).

Geometry (`obstacles`, `drivers`) is broadcast on the `status` channel, not the per-frame `simulation_update`, because it changes at human pace. Obstacle brush strokes are batched on the frontend (~33 ms flush window) so one stroke arrives as O(strokes) socket events rather than O(cells).

### Data I/O

`simulation/data_io.py` reads/writes HDF5 archives via `h5py`. Two save types: `full_history` (whole pressure field over time, gzip compressed) and `sensor_results` (per-sensor timeseries only). The reader registry keys on the file-level `save_type` attribute, so this attribute must be set on every archive written.

## Evolve harness

`tests/perf/check_simulate.py` and `bench_simulate.py` exist because the FDTD kernel has been the target of an evolutionary optimisation run (see the per-round attribution in `simulation/calculate.py`'s docstring and the `evolve/fdtd-runtime/round-*` branch lineage). Any change to the kernel must:

1. Keep `check_simulate.py` green within `atol=1e-5`, `rtol=1e-4`.
2. Be benchmarked with `bench_simulate.py` (median of ≥5 trials), not ad-hoc timing — the first trial absorbs JIT/page-fault cost.

The kernel's public attribute surface (listed in `REQUIRED_ATTRS` in `check_simulate.py`) is part of the contract: `p`, `p_prev`, `time`, `step_count`, `grid_shape`, `timestep`, `wavespeed`, `gridstep`.

## Conventions

- Per-script documentation lives in `docs/<script>.md`; the index is `docs/index.md`. Update both when adding or renaming a module.
- `CURRENT_STATE.md` is the rolling status note. Update it when the "what works / what's blocked" picture shifts.
- `PROJECT_PLAN.md` uses the phased `[pending|in-progress|completed]` format from the user's global CLAUDE.md.
- The repo does NOT yet use the uv / ruff / ty workflow from the global instructions; `environment.yml` (conda) is the canonical env. Propose migration before introducing it.
