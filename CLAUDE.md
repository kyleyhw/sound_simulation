# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

N-dimensional FDTD simulation of acoustic wave propagation, written to generate acoustic datasets for ML echolocation / room-mapping models. The longer-term goal is a closed-loop acoustic control system (sense → infer → beamform); see `PROJECT_PLAN.md` (ten sequential phases; §4 lists the execution rules).

`CURRENT_STATE.md` is the live status note — read it for the latest "what works, what's blocked" picture before diagnosing anything.

## Layout

The Python package is installed under `src/acoustic_system/`:

- `simulation/` — the standalone FDTD engine (`Simulate`, kernel, waveforms, I/O, plotting). Dimension-agnostic, but the 2D path is hot.
- `learning/` — sensing datasets, models, training, calibration, metrics.
- `control/` — sound-field control (Phase 7): `transfer` (FDTD transfer functions), `beamforming` (DAS, pressure matching, ACC, time reversal, FIR), `ctc`, `anc` (FxLMS), `requirements` (control vs sensing error), `differentiable` (TorchFDTD optimisation).
- `imaging/` — physics-based (no-ML) room imaging, the Phase 6 baselines: deconvolution, echo ellipses and carving, back-projection, time reversal, FWI, CRLB, room parameters.
- `utils/` — shared helpers (`room_ir`: IR analysis for real captures).

`web/` is the browser app (Vite + React + TypeScript): the FDTD engine ported to TypeScript and run client-side, deployed to GitHub Pages. It replaced the former `frontend/` + FastAPI/Socket.IO server (removed in plan 4.5.4). See `docs/web_app.md`. `tests/perf/` contains the correctness gate and benchmark used by the evolve harness.

## Commands

### Web app

```
cd web && npm install && npm run dev   # vite on 127.0.0.1:3000
npm test                               # Vitest unit tests (incl. Python-parity fixtures)
npm run e2e                            # Playwright end-to-end (PW_CHROMIUM=/opt/pw-browsers/chromium in the cloud container)
```

The TypeScript engine's fast path must keep matching the Python engine: regenerate fixtures with `uv run python scripts/make_web_fixtures.py` whenever the Python kernel's numerics change.

### Standalone batch simulation

```
cd src && python -m acoustic_system.simulation.main
```

Renders a 256×256 / 500-step run with a Ricker pulse and a tone plus three sensors, then shows the sensor timeseries, FFT and field animation (`--save` writes PNGs headlessly).

### Performance / correctness gates on the FDTD kernel

```
python tests/perf/check_simulate.py     # field equality vs reference.npz (atol 1e-5, rtol 1e-4); also asserts reset() and the CFL RuntimeWarning
python tests/perf/bench_simulate.py --grid 512 --steps 1000 --trials 5
python tests/perf/make_reference.py     # ONLY on the protected baseline — regenerates the snapshot candidates must match
python tests/perf/check_simulate_gpu.py # GPU backend vs CPU backend end-state equality (needs --extra gpu + CUDA device)
python tests/perf/bench_simulate_gpu.py --steps 500 --trials 5   # CPU-vs-GPU speedup sweep
```

`tests/perf/reference.npz` is the truth that candidate kernel implementations must reproduce. Do not regenerate it from a modified branch.

### Environment

Managed by `uv` with `pyproject.toml` + `uv.lock`. To bootstrap:

```
uv sync --extra dev
```

Optional extras: `--extra ml` (torch/torchaudio for Phase 2 training), `--extra gpu` (cupy-cuda12x for the `backend="gpu"` FDTD path; needs an NVIDIA driver ≥ 525).

This creates `.venv/` and installs the project editable, so `import acoustic_system` works from anywhere. Run commands inside the env with `uv run <cmd>` (e.g. `uv run python tests/perf/check_simulate.py`).

Python `>=3.10`. Runtime deps: numpy, scipy, numba (required by the 2D FDTD kernel), h5py, matplotlib, tqdm. Dev toolchain (via `--extra dev`): ruff, ty, detect-secrets, pre-commit. Exact versions are pinned in `uv.lock`.

The legacy `environment.yml` is gone — do not re-introduce a conda workflow. If you're tempted to `pip install` something, add it to `pyproject.toml` and run `uv sync` instead.

## Architecture

### FDTD engine (`Simulate`)

A stateful, step-at-a-time leap-frog solver of

$$ \frac{\partial^2 p}{\partial t^2} = c^2 \nabla^2 p $$

discretised as

$$ p^{n+1} = 2 p^n - p^{n-1} + (c \Delta t)^2 \, \nabla^2 p^n $$

with the standard central second-order stencil. `Simulate` owns `p`, `p_prev`, time, step counter, the driver list, and an interior `obstacle_mask`. Mutating methods: `step()`, `reset()`, `add_driver/remove_driver/set_drivers`, `set_obstacle/clear_obstacles`. The step-at-a-time design is what makes the interactive UI possible — the loop can pause, resume, mutate geometry, and reconfigure between steps.

Interior obstacles are a boolean mask of the same shape as the field. Between the stencil pass and driver injection, `step()` zeroes `p_next` at the masked cells (Dirichlet $p = 0$: a pressure-release, i.e. acoustically *soft*, wall with $\Gamma = -1$ — not a rigid wall, which is Neumann $\partial p/\partial n = 0$ with $\Gamma = +1$). The hot path is guarded by `_has_obstacles`, so a `Simulate` with no obstacles is bit-identical to the pre-feature numerics and `check_simulate.py` keeps matching `reference.npz`.

Two code paths, dispatched once at construction by `self.dims`:

- **2D / 3D fast path**: a numba `@njit(parallel=True, fastmath=True)` fused kernel — `fused_leapfrog_step_2d` (5-point stencil) or `fused_leapfrog_step_3d` (7-point stencil) — fuses the Laplacian, the leap-frog combine, and Dirichlet face-zeroing into one pass. The outer `i` loop is parallelised with `prange`; the innermost loop is unit-stride for SIMD. Per-step measured wall clock on the bench machine: 0.05 ms at 2D 256², 0.11 ms at 2D 512², 0.04 ms at 3D 32³, 0.21 ms at 3D 64³, 0.95 ms at 3D 100³, 2.1 ms at 3D 128³, 7.0 ms at 3D 200³. 3D costs ~2× per cell vs 2D for the same cell count (extra reads + worse cache reuse along the slowest axis).
- **1D fallback**: `scipy.ndimage.laplace` divided by `gridstep**2`, then the leap-frog combine in NumPy, then `set_edge_values(arr, 0)` to enforce hard walls. Kept for behavioural compatibility with anyone building a 1D `Simulate`.

Both paths inject driver values **after** boundary zeroing — a driver placed on a wall intentionally overwrites the zero. Do not reorder this.

The 2D path also uses a three-buffer rotation (`p_prev`, `p`, `_p_next`) to avoid heap allocations per step, a pre-bound kernel reference / cached coefficient (`self._kernel`, `self._coeff`), and a single-driver fast path that precomputes the tuple index. The fast-path cache is refreshed on every driver mutation (not just at construction) via `_refresh_driver_cache`, so a live `add_driver` / `remove_driver` sequence that ends with exactly one driver still hits the precomputed indexed write. The rationale is documented inline in `simulate.py` and `calculate.py` — preserve it if you touch the hot loop, and re-run `check_simulate.py` and `bench_simulate.py` afterwards.

### CFL stability

The leap-frog scheme is stable iff $\sigma \equiv c \Delta t / \Delta x \le 1/\sqrt{d}$ in $d$ dimensions. The constructor enforces this in two modes:

- `timestep=None` (default): pick $\Delta t = \kappa \, \Delta x / c$ where $\kappa = \min(\texttt{courant}, 0.95 / \sqrt{d})$. The default `courant` is 0.5.
- `timestep=<value>`: caller-provided. The constructor emits a `RuntimeWarning` whose message contains `"CFL"` when violated. `check_simulate.py` asserts this warning is fired.

### Thread cap

`simulation/calculate.py` calls `numba.set_num_threads(max(min(cpu_count - 3, 13), 4) if cpu_count >= 8 else cpu_count)` for the fused kernels (the count is per calling thread — a `Simulate` stepped from a worker thread uses numba's default unless that thread sets it). This is an empirically tuned sweet spot on a 16-logical-core machine; above ~13 the kernel oversubscribes the memory bus. The cap is set **twice** (before and after kernel registration) because the first `@njit(parallel=True)` decoration lazily initialises numba's threading runtime and the pre-init `set_num_threads` call can be reset. Users can override at runtime with `numba.set_num_threads(n)` after import.

### Waveforms

`simulation/waveforms.py` exposes `Cosine`, `GaussianPulse`, `RickerWavelet`, all registered in `waveform_registry`. The default UI source is `RickerWavelet` (mean-zero, broadband, finite duration) — the standard choice for FDTD.

Watch out for sampling errors on `Cosine`: the source must satisfy $f \cdot \Delta t < 0.5$ (Nyquist), ideally $< 0.1$ (≥10 samples/period). A previous bug used `Cosine(frequency=20, timestep=0.1)`, which evaluates to $\cos(4\pi n) = 1$ for every integer $n$ — the "oscillator" injected DC and broke the UI. See `docs/web_ui.md` and `docs/waveforms.md`.

`Driver.waveform` takes an instance, not a class.

### Boundary conditions

The fast path is Dirichlet $p = 0$ (pressure-release) only. Its wall enforcement lives in the fused 2D/3D kernels and in `set_edge_values` (1D/N-D fallback). Any other setting routes `step()` through the general path (`simulation/physics.py`), which covers:

- rigid walls and impedance materials, using a per-cell material map;
- the outer boundary kinds `soft | rigid | absorb | mur | sponge | cpml`, set for all faces or per face;
- a `c(x)` speed map.

The CPML (`simulation/cpml.py`) is the only boundary below −40 dB at every angle. Its layer update masks faces that touch rigid or impedance cells, the same way the kernel's Laplacian does. The browser engine mirrors all of this (`web/src/engine/`), and the parity fixtures come from `scripts/make_web_fixtures.py`. Verification: `scripts/verify_physics.py`, reported in `docs/physics.md`.

### Web app (`web/`)

Architecture, UX spec and tests: `docs/web_app.md`. Invariants:

- The zustand store's `scene` is the single source of truth for geometry and sources; the `Runtime` mirrors it into the live `Simulation`. Mutations go through store actions (they snapshot for undo/redo).
- The engine's fast path (p = 0 walls, uniform c) is the exact Python-parity path, gated by `web/tests/unit/parity.test.ts`. The general path (rigid, impedance, absorbing layer, c(x)) uses precomputed per-cell coefficients; keep the step loop allocation-free.
- Driver injection happens after wall zeroing (soft source, `+=`), exactly as in Python.
- Every feature has a Playwright test in `web/tests/e2e/`; tests read engine state via `window.__app`.

### Data I/O

`simulation/data_io.py` reads/writes HDF5 archives via `h5py`. Two save types: `full_history` (whole pressure field over time, gzip compressed) and `sensor_results` (per-sensor timeseries only). The reader registry keys on the file-level `save_type` attribute, so this attribute must be set on every archive written.

## Evolve harness

`tests/perf/check_simulate.py` and `bench_simulate.py` exist because the 2D FDTD kernel has been the target of an evolutionary optimisation run (see the per-round attribution in `simulation/calculate.py`'s docstring and the `evolve/fdtd-runtime/round-*` branch lineage). Any change to the 2D kernel must:

1. Keep `check_simulate.py` green within `atol=1e-5`, `rtol=1e-4`.
2. Be benchmarked with `bench_simulate.py` (median of ≥5 trials), not ad-hoc timing — the first trial absorbs JIT/page-fault cost.

The 3D fused kernel has parallel scripts: `tests/perf/check_simulate_3d.py` (asserts the fused 7-point output matches a `scipy.ndimage.laplace` reference within `atol=1e-4`, `rtol=1e-3` — looser than 2D because the reference is computed inline rather than restored from a stored snapshot, so the fastmath FMA error budget compounds across 100 steps) and `tests/perf/bench_simulate_3d.py` (sweep `--grids 32 64 100 128`, opt into `--include-large` for the slow `200^3` case).

The kernel's public attribute surface (listed in `REQUIRED_ATTRS` in `check_simulate.py`) is part of the contract: `p`, `p_prev`, `time`, `step_count`, `grid_shape`, `timestep`, `wavespeed`, `gridstep`.

## Conventions

- Per-script documentation lives in `docs/<script>.md`; the index is `docs/index.md`. Update both when adding or renaming a module.
- `CURRENT_STATE.md` is the rolling status note. Update it when the "what works / what's blocked" picture shifts.
- `PROJECT_PLAN.md` uses the phased `[pending|in-progress|completed]` format from the user's global CLAUDE.md.
- The repo uses the uv / ruff / ty workflow from the global instructions. `pyproject.toml` declares deps + dev toolchain; `uv.lock` pins; pre-commit runs ruff, ruff-format, detect-secrets, and `uv run ty check` on every commit. `pre-commit run --all-files` is the way to apply the formatter sweep across the codebase.
