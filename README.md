# Sound Wave Propagation

An N-dimensional simulation of sound wave propagation via the Finite-Difference Time-Domain (FDTD) method. Written with the goal of generating acoustic data to train machine learning models for environmental echolocation mapping.

## Highlights

- **N-dimensional FDTD engine** ([`src/acoustic_system/simulation/`](./src/acoustic_system/simulation/)) — the spatial dimension is set by the shape of the pressure grid; 1D, 2D, and 3D simulations all use the same code path. Hard-wall (Dirichlet) boundary conditions are enforced; CFL stability is checked at construction time, and the timestep is auto-derived from a target Courant number when one is not supplied.
- **Numba-JIT'd 2D fast path** — for the dominant 2D use case the inner stencil is fused into a single `@njit(parallel=True, fastmath=True)` kernel; the 1D and 3D paths fall back to the generic `scipy.ndimage.laplace` route. Together with thread-count tuning and per-step Python-overhead reduction, the optimised path runs the reference 512×512 / 1000-step benchmark in roughly 92 ms — about **64× faster** than the unoptimised baseline. See [`docs/simulate.md`](./docs/simulate.md) for the technique stack.
- **Interactive browser-based web UI** ([`src/acoustic_system/app/`](./src/acoustic_system/app/) backend, [`frontend/`](./frontend/) React + canvas frontend) — drives the engine in real time over Socket.IO, renders the live pressure field as a diverging colour map, exposes Start / Stop / Reset, and surfaces the engine's current Courant number and step count for at-a-glance CFL verification. Architecture and wire protocol documented in [`docs/web_ui.md`](./docs/web_ui.md).
- **Standalone batch generator** ([`src/acoustic_system/simulation/main.py`](./src/acoustic_system/simulation/main.py), [`src/acoustic_system/simulation/generate.py`](./src/acoustic_system/simulation/generate.py)) — runs predefined or randomised driver/sensor scenes and writes results to HDF5 for offline ML training.

## Mathematical formulation

The simulator solves the homogeneous acoustic wave equation in $d$ spatial dimensions,

$$
\frac{\partial^2 p}{\partial t^2} \;=\; c^2\, \nabla^2 p,
$$

with $p$ the acoustic pressure and $c$ the wavespeed of the medium. Using central second-order finite differences in both time and space, the explicit *leap-frog* update is

$$
p^{\,n+1}_{\mathbf{i}} \;=\; 2\, p^{\,n}_{\mathbf{i}} - p^{\,n-1}_{\mathbf{i}} + (c\, \Delta t)^2 \, \big( \nabla^2 p^{\,n} \big)_{\mathbf{i}},
$$

where $\mathbf{i}$ indexes the spatial grid and the discrete Laplacian is the standard 3-point central stencil per axis,

$$
\big(\nabla^2 p\big)_{\mathbf{i}} \;\approx\; \sum_{k=1}^{d}\, \frac{p_{\mathbf{i}+\mathbf{e}_k} + p_{\mathbf{i}-\mathbf{e}_k} - 2\,p_{\mathbf{i}}}{(\Delta x)^2}.
$$

For the explicit leap-frog scheme to be numerically stable, the Courant number must satisfy the CFL condition

$$
C \;\equiv\; \frac{c\, \Delta t}{\Delta x} \;\le\; \frac{1}{\sqrt{d}}.
$$

The engine warns when an explicit timestep violates this bound and, if no timestep is supplied, derives one automatically from a configurable target Courant number (default $C = 0.5$ — comfortably stable in 2D, where the limit is $1/\sqrt{2} \approx 0.707$).

## Project layout

```
sound_simulation/
├── src/
│   └── acoustic_system/
│       ├── app/                    # Web UI backend (FastAPI + Socket.IO)
│       │   └── main.py
│       ├── simulation/             # FDTD engine
│       │   ├── simulate.py         # Simulate class — step-at-a-time time stepper
│       │   ├── calculate.py        # 2D fast-path JIT kernel + scipy fallback
│       │   ├── waveforms.py        # Cosine, GaussianPulse, RickerWavelet
│       │   ├── setup.py            # Driver / Sensor dataclasses
│       │   ├── boundary.py         # Boundary scaffold (currently hard-wall)
│       │   ├── utils.py            # Edge-index helpers
│       │   ├── data_io.py          # HDF5 save / load
│       │   ├── visualize.py        # Offline plots / animations
│       │   ├── interactive_setup.py# Legacy matplotlib scene editor
│       │   ├── main.py             # Standalone batch run
│       │   └── generate.py         # Randomised scene generator for ML data
│       ├── control/                # (placeholder) beamforming, future Phase 3
│       ├── learning/               # (placeholder) inference models, future Phase 2
│       └── utils/                  # (placeholder) cross-cutting helpers
├── frontend/                       # React + Vite + Canvas UI
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   └── vite.config.ts
├── scripts/
│   └── run_ui_server.py            # Launches the web-UI backend on :8001
├── docs/                           # See documentation index below
├── data/                           # Generated outputs (HDF5, plots) — gitignored
├── configs/                        # Run configurations
├── notebooks/                      # Exploration notebooks
├── pyproject.toml                  # uv-managed Python deps
├── uv.lock
├── .pre-commit-config.yaml
├── PROJECT_PLAN.md
└── README.md
```

## Quickstart

### Backend / engine (Python)

This project uses [**uv**](https://github.com/astral-sh/uv) for Python dependency management. Install uv once (see its README), then from the project root:

```bash
uv sync                        # creates .venv/, installs runtime + dev deps
uv run pre-commit install      # installs the local git hooks
```

`uv sync` reads `pyproject.toml` and `uv.lock` and produces a reproducible environment (Python ≥ 3.11). Subsequent commands either use `uv run <cmd>` or activate the environment directly (`.venv/Scripts/activate` on Windows, `.venv/bin/activate` elsewhere).

> **Migrating from the legacy conda environment**
> Earlier versions of the project used `environment.yml` with conda. That file is retained in the repository for reference but should not be used for new installs — `uv sync` is the supported path. The two environments resolve the same set of runtime libraries.

### Web UI (interactive simulation in the browser)

In two terminals from the project root:

```bash
# Terminal 1 — Python backend (FastAPI + Socket.IO on 127.0.0.1:8001)
uv run python scripts/run_ui_server.py

# Terminal 2 — Vite dev server (React + Canvas on 127.0.0.1:3000)
cd frontend
npm install            # first time only
npm run dev
```

Open http://127.0.0.1:3000. The header shows connection state, run state, simulated time, current $\max|p|$, and the renderer's frame rate; the footer shows engine geometry and the Courant number. Click *Start* to launch a Ricker pulse from the centre and watch the wavefront radiate.

### Standalone batch run

For ML data generation or offline analysis, the legacy script-style entry points still work:

```bash
uv run python -m acoustic_system.simulation.main      # one predefined run, with plots
uv run python -m acoustic_system.simulation.generate  # randomised dataset → HDF5
```

## Documentation

The [`docs/`](./docs/) directory contains the long-form documentation. Start at [`docs/index.md`](./docs/index.md). Highlights:

- [`docs/web_ui.md`](./docs/web_ui.md) — Architecture, wire protocol, and operation of the interactive browser UI.
- [`docs/simulate.md`](./docs/simulate.md) — Mathematical derivation, leap-frog stability, and the 2D fast-path implementation.
- [`docs/waveforms.md`](./docs/waveforms.md) — Source waveform catalogue (Cosine, GaussianPulse, RickerWavelet) with sampling guidance.

Per-script documentation for each module in `src/acoustic_system/simulation/` is linked from the index.

## Development

- **Lint / format**: `uv run ruff check` and `uv run ruff format`. Both are wired into the pre-commit config.
- **Type check**: `uv run ty check`. Currently the strictest scope is `src/` and `scripts/`; legacy paths under `interactive_setup.py` are excluded until they are rewritten or removed.
- **Secrets gate**: `uv run detect-secrets scan --baseline .secrets.baseline` re-validates the baseline.
- **Performance**: the benchmark + correctness harness lives on the `evolve/fdtd-runtime/baseline` branch under `tests/perf/`. Switch to that branch and run `python tests/perf/check_simulate.py` (correctness) and `python tests/perf/bench_simulate.py` (timing) for a headline speed measurement.

## Project plan

The phased plan is tracked in [`PROJECT_PLAN.md`](./PROJECT_PLAN.md). At the time of writing, Phase 1 (interactive browser UI) is functional end-to-end; the natural next deliverable is interactive obstacle drawing (Phase 1 Task 1.4) which feeds Phase 2's echolocation work.

## References

<a name="reference-1"></a>[1] Schneider, J. B. (2023). *Chapter 12: Acoustic FDTD Simulation*. In Electrical Engineering, Washington State University. Available: <https://eecs.wsu.edu/~schneidj/ufdtd/ufdtd.pdf>

<a name="reference-2"></a>[2] Kinsler, L. E., Frey, A. R., Coppens, A. B., & Sanders, J. V. (2000). *Fundamentals of Acoustics* (4th ed.). John Wiley & Sons.

<a name="reference-3"></a>[3] Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.
