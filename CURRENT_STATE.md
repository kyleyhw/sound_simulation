# Project Status as of 2026-05-05

## 1. Resolution

The web UI (Phase 1, Task 1.3) is functional end-to-end. Wave
propagation is visible on the canvas; Start, Stop, and Reset behave as
expected; backend and frontend states stay in sync via a `status`
event channel.

## 2. Root causes that had blocked it

The previous status note (2025-11-25) reported that the UI loaded but
no simulation ran. The actual blockers, all now fixed, were:

1. **DC source bug.** Default `Cosine(frequency=20)` with
   `timestep=0.1` evaluated to $\cos(4 \pi n) = 1$ for every integer
   step $n$. The "oscillator" was a constant DC injection; the field
   grew monotonically with no propagation.
2. **Start race.** `is_running` was set inside the broadcast
   coroutine after the first `await`, so duplicated `start_simulation`
   events (auto-start on connect + manual click) spawned multiple
   parallel step loops on the same field.
3. **Orphan emits.** `asyncio.create_task(sio.emit(...))` swallowed
   exceptions and dropped back-pressure.
4. **No reset.** Stop → Start resumed with the previous field intact.
5. **`numpy.float32` payload.** `max_val` was sent as a numpy scalar,
   not a native Python float.
6. **Slow renderer.** Per-cell `fillRect` + `fillStyle` strings.
7. **Local-only running flag.** UI state desynced from backend.

See `docs/web_ui.md` for the full breakdown and the architectural fix.

## 3. What changed

| layer | change |
| ----- | ------ |
| `simulation/simulate.py` | added `reset()`, CFL stability check, automatic `timestep` derivation when one is not supplied |
| `simulation/waveforms.py` | added `RickerWavelet` (mean-zero, broadband, the new default UI source) |
| `simulation/setup.py` | `Driver.waveform` now takes an instance, not a bare class |
| `app/main.py` | rewrote `SimulationManager` with an `asyncio.Lock`, awaited emits, status broadcasts, no auto-start, native-Python type casting |
| `scripts/run_ui_server.py` | tidier startup, configurable host/port via env |
| `frontend/src/App.tsx` | lazy-init socket, `ImageData` blit renderer, decoupled rAF render loop, status-driven button state, live engine readout |
| `frontend/vite.config.ts` | added `changeOrigin`, pinned host/port |

## 4. Next steps

The interactive UI is ready for the next deliverable batch in
`PROJECT_PLAN.md`: enabling click-to-place obstacles and drivers
(Task 1.3.2 / 1.4) and runtime configuration of waveform / grid
parameters from the UI (Task 1.4.2). The wire protocol already
supports `update_config`; the only missing piece is the corresponding
UI controls.
