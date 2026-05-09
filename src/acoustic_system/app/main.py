"""FastAPI + Socket.IO server hosting an interactive FDTD acoustic simulation.

Architecture
------------
- A single ``SimulationManager`` owns the ``Simulate`` engine, the running
  asyncio task, and the broadcast cadence.
- The manager exposes async ``start``, ``stop``, ``reset`` and ``configure``
  methods. ``start`` synchronously sets ``is_running`` before scheduling the
  background coroutine, eliminating the double-start race that previously
  let multiple step-loops run concurrently against the same field.
- The simulation loop awaits each emit (no fire-and-forget tasks) so
  exceptions surface and back-pressure is honoured.
- All grid payloads are downsampled and cast to native Python floats before
  emission so socket.io's default JSON encoder can serialise them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from acoustic_system.simulation.setup import Driver
from acoustic_system.simulation.simulate import Simulate
from acoustic_system.simulation.waveforms import (
    RickerWavelet,
    waveform_registry,
)

logger = logging.getLogger("acoustic_system")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


DEFAULT_CONFIG: dict[str, Any] = {
    "grid_shape": [200, 200],
    "wavespeed": 1.0,
    "gridstep": 1.0,
    # Auto-derived from CFL when ``timestep`` is None; see Simulate.
    "timestep": None,
    "courant": 0.5,
    "driver_position": [100, 100],
    "waveform": {
        "type": "RickerWavelet",
        "amplitude": 5.0,
        "frequency": 0.1,
        "delay": 20.0,
    },
    # Visualisation cadence
    "downsample": 2,
    "broadcast_hz": 30.0,
}


def build_waveform(spec: dict[str, Any]):
    """Instantiate a Waveform from a {type, ...kwargs} dict.

    Unknown types fall back to a Ricker wavelet so the UI stays demo-able.
    """
    wf_type = spec.get("type", "RickerWavelet")
    cls = waveform_registry.get(wf_type, RickerWavelet)
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    try:
        return cls(**kwargs)
    except TypeError:
        logger.warning("Bad waveform kwargs for %s: %s — falling back to defaults", wf_type, kwargs)
        return cls()


class SimulationManager:
    """Encapsulates the FDTD engine and its broadcast loop."""

    def __init__(self, sio: socketio.AsyncServer) -> None:
        self.sio = sio
        self.config: dict[str, Any] = dict(DEFAULT_CONFIG)
        self.simulation: Simulate | None = None
        self.task: asyncio.Task | None = None
        self.is_running: bool = False
        self._lock = asyncio.Lock()
        self._build_simulation()

    # ----- Configuration ------------------------------------------------- #

    def _build_simulation(self) -> None:
        cfg = self.config
        grid_shape = tuple(cfg["grid_shape"])
        waveform = build_waveform(cfg["waveform"])

        position = tuple(cfg["driver_position"])
        position = tuple(
            int(np.clip(p, 1, s - 2)) for p, s in zip(position, grid_shape, strict=True)
        )
        driver = Driver(position=position, waveform=waveform)

        self.simulation = Simulate(
            grid_shape=grid_shape,
            drivers=[driver],
            wavespeed=cfg["wavespeed"],
            timestep=cfg["timestep"],
            gridstep=cfg["gridstep"],
            courant=cfg["courant"],
        )
        logger.info(
            "Built simulation: shape=%s, dt=%.4g, c=%.4g, dx=%.4g, driver@%s, waveform=%s",
            grid_shape,
            self.simulation.timestep,
            self.simulation.wavespeed,
            self.simulation.gridstep,
            position,
            cfg["waveform"],
        )

    async def configure(self, partial: dict[str, Any]) -> None:
        """Merge user-provided fields into the config and rebuild the engine.

        Stops any running loop first so the rebuild is safe.
        """
        async with self._lock:
            await self._stop_locked()
            for key, value in partial.items():
                if key in self.config:
                    if key == "waveform" and isinstance(value, dict):
                        self.config["waveform"] = {**self.config["waveform"], **value}
                    else:
                        self.config[key] = value
            self._build_simulation()
        await self._broadcast_status()

    # ----- Lifecycle ----------------------------------------------------- #

    async def start(self) -> None:
        """Begin (or resume) the broadcast loop."""
        async with self._lock:
            if self.is_running:
                return
            if self.simulation is None:
                self._build_simulation()
            # Set the flag *before* scheduling so subsequent start() calls
            # observe is_running=True even if the coroutine has not begun.
            self.is_running = True
            self.task = asyncio.create_task(self._run_loop())
        await self._broadcast_status()

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()
        await self._broadcast_status()

    async def _stop_locked(self) -> None:
        """Stop the loop assuming the lock is already held."""
        self.is_running = False
        task, self.task = self.task, None
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except TimeoutError:
                task.cancel()

    async def reset(self) -> None:
        async with self._lock:
            await self._stop_locked()
            if self.simulation is not None:
                self.simulation.reset()
        await self._broadcast_state()
        await self._broadcast_status()

    # ----- Main loop ----------------------------------------------------- #

    async def _run_loop(self) -> None:
        assert self.simulation is not None
        sim = self.simulation
        broadcast_period = 1.0 / float(self.config.get("broadcast_hz", 30.0))
        try:
            while self.is_running:
                # Run several physics steps per broadcast to keep wall-clock
                # cadence reasonable even when broadcast_hz is low.
                steps_per_frame = max(1, int(broadcast_period / max(sim.timestep, 1e-9) / 4))
                for _ in range(steps_per_frame):
                    sim.step()

                await self._broadcast_state()
                await asyncio.sleep(broadcast_period)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Simulation loop crashed")
            self.is_running = False
            await self._broadcast_status()
            raise

    # ----- Outbound messages -------------------------------------------- #

    async def _broadcast_state(self) -> None:
        if self.simulation is None:
            return
        sim = self.simulation
        downsample = max(1, int(self.config.get("downsample", 2)))
        slicing = (slice(None, None, downsample),) * sim.dims
        view = sim.p[slicing]
        max_val = float(np.max(np.abs(view))) if view.size else 0.0
        await self.sio.emit(
            "simulation_update",
            {
                "grid": view.tolist(),
                "max_val": max_val,
                "step": int(sim.step_count),
                "time": float(sim.time),
            },
        )

    async def _broadcast_status(self) -> None:
        await self.sio.emit(
            "status",
            {
                "is_running": bool(self.is_running),
                "config": self._safe_config(),
                "engine": self._engine_info(),
            },
        )

    def _safe_config(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.config.items()}

    def _engine_info(self) -> dict[str, Any]:
        if self.simulation is None:
            return {}
        sim = self.simulation
        courant = sim.wavespeed * sim.timestep / sim.gridstep
        return {
            "grid_shape": list(sim.grid_shape),
            "timestep": float(sim.timestep),
            "wavespeed": float(sim.wavespeed),
            "gridstep": float(sim.gridstep),
            "courant": float(courant),
            "downsample": int(self.config.get("downsample", 2)),
        }


# --- ASGI app ------------------------------------------------------------ #

app = FastAPI(title="Acoustic Simulation Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
sim_manager = SimulationManager(sio)


@app.get("/")
async def read_root():
    return {
        "message": "Acoustic Simulation Server is running",
        "engine": sim_manager._engine_info(),
    }


@app.get("/healthz")
async def healthz():
    return {"ok": True, "is_running": sim_manager.is_running}


# --- Socket.IO handlers --------------------------------------------------- #


@sio.event
async def connect(sid, environ):
    logger.info("Socket connected: %s", sid)
    # Send current state to the new client; do NOT auto-start.
    await sim_manager._broadcast_status()
    await sim_manager._broadcast_state()


@sio.event
async def disconnect(sid):
    logger.info("Socket disconnected: %s", sid)
    # Keep the simulation running for any other connected clients; only stop
    # if this was the last client. python-socketio doesn't expose a count
    # cheaply, so we leave it running and let an explicit stop request kill it.


@sio.event
async def start_simulation(sid, data=None):
    logger.info("start_simulation from %s", sid)
    await sim_manager.start()


@sio.event
async def stop_simulation(sid, data=None):
    logger.info("stop_simulation from %s", sid)
    await sim_manager.stop()


@sio.event
async def reset_simulation(sid, data=None):
    logger.info("reset_simulation from %s", sid)
    await sim_manager.reset()


@sio.event
async def update_config(sid, config):
    logger.info("update_config from %s: %s", sid, config)
    await sim_manager.configure(config or {})


@sio.event
async def request_status(sid, data=None):
    await sim_manager._broadcast_status()
