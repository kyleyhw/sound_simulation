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
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


DEFAULT_CONFIG: Dict[str, Any] = {
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

# Checkpoint used by the `sense_room` event (the Phase 2 sensing recipe:
# joint-trained encoder per pose + Bayes fusion — docs/learning.md).
# Resolved relative to the repo root; override with the env var when
# demonstrating a different model. Inference is CPU and takes well under
# a second, so it runs on demand rather than being preloaded.
SENSE_CHECKPOINT = os.environ.get(
    "ACOUSTIC_SENSE_CHECKPOINT",
    str(Path(__file__).resolve().parents[3] / "checkpoints" / "skip_v2" / "best_iou.pt"),
)


def build_waveform(spec: Dict[str, Any]):
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


def quantize_to_uint8_bytes(view: np.ndarray, max_val: float) -> bytes:
    """Map a float pressure array to ``uint8`` bytes with 128 = zero pressure.

    Encoding
    --------
    Each voxel ``p`` is normalised against the per-frame peak

    $$ p_{\\text{norm}} = \\operatorname{clip}\\!\\left(p / p_{\\max},\\; -1,\\; 1\\right), $$

    and quantised to an 8-bit unsigned integer

    $$ b = \\operatorname{round}\\!\\bigl(127.5 \\cdot (p_{\\text{norm}} + 1)\\bigr), $$

    so 0 maps to the most negative pressure, 128 maps to silence, and
    255 maps to the most positive pressure. The receiver inverts this
    in the fragment shader as ``p_norm = 2 * sample - 1`` before
    feeding the diverging colormap.

    256 levels is comfortably more than the eye can resolve in a
    transparent volume — this is the same reasoning medical-imaging
    pipelines use to ship CT volumes as uint8. The quantisation noise
    floor is $p_{\\max} / 128 \\approx 0.78\\%$ of peak amplitude, well
    below the per-frame normalisation jitter.

    Returns the contiguous C-order byte buffer ready for socket.io's
    binary-attachment serialisation. An empty input returns ``b""``;
    a zero ``max_val`` returns an all-128 (silent) buffer of the
    expected length.
    """
    if view.size == 0:
        return b""
    if max_val <= 0.0:
        return bytes(np.full(view.size, 128, dtype=np.uint8))
    norm = np.clip(view / np.float32(max_val), -1.0, 1.0).astype(np.float32)
    quantised = np.rint((norm + 1.0) * 127.5).astype(np.uint8)
    return quantised.tobytes(order="C")


class SimulationManager:
    """Encapsulates the FDTD engine and its broadcast loop."""

    def __init__(self, sio: socketio.AsyncServer) -> None:
        self.sio = sio
        self.config: Dict[str, Any] = dict(DEFAULT_CONFIG)
        self.simulation: Optional[Simulate] = None
        self.task: Optional[asyncio.Task] = None
        self.is_running: bool = False
        self._lock = asyncio.Lock()
        self._build_simulation()

    # ----- Configuration ------------------------------------------------- #

    def _build_simulation(self) -> None:
        cfg = self.config
        grid_shape = tuple(cfg["grid_shape"])
        waveform = build_waveform(cfg["waveform"])

        raw_position = list(cfg["driver_position"])
        if len(raw_position) != len(grid_shape):
            # Dimensionality switched (typically 2D <-> 3D via update_config).
            # The previous driver_position no longer fits the new grid; fall
            # back to the centre and update the config so subsequent rebuilds
            # start from a consistent state.
            raw_position = [s // 2 for s in grid_shape]
            cfg["driver_position"] = list(raw_position)
        position = tuple(int(np.clip(p, 1, s - 2)) for p, s in zip(raw_position, grid_shape))
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

    async def configure(self, partial: Dict[str, Any]) -> None:
        """Merge user-provided fields into the config and rebuild the engine.

        Stops any running loop first so the rebuild is safe. Structural
        changes (grid shape, wavespeed, gridstep, courant) discard the field
        because the geometry itself has changed; live changes (drivers,
        obstacles) go through the dedicated mutation methods below and do
        not lose state.
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

    # ----- Live geometry mutation --------------------------------------- #

    async def set_obstacle(
        self,
        positions: Any,
        value: bool = True,
    ) -> None:
        """Mark (``value=True``) or clear (``value=False``) a batch of cells.

        Positions are expected as an iterable of ``[i, j]`` lists from the
        wire. We coerce defensively because the JSON encoder happily lets a
        client send strings; the engine ignores anything out-of-bounds.
        """
        coerced: List[Tuple[int, ...]] = []
        try:
            for pos in positions or []:
                if isinstance(pos, (list, tuple)) and len(pos) >= 1:
                    coerced.append(tuple(int(c) for c in pos))
        except (TypeError, ValueError):
            logger.warning("set_obstacle: malformed positions %r", positions)
            return
        if not coerced:
            return
        async with self._lock:
            if self.simulation is None:
                return
            self.simulation.set_obstacle(coerced, bool(value))
        await self._broadcast_status()

    async def clear_obstacles(self) -> None:
        async with self._lock:
            if self.simulation is None:
                return
            self.simulation.clear_obstacles()
        await self._broadcast_status()

    async def add_driver(
        self,
        position: Any,
        waveform_spec: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a driver. Position is clipped one cell inside each wall so
        a click anywhere on the canvas (including the outer ring) still
        lands on a legal interior cell."""
        if self.simulation is None:
            return
        try:
            pos = tuple(int(c) for c in position)
        except (TypeError, ValueError):
            logger.warning("add_driver: malformed position %r", position)
            return
        if len(pos) != len(self.simulation.grid_shape):
            logger.warning(
                "add_driver: position dim %d != grid dim %d",
                len(pos),
                len(self.simulation.grid_shape),
            )
            return
        spec = waveform_spec if isinstance(waveform_spec, dict) else self.config["waveform"]
        waveform = build_waveform(spec)
        clipped = tuple(int(np.clip(c, 1, s - 2)) for c, s in zip(pos, self.simulation.grid_shape))
        async with self._lock:
            self.simulation.add_driver(Driver(position=clipped, waveform=waveform))
        await self._broadcast_status()

    async def remove_driver(self, index: int) -> None:
        try:
            idx = int(index)
        except (TypeError, ValueError):
            return
        async with self._lock:
            if self.simulation is None:
                return
            try:
                self.simulation.remove_driver(idx)
            except IndexError:
                logger.warning("remove_driver: index %d out of range", idx)
                return
        await self._broadcast_status()

    async def clear_drivers(self) -> None:
        async with self._lock:
            if self.simulation is None:
                return
            self.simulation.set_drivers([])
        await self._broadcast_status()

    # ----- Acoustic sensing (Phase 2 bridge) ----------------------------- #

    async def sense(self, n_poses: int = 8, seed: int = 0) -> Dict[str, Any]:
        """Map the current room acoustically and return the fused estimate.

        Snapshot the obstacle mask under the lock, then run the full
        sense -> infer -> fuse pipeline (``learning/sensing.py``) in a
        worker thread so the event loop keeps serving frames — the
        pipeline is synchronous CPU work (K forward FDTD runs + K model
        forwards, ~0.5 s at K=8).

        Returns the payload for the ``sense_result`` event; failures
        (missing torch extra, missing checkpoint, 3D grid) come back as
        ``{"ok": False, "error": ...}`` rather than raising, so the UI
        can display them.
        """
        async with self._lock:
            if self.simulation is None or self.simulation.dims != 2:
                return {"ok": False, "error": "sensing demo supports 2D rooms only"}
            mask = np.asarray(self.simulation.obstacle_mask).copy()

        def _run() -> Dict[str, Any]:
            try:
                from acoustic_system.learning.sensing import load_sensing_model
                from acoustic_system.learning.sensing import sense_room as _sense_room
            except ImportError as exc:
                return {
                    "ok": False,
                    "error": f"ML extra not installed ({exc}); run `uv sync --extra ml`",
                }
            if not Path(SENSE_CHECKPOINT).exists():
                return {
                    "ok": False,
                    "error": f"no checkpoint at {SENSE_CHECKPOINT}; train one (docs/learning.md)",
                }
            t0 = time.perf_counter()
            result = _sense_room(mask, SENSE_CHECKPOINT, n_poses=n_poses, seed=seed)
            _, cfg = load_sensing_model(SENSE_CHECKPOINT)
            final = result.fused_probs[-1]
            return {
                "ok": True,
                "shape": list(final.shape),
                "prob": [[round(float(v), 4) for v in row] for row in final],
                "truth": result.truth.astype(np.uint8).tolist(),
                "ious": [round(float(v), 4) for v in result.ious],
                "poses": int(len(result.ious)),
                # The val-selected decision threshold the IoUs are scored
                # at (Task 2.3e operating point; 0.5 for uncalibrated
                # checkpoints) — shown in the UI caption.
                "threshold": round(float(cfg.threshold), 2),
                "elapsed": round(time.perf_counter() - t0, 2),
            }

        return await asyncio.to_thread(_run)

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
            except asyncio.TimeoutError:
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

        if sim.dims == 3:
            # Binary path: a downsampled 3D pressure field is far too
            # large to JSON-encode as nested floats (a 100^3 view is
            # 1 MB raw, ~10 MB as JSON text). Quantise to uint8 with
            # 128 = silence and ship the contiguous byte buffer via
            # socket.io's binary-attachment mechanism. The receiver
            # uploads it directly into a Three.js Data3DTexture and
            # de-quantises in the volume-rendering fragment shader.
            await self.sio.emit(
                "simulation_update",
                {
                    "dims": 3,
                    "shape": list(view.shape),
                    "max_val": max_val,
                    "step": int(sim.step_count),
                    "time": float(sim.time),
                    "data": quantize_to_uint8_bytes(view, max_val),
                },
            )
        else:
            # 2D / 1D JSON path. Kept verbatim for backward compatibility
            # with the existing canvas frontend; the new ``dims`` and
            # ``shape`` fields are additive and let any future client
            # dispatch on dim without re-checking ``engine.dims``.
            await self.sio.emit(
                "simulation_update",
                {
                    "dims": int(sim.dims),
                    "shape": list(view.shape),
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
                "obstacles": self._obstacle_payload(),
                "drivers": self._driver_payload(),
            },
        )

    def _obstacle_payload(self) -> Dict[str, Any]:
        """Downsampled obstacle mask + dimensions, ready for the wire.

        Geometry is broadcast on every status (not every frame): obstacles
        change at human pace, so this lives outside the hot
        ``simulation_update`` path. The shape is the *downsampled* shape so
        the frontend can blit it directly onto the same renderer it uses
        for the pressure field.

        Wire format depends on dim:
          * 2D — ``mask`` is a nested list of 0/1 ints (current schema).
          * 3D — ``mask`` is a contiguous ``bytes`` buffer of 0 / 1 uint8
            voxels in C order, shape ``Nz * Ny * Nx``. This is the natural
            input for the Three.js volume renderer's mask texture and
            keeps the wire size at one byte per voxel.
        """
        if self.simulation is None:
            return {"shape": [0, 0], "downsample": 1, "dims": 2, "mask": []}
        sim = self.simulation
        downsample = max(1, int(self.config.get("downsample", 2)))
        slicing = (slice(None, None, downsample),) * sim.dims
        view = sim.obstacle_mask[slicing]
        if sim.dims == 3:
            return {
                "shape": list(view.shape),
                "downsample": downsample,
                "dims": 3,
                "mask": np.ascontiguousarray(view, dtype=np.uint8).tobytes(order="C"),
            }
        return {
            "shape": list(view.shape),
            "downsample": downsample,
            "dims": int(sim.dims),
            "mask": view.astype(np.uint8).tolist(),
        }

    def _driver_payload(self) -> List[Dict[str, Any]]:
        """List of {position, waveform: {type, ...kwargs}} for the UI."""
        if self.simulation is None:
            return []
        out: List[Dict[str, Any]] = []
        for driver in self.simulation.drivers:
            wf = driver.waveform
            wf_kwargs = {
                k: (float(v) if isinstance(v, (np.floating, float, int)) else v)
                for k, v in wf.__dict__.items()
            }
            out.append(
                {
                    "position": [int(c) for c in driver.position],
                    "waveform": {"type": wf.__class__.__name__, **wf_kwargs},
                }
            )
        return out

    def _safe_config(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.config.items()}

    def _engine_info(self) -> Dict[str, Any]:
        if self.simulation is None:
            return {}
        sim = self.simulation
        courant = sim.wavespeed * sim.timestep / sim.gridstep
        return {
            "grid_shape": list(sim.grid_shape),
            "dims": int(sim.dims),
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


# --- Live geometry events ------------------------------------------------ #


@sio.event
async def set_obstacle(sid, data):
    """Batched obstacle mutation.

    Payload: ``{"positions": [[i, j], ...], "value": bool}``. Batched on
    the frontend so a brush stroke arrives as one socket event rather than
    one per cell, keeping the wire and the lock-contention low.
    """
    if not isinstance(data, dict):
        return
    await sim_manager.set_obstacle(data.get("positions", []), bool(data.get("value", True)))


@sio.event
async def clear_obstacles(sid, data=None):
    await sim_manager.clear_obstacles()


@sio.event
async def add_driver(sid, data):
    """Payload: ``{"position": [i, j], "waveform": {type, ...kwargs}}``.

    Waveform is optional — if omitted, the manager falls back to the
    current default in ``config['waveform']``.
    """
    if not isinstance(data, dict):
        return
    await sim_manager.add_driver(data.get("position"), data.get("waveform"))


@sio.event
async def remove_driver(sid, data):
    """Payload: ``{"index": int}``."""
    if not isinstance(data, dict):
        return
    await sim_manager.remove_driver(data.get("index"))


@sio.event
async def clear_drivers(sid, data=None):
    await sim_manager.clear_drivers()


# --- Acoustic sensing (Phase 2 bridge) ------------------------------------ #


@sio.event
async def sense_room(sid, data=None):
    """Run the Phase 2 sensing recipe on the currently drawn room.

    Payload (all optional): ``{"poses": int (1-16, default 8),
    "seed": int}``. Replies to the requesting client only with a
    ``sense_result`` event: the Bayes-fused obstacle-probability map,
    the per-K IoU progression against the drawn room, and the runtime.
    Emitted to the requester (not broadcast) because the result answers
    one client's button press, unlike the shared geometry state.
    """
    payload = data if isinstance(data, dict) else {}
    try:
        n_poses = max(1, min(16, int(payload.get("poses", 8))))
        seed = int(payload.get("seed", 0))
    except (TypeError, ValueError):
        n_poses, seed = 8, 0
    logger.info("sense_room from %s: poses=%d seed=%d", sid, n_poses, seed)
    result = await sim_manager.sense(n_poses=n_poses, seed=seed)
    await sio.emit("sense_result", result, to=sid)
