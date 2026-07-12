import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import Volume, { VolumeBuffer } from './Volume';

// =====================================================================
// Wire types — must match the payloads emitted by SimulationManager.
//
// The schema is dim-tagged: the same `simulation_update` event carries
// a 2D nested-list grid OR a 3D binary buffer; the same status.obstacles
// field carries a 2D nested-list mask OR a 3D binary buffer. Each
// payload includes a `dims` field so the UI can dispatch without
// inferring from `engine.grid_shape.length`.
// =====================================================================

type SimulationUpdate2D = {
  dims?: 2;
  shape?: [number, number];
  grid: number[][];
  max_val: number;
  step: number;
  time: number;
};

type SimulationUpdate3D = {
  dims: 3;
  shape: [number, number, number];
  /** socket.io-client materialises binary attachments as ArrayBuffer. */
  data: ArrayBuffer | Uint8Array;
  max_val: number;
  step: number;
  time: number;
};

type SimulationUpdate = SimulationUpdate2D | SimulationUpdate3D;

interface EngineInfo {
  grid_shape?: number[];
  dims?: number;
  timestep?: number;
  wavespeed?: number;
  gridstep?: number;
  courant?: number;
  downsample?: number;
}

type ObstaclePayload2D = {
  dims?: 2;
  shape: [number, number];
  downsample: number;
  mask: number[][];
};

type ObstaclePayload3D = {
  dims: 3;
  shape: [number, number, number];
  downsample: number;
  mask: ArrayBuffer | Uint8Array;
};

type ObstaclePayload = ObstaclePayload2D | ObstaclePayload3D;

interface DriverInfo {
  position: number[];
  waveform: { type: string; [k: string]: unknown };
}

interface StatusPayload {
  is_running: boolean;
  engine: EngineInfo;
  config: Record<string, unknown>;
  obstacles?: ObstaclePayload;
  drivers?: DriverInfo[];
}

/** Reply to `sense_room`: the Phase 2 acoustic-mapping bridge. */
interface SenseResult {
  ok: boolean;
  error?: string;
  /** Bayes-fused obstacle-probability map, `shape` (64x64), row-major. */
  prob?: number[][];
  /** The room mask (resampled to `shape`) the IoUs are scored against. */
  truth?: number[][];
  shape?: [number, number];
  /** IoU progression: ious[k-1] uses the first k poses. */
  ious?: number[];
  poses?: number;
  elapsed?: number;
}

// Interaction mode for the canvas (2D only — 3D uses orbit controls
// instead and the click-modes do not apply).
type Mode = 'view' | 'draw-obstacle' | 'erase-obstacle' | 'place-driver' | 'remove-driver';

const CANVAS_PX = 600;
const DRAG_FLUSH_MS = 33;
const DRIVER_PICK_RADIUS_PX = 12;

const OBSTACLE_R = 80;
const OBSTACLE_G = 80;
const OBSTACLE_B = 90;

// =====================================================================
// Component
// =====================================================================

function App(): React.JSX.Element {
  // -- Refs that survive renders ----------------------------------------
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<Socket | null>(null);
  /** Last 2D pressure-frame, kept on a ref so the rAF loop can read it without re-rendering React. */
  const last2DRef = useRef<SimulationUpdate2D | null>(null);
  /** Per-frame 2D obstacle mask cached from the most recent status. */
  const obstacles2DRef = useRef<ObstaclePayload2D>({
    shape: [0, 0], downsample: 1, mask: [],
  });
  const driversRef = useRef<DriverInfo[]>([]);
  const engineRef = useRef<EngineInfo>({});
  const fpsRef = useRef<{ frames: number; last: number; fps: number }>({
    frames: 0, last: performance.now(), fps: 0,
  });

  // -- Connection / sim state ------------------------------------------
  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [engine, setEngine] = useState<EngineInfo>({});
  const [step, setStep] = useState(0);
  const [simTime, setSimTime] = useState(0);
  const [maxVal, setMaxVal] = useState(0);
  const [fps, setFps] = useState(0);
  const [driverListUI, setDriverListUI] = useState<DriverInfo[]>([]);

  // -- 3D buffers held in refs (NOT state). React 19's dev-mode
  //    profiler tries to deep-clone every prop into the Performance API
  //    via `performance.measure`; a typed array of even tens of KB
  //    triggers DataCloneError and derails the render cycle. Refs are
  //    opaque to the profiler, so the heavy buffer never enters that
  //    path. The frame counters below are the React-visible "this
  //    upload should re-fire" signal.
  const pressureRef = useRef<VolumeBuffer>({ bytes: null, shape: null });
  const obstacleRef = useRef<VolumeBuffer>({ bytes: null, shape: null });
  const [pressureFrame, setPressureFrame] = useState(0);
  const [obstacleFrame, setObstacleFrame] = useState(0);

  // -- Interaction mode + brush (2D only) -------------------------------
  const [mode, setMode] = useState<Mode>('view');
  const [brushRadius, setBrushRadius] = useState(3);

  // -- Acoustic sensing (2D only) ----------------------------------------
  const senseCanvasRef = useRef<HTMLCanvasElement>(null);
  const [senseBusy, setSenseBusy] = useState(false);
  const [sensePoses, setSensePoses] = useState(8);
  const [senseResult, setSenseResult] = useState<SenseResult | null>(null);

  // -- Parameter form (sent via update_config) --------------------------
  const [waveformType, setWaveformType] = useState('RickerWavelet');
  const [waveformAmp, setWaveformAmp] = useState(5.0);
  const [waveformFreq, setWaveformFreq] = useState(0.1);
  const [waveformDelay, setWaveformDelay] = useState(20.0);
  const [gridW, setGridW] = useState(200);
  const [gridH, setGridH] = useState(200);
  const [gridD, setGridD] = useState(64);          // depth, only used in 3D mode
  const [renderMode, setRenderMode] = useState<'2D' | '3D'>('2D');
  const [courantForm, setCourantForm] = useState(0.5);
  const [downsampleForm, setDownsampleForm] = useState(2);
  const [broadcastHz, setBroadcastHz] = useState(30.0);

  // -- 3D shader controls -----------------------------------------------
  const [gamma, setGamma] = useState(0.5);
  const [alphaScale, setAlphaScale] = useState(0.05);
  const [obstacleAlpha, setObstacleAlpha] = useState(0.6);
  const [raySteps, setRaySteps] = useState(96);
  const [resetCameraNonce, setResetCameraNonce] = useState(0);

  // -- Drag accumulator -------------------------------------------------
  const dragRef = useRef<{
    active: boolean;
    value: boolean;
    pending: Set<string>;
    flushTimer: number | null;
  }>({ active: false, value: true, pending: new Set(), flushTimer: null });

  const is3D = engine.dims === 3;

  // ===================================================================
  // Socket lifecycle
  // ===================================================================
  useEffect(() => {
    const socket = io('/', { path: '/socket.io/', transports: ['websocket', 'polling'] });
    socketRef.current = socket;

    socket.on('connect', () => setIsConnected(true));
    socket.on('disconnect', () => setIsConnected(false));
    socket.on('connect_error', (err) => console.warn('connect_error', err.message));

    socket.on('status', (payload: StatusPayload) => {
      setIsRunning(payload.is_running);
      setEngine(payload.engine || {});
      engineRef.current = payload.engine || {};

      // Obstacle payload — dispatch on dim. 2D goes to the canvas-side
      // ref; 3D goes to the volume buffer ref + frame counter so
      // <Volume /> uploads to GPU on the next effect tick.
      if (payload.obstacles) {
        if (payload.obstacles.dims === 3) {
          const buf = payload.obstacles.mask instanceof ArrayBuffer
            ? new Uint8Array(payload.obstacles.mask)
            : (payload.obstacles.mask as Uint8Array);
          obstacleRef.current = { bytes: buf, shape: payload.obstacles.shape };
          setObstacleFrame((n) => n + 1);
        } else {
          obstacles2DRef.current = payload.obstacles as ObstaclePayload2D;
        }
      }

      const drivers = payload.drivers || [];
      driversRef.current = drivers;
      setDriverListUI(drivers);

      // Mirror config waveform / geometry into the form on first arrival
      // so the user sees the engine's actual defaults rather than the
      // React initial state.
      const cfg = payload.config || {};
      const wf = (cfg.waveform as Record<string, unknown> | undefined) ?? {};
      if (typeof wf.type === 'string') setWaveformType(wf.type);
      if (typeof wf.amplitude === 'number') setWaveformAmp(wf.amplitude);
      if (typeof wf.frequency === 'number') setWaveformFreq(wf.frequency);
      if (typeof wf.delay === 'number') setWaveformDelay(wf.delay);
      const gs = cfg.grid_shape as number[] | undefined;
      if (Array.isArray(gs)) {
        if (gs.length === 2) {
          setGridH(gs[0]);
          setGridW(gs[1]);
          setRenderMode('2D');
        } else if (gs.length === 3) {
          setGridH(gs[0]);
          setGridW(gs[1]);
          setGridD(gs[2]);
          setRenderMode('3D');
        }
      }
      if (typeof cfg.courant === 'number') setCourantForm(cfg.courant);
      if (typeof cfg.downsample === 'number') setDownsampleForm(cfg.downsample);
      if (typeof cfg.broadcast_hz === 'number') setBroadcastHz(cfg.broadcast_hz);
    });

    socket.on('sense_result', (payload: SenseResult) => {
      setSenseBusy(false);
      setSenseResult(payload);
    });

    socket.on('simulation_update', (payload: SimulationUpdate) => {
      setStep(payload.step);
      setSimTime(payload.time);
      setMaxVal(payload.max_val);

      if (payload.dims === 3) {
        // Binary 3D payload. Stash the bytes on the ref + bump the
        // counter; the Volume component's effect re-runs and uploads
        // the Uint8Array directly into its Data3DTexture.
        const buf = payload.data instanceof ArrayBuffer
          ? new Uint8Array(payload.data)
          : (payload.data as Uint8Array);
        pressureRef.current = { bytes: buf, shape: payload.shape };
        setPressureFrame((n) => n + 1);
      } else {
        last2DRef.current = payload as SimulationUpdate2D;
      }
    });

    return () => {
      socket.removeAllListeners();
      socket.disconnect();
      socketRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ===================================================================
  // 2D render loop — only paints when in 2D mode (the canvas is hidden
  // otherwise but the rAF loop is harmless to leave running).
  // ===================================================================
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const data = last2DRef.current;
      const obs = obstacles2DRef.current;
      const drivers = driversRef.current;
      const eng = engineRef.current;
      // Skip painting when in 3D mode — the canvas element is unmounted
      // anyway, but defensive guard avoids wasted work.
      if (data && eng.dims !== 3) {
        drawScene(canvasRef.current, data, obs, drivers, eng);
        const now = performance.now();
        const f = fpsRef.current;
        f.frames += 1;
        if (now - f.last > 500) {
          f.fps = (1000 * f.frames) / (now - f.last);
          f.frames = 0;
          f.last = now;
          setFps(f.fps);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // ===================================================================
  // Sensing result canvas — repainted whenever a new result arrives.
  // Viridis for the probability map; the drawn room is ghosted in as
  // whitened cells so prediction and truth are comparable in one image.
  // ===================================================================
  useEffect(() => {
    const canvas = senseCanvasRef.current;
    const r = senseResult;
    if (!canvas || !r?.ok || !r.prob || !r.shape) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const [h, w] = r.shape;
    const buf = ctx.createImageData(w, h);
    const px = buf.data;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let [cr, cg, cb] = viridis(r.prob[y][x]);
        if (r.truth && r.truth[y][x]) {
          // Ghost the true obstacle cells: 55% colour, 45% white.
          cr = Math.round(0.55 * cr + 0.45 * 255);
          cg = Math.round(0.55 * cg + 0.45 * 255);
          cb = Math.round(0.55 * cb + 0.45 * 255);
        }
        const idx = (y * w + x) * 4;
        px[idx] = cr;
        px[idx + 1] = cg;
        px[idx + 2] = cb;
        px[idx + 3] = 255;
      }
    }
    if (senseOffscreen === null) senseOffscreen = document.createElement('canvas');
    if (senseOffscreen.width !== w || senseOffscreen.height !== h) {
      senseOffscreen.width = w;
      senseOffscreen.height = h;
    }
    const offCtx = senseOffscreen.getContext('2d');
    if (!offCtx) return;
    offCtx.putImageData(buf, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(senseOffscreen, 0, 0, canvas.width, canvas.height);
  }, [senseResult]);

  // ===================================================================
  // Socket emit helpers
  // ===================================================================
  const send = useCallback((event: string, payload: unknown = {}) => {
    socketRef.current?.emit(event, payload);
  }, []);

  const flushDrag = useCallback(() => {
    const d = dragRef.current;
    if (d.pending.size === 0) return;
    const positions: number[][] = [];
    d.pending.forEach((key) => {
      positions.push(key.split(',').map((s) => parseInt(s, 10)));
    });
    d.pending.clear();
    send('set_obstacle', { positions, value: d.value });
  }, [send]);

  const scheduleFlush = useCallback(() => {
    const d = dragRef.current;
    if (d.flushTimer != null) return;
    d.flushTimer = window.setTimeout(() => {
      d.flushTimer = null;
      flushDrag();
    }, DRAG_FLUSH_MS);
  }, [flushDrag]);

  // ===================================================================
  // 2D coordinate translation: canvas px -> full-grid (i, j).
  // ===================================================================
  const canvasToGrid = useCallback(
    (cx: number, cy: number): [number, number] | null => {
      const gs = engineRef.current.grid_shape;
      if (!gs || gs.length !== 2) return null;
      const [fh, fw] = gs;
      const i = Math.max(0, Math.min(fh - 1, Math.floor((cy / CANVAS_PX) * fh)));
      const j = Math.max(0, Math.min(fw - 1, Math.floor((cx / CANVAS_PX) * fw)));
      return [i, j];
    },
    [],
  );

  const eventToCanvasPx = useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement>): [number, number] => {
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / rect.width;
      const sy = canvas.height / rect.height;
      return [(ev.clientX - rect.left) * sx, (ev.clientY - rect.top) * sy];
    },
    [],
  );

  const brushCells = useCallback(
    (i0: number, j0: number): [number, number][] => {
      const gs = engineRef.current.grid_shape;
      if (!gs || gs.length !== 2) return [];
      const [fh, fw] = gs;
      const r = Math.max(0, brushRadius | 0);
      const r2 = r * r;
      const out: [number, number][] = [];
      for (let di = -r; di <= r; di++) {
        const i = i0 + di;
        if (i < 0 || i >= fh) continue;
        for (let dj = -r; dj <= r; dj++) {
          const j = j0 + dj;
          if (j < 0 || j >= fw) continue;
          if (di * di + dj * dj > r2) continue;
          out.push([i, j]);
        }
      }
      return out;
    },
    [brushRadius],
  );

  // ===================================================================
  // 2D canvas mouse handlers — only fire when the canvas is mounted
  // (i.e. in 2D mode). 3D mode swaps in the Volume component which
  // owns its own input handling via OrbitControls.
  // ===================================================================
  const handleMouseDown = useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement>) => {
      if (mode === 'view') return;
      const [cx, cy] = eventToCanvasPx(ev);
      const ij = canvasToGrid(cx, cy);
      if (!ij) return;

      if (mode === 'draw-obstacle' || mode === 'erase-obstacle') {
        dragRef.current.active = true;
        dragRef.current.value = mode === 'draw-obstacle';
        for (const [i, j] of brushCells(ij[0], ij[1])) {
          dragRef.current.pending.add(`${i},${j}`);
        }
        scheduleFlush();
      } else if (mode === 'place-driver') {
        send('add_driver', {
          position: ij,
          waveform: buildWaveformSpec(waveformType, waveformAmp, waveformFreq, waveformDelay),
        });
      } else if (mode === 'remove-driver') {
        const gs = engineRef.current.grid_shape;
        if (!gs || gs.length !== 2) return;
        const [fh, fw] = gs;
        let bestIdx = -1;
        let bestD2 = DRIVER_PICK_RADIUS_PX * DRIVER_PICK_RADIUS_PX;
        for (let i = 0; i < driversRef.current.length; i++) {
          const [iy, ix] = driversRef.current[i].position;
          const dx = (ix / fw) * CANVAS_PX - cx;
          const dy = (iy / fh) * CANVAS_PX - cy;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) {
            bestD2 = d2;
            bestIdx = i;
          }
        }
        if (bestIdx >= 0) send('remove_driver', { index: bestIdx });
      }
    },
    [mode, brushCells, canvasToGrid, eventToCanvasPx, scheduleFlush, send,
     waveformType, waveformAmp, waveformFreq, waveformDelay],
  );

  const handleMouseMove = useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement>) => {
      if (!dragRef.current.active) return;
      if (mode !== 'draw-obstacle' && mode !== 'erase-obstacle') return;
      const [cx, cy] = eventToCanvasPx(ev);
      const ij = canvasToGrid(cx, cy);
      if (!ij) return;
      for (const [i, j] of brushCells(ij[0], ij[1])) {
        dragRef.current.pending.add(`${i},${j}`);
      }
      scheduleFlush();
    },
    [mode, brushCells, canvasToGrid, eventToCanvasPx, scheduleFlush],
  );

  const handleMouseUp = useCallback(() => {
    if (!dragRef.current.active) return;
    dragRef.current.active = false;
    if (dragRef.current.flushTimer != null) {
      window.clearTimeout(dragRef.current.flushTimer);
      dragRef.current.flushTimer = null;
    }
    flushDrag();
  }, [flushDrag]);

  // ===================================================================
  // Top-level controls
  // ===================================================================
  const handleStart = () => send('start_simulation');
  const handleStop = () => send('stop_simulation');
  const handleReset = () => send('reset_simulation');
  const handleSense = () => {
    setSenseBusy(true);
    // Fresh random poses each click: the seed is drawn client-side so
    // repeated clicks show the pose-to-pose variability honestly.
    send('sense_room', { poses: sensePoses, seed: Math.floor(Math.random() * 1e9) });
  };
  const handleClearObstacles = () => send('clear_obstacles');
  const handleClearDrivers = () => send('clear_drivers');
  const handleResetCamera = () => setResetCameraNonce((n) => n + 1);

  /** Apply the geometry/waveform form. Switching dim is a structural
   *  rebuild — the backend resets driver_position to the new grid centre
   *  and discards the field, drivers, and obstacles. */
  const handleApplyConfig = () => {
    const gridShape = renderMode === '3D' ? [gridH, gridW, gridD] : [gridH, gridW];
    send('update_config', {
      grid_shape: gridShape,
      courant: courantForm,
      downsample: downsampleForm,
      broadcast_hz: broadcastHz,
      waveform: buildWaveformSpec(waveformType, waveformAmp, waveformFreq, waveformDelay),
    });
  };

  // ===================================================================
  // UI labels
  // ===================================================================
  const cflLabel = useMemo(() => {
    const c = engine.courant;
    if (c === undefined) return '—';
    return c.toFixed(3);
  }, [engine.courant]);

  const cursorStyle: React.CSSProperties['cursor'] = useMemo(() => {
    if (is3D) return 'grab';
    switch (mode) {
      case 'view': return 'default';
      case 'draw-obstacle':
      case 'erase-obstacle': return 'crosshair';
      case 'place-driver': return 'copy';
      case 'remove-driver': return 'pointer';
      default: return 'default';
    }
  }, [mode, is3D]);

  // Convert 2D drivers to 3-tuples for the Volume markers (pad missing axis with 0).
  const drivers3D = useMemo(
    () =>
      driverListUI.map((d) => ({
        position:
          d.position.length === 3
            ? ([d.position[0], d.position[1], d.position[2]] as [number, number, number])
            : ([d.position[0], d.position[1], 0] as [number, number, number]),
      })),
    [driverListUI],
  );
  const fullGridShape3D: [number, number, number] | null = useMemo(() => {
    const gs = engine.grid_shape;
    if (!gs || gs.length !== 3) return null;
    return [gs[0], gs[1], gs[2]];
  }, [engine.grid_shape]);

  return (
    <div className="App">
      <header>
        <h1>Acoustic FDTD Simulation</h1>
        <div className="status-line">
          <span className={isConnected ? 'pill ok' : 'pill bad'}>
            {isConnected ? 'connected' : 'disconnected'}
          </span>
          <span className={isRunning ? 'pill ok' : 'pill idle'}>
            {isRunning ? 'running' : 'idle'}
          </span>
          <span className="pill idle">{is3D ? '3D' : '2D'}</span>
          <span className="meta">step {step}</span>
          <span className="meta">t = {simTime.toFixed(2)}</span>
          <span className="meta">max|p| = {maxVal.toExponential(2)}</span>
          <span className="meta">{is3D ? 'orbit' : `render ${fps.toFixed(0)} fps`}</span>
        </div>
      </header>

      <div className="controls">
        <button onClick={handleStart} disabled={isRunning || !isConnected}>Start</button>
        <button onClick={handleStop} disabled={!isRunning || !isConnected}>Stop</button>
        <button onClick={handleReset} disabled={!isConnected}>Reset</button>
      </div>

      <div className="layout">
        <div className="canvas-container">
          {is3D ? (
            <Volume
              pressureRef={pressureRef}
              pressureFrame={pressureFrame}
              maxVal={maxVal}
              obstacleRef={obstacleRef}
              obstacleFrame={obstacleFrame}
              drivers={drivers3D}
              fullGridShape={fullGridShape3D}
              gamma={gamma}
              alphaScale={alphaScale}
              obstacleAlpha={obstacleAlpha}
              raySteps={raySteps}
              resetCameraNonce={resetCameraNonce}
              width={CANVAS_PX}
              height={CANVAS_PX}
            />
          ) : (
            <canvas
              ref={canvasRef}
              width={CANVAS_PX}
              height={CANVAS_PX}
              style={{ border: '1px solid #444', imageRendering: 'pixelated', cursor: cursorStyle }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            />
          )}
        </div>

        <aside className="side-panel">
          {!is3D && (
            <section className="panel">
              <h2>Mode</h2>
              <div className="mode-grid">
                {([
                  ['view', 'View'],
                  ['draw-obstacle', 'Draw obstacle'],
                  ['erase-obstacle', 'Erase obstacle'],
                  ['place-driver', 'Place driver'],
                  ['remove-driver', 'Remove driver'],
                ] as [Mode, string][]).map(([m, label]) => (
                  <button
                    key={m}
                    className={mode === m ? 'mode-btn active' : 'mode-btn'}
                    onClick={() => setMode(m)}
                    disabled={!isConnected}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <label className="row">
                brush radius
                <input
                  type="number"
                  min={0}
                  max={30}
                  value={brushRadius}
                  onChange={(e) => setBrushRadius(Math.max(0, parseInt(e.target.value || '0', 10)))}
                  disabled={mode !== 'draw-obstacle' && mode !== 'erase-obstacle'}
                />
                <span className="hint">cells</span>
              </label>
              <div className="row">
                <button onClick={handleClearObstacles} disabled={!isConnected}>Clear obstacles</button>
                <button onClick={handleClearDrivers} disabled={!isConnected}>Clear drivers</button>
              </div>
            </section>
          )}

          {is3D && (
            <section className="panel">
              <h2>3D view</h2>
              <label className="row">
                γ (alpha exponent)
                <input
                  type="range"
                  min={0.3}
                  max={4.0}
                  step={0.05}
                  value={gamma}
                  onChange={(e) => setGamma(parseFloat(e.target.value))}
                />
                <span className="hint">{gamma.toFixed(2)}</span>
              </label>
              <label className="row">
                α scale
                <input
                  type="range"
                  min={0.005}
                  max={0.5}
                  step={0.005}
                  value={alphaScale}
                  onChange={(e) => setAlphaScale(parseFloat(e.target.value))}
                />
                <span className="hint">{alphaScale.toFixed(3)}</span>
              </label>
              <label className="row">
                obstacle α
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={obstacleAlpha}
                  onChange={(e) => setObstacleAlpha(parseFloat(e.target.value))}
                />
                <span className="hint">{obstacleAlpha.toFixed(2)}</span>
              </label>
              <label className="row">
                ray steps
                <select value={raySteps} onChange={(e) => setRaySteps(parseInt(e.target.value, 10))}>
                  <option value={32}>32</option>
                  <option value={64}>64</option>
                  <option value={96}>96</option>
                  <option value={128}>128</option>
                  <option value={192}>192</option>
                  <option value={256}>256</option>
                </select>
              </label>
              <div className="row">
                <button onClick={handleResetCamera} disabled={!isConnected}>Reset camera</button>
                <button onClick={handleClearObstacles} disabled={!isConnected}>Clear obstacles</button>
                <button onClick={handleClearDrivers} disabled={!isConnected}>Clear drivers</button>
              </div>
              <div className="hint">
                Drag to orbit, scroll to zoom. Obstacle / driver placement is currently
                2D-only; for 3D scenes use <code>scripts/generate_active_sensing.py</code>
                or send <code>set_obstacle</code> over the socket directly.
              </div>
            </section>
          )}

          {!is3D && (
            <section className="panel">
              <h2>Acoustic sensing</h2>
              <div className="hint">
                Phase 2 demo: chirp + record at K virtual poses in the drawn room,
                infer per pose, Bayes-fuse into an obstacle-probability map.
              </div>
              <label className="row">
                poses (K)
                <input
                  type="number" min={1} max={16} step={1}
                  value={sensePoses}
                  onChange={(e) => setSensePoses(Math.max(1, Math.min(16, parseInt(e.target.value || '8', 10))))}
                />
              </label>
              <div className="row">
                <button onClick={handleSense} disabled={!isConnected || senseBusy}>
                  {senseBusy ? 'Sensing…' : 'Sense room'}
                </button>
              </div>
              {senseResult && !senseResult.ok && (
                <div className="hint" style={{ color: '#d66' }}>{senseResult.error}</div>
              )}
              {senseResult?.ok && (
                <>
                  <canvas
                    ref={senseCanvasRef}
                    width={256}
                    height={256}
                    style={{ border: '1px solid #444', imageRendering: 'pixelated', width: '100%' }}
                  />
                  <div className="hint">
                    Viridis: dark = free, bright = inferred obstacle. True obstacle
                    cells are whitened for comparison. IoU by poses fused:{' '}
                    {senseResult.ious?.map((v, i) => `K=${i + 1}: ${v.toFixed(3)}`).join('  ')}
                    {'  '}({senseResult.elapsed}s)
                  </div>
                </>
              )}
            </section>
          )}

          <section className="panel">
            <h2>Waveform</h2>
            <label className="row">
              type
              <select value={waveformType} onChange={(e) => setWaveformType(e.target.value)}>
                <option value="RickerWavelet">RickerWavelet</option>
                <option value="GaussianPulse">GaussianPulse</option>
                <option value="Cosine">Cosine</option>
              </select>
            </label>
            <label className="row">
              amplitude
              <input
                type="number"
                step={0.1}
                value={waveformAmp}
                onChange={(e) => setWaveformAmp(parseFloat(e.target.value || '0'))}
              />
            </label>
            <label className="row">
              frequency
              <input
                type="number"
                step={0.01}
                value={waveformFreq}
                onChange={(e) => setWaveformFreq(parseFloat(e.target.value || '0'))}
              />
            </label>
            <label className="row">
              delay (Ricker/Gaussian only)
              <input
                type="number"
                step={1}
                value={waveformDelay}
                onChange={(e) => setWaveformDelay(parseFloat(e.target.value || '0'))}
              />
            </label>
            <div className="hint">
              Used by <em>Apply</em> below and by every <em>Place driver</em> click in 2D.
            </div>
          </section>

          <section className="panel">
            <h2>Geometry &amp; cadence</h2>
            <div className="row" role="radiogroup">
              dim
              <label>
                <input
                  type="radio"
                  name="renderMode"
                  checked={renderMode === '2D'}
                  onChange={() => setRenderMode('2D')}
                /> 2D
              </label>
              <label>
                <input
                  type="radio"
                  name="renderMode"
                  checked={renderMode === '3D'}
                  onChange={() => setRenderMode('3D')}
                /> 3D
              </label>
              <span className="hint">Apply to take effect</span>
            </div>
            <label className="row">
              grid (i)
              <input
                type="number" min={16} max={1024} step={2}
                value={gridH}
                onChange={(e) => setGridH(parseInt(e.target.value || '0', 10))}
              />
            </label>
            <label className="row">
              grid (j)
              <input
                type="number" min={16} max={1024} step={2}
                value={gridW}
                onChange={(e) => setGridW(parseInt(e.target.value || '0', 10))}
              />
            </label>
            {renderMode === '3D' && (
              <label className="row">
                grid (k)
                <input
                  type="number" min={16} max={512} step={2}
                  value={gridD}
                  onChange={(e) => setGridD(parseInt(e.target.value || '0', 10))}
                />
              </label>
            )}
            <label className="row">
              courant
              <input
                type="number" min={0.05} max={0.7} step={0.05}
                value={courantForm}
                onChange={(e) => setCourantForm(parseFloat(e.target.value || '0'))}
              />
              <span className="hint">CFL ≤ 1/√d</span>
            </label>
            <label className="row">
              downsample
              <input
                type="number" min={1} max={8} step={1}
                value={downsampleForm}
                onChange={(e) => setDownsampleForm(parseInt(e.target.value || '1', 10))}
              />
              <span className="hint">wire-only</span>
            </label>
            <label className="row">
              broadcast Hz
              <input
                type="number" min={1} max={120} step={1}
                value={broadcastHz}
                onChange={(e) => setBroadcastHz(parseFloat(e.target.value || '0'))}
              />
            </label>
            <div className="row">
              <button onClick={handleApplyConfig} disabled={!isConnected}>
                Apply (rebuilds simulation)
              </button>
            </div>
            <div className="hint">
              Apply discards the current field, drivers, and obstacles.
            </div>
          </section>

          <section className="panel">
            <h2>Drivers ({driverListUI.length})</h2>
            {driverListUI.length === 0 ? (
              <div className="hint">No drivers placed.</div>
            ) : (
              <ul className="driver-list">
                {driverListUI.map((d, i) => (
                  <li key={i}>
                    <span>
                      [{i}] {d.waveform.type} @ ({d.position.join(', ')})
                    </span>
                    <button onClick={() => send('remove_driver', { index: i })} className="mini">
                      remove
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </aside>
      </div>

      <footer className="meta-block">
        <div>
          grid {engine.grid_shape?.join(' x ') ?? '—'} &middot;
          dt = {engine.timestep?.toExponential(2) ?? '—'} &middot;
          c = {engine.wavespeed ?? '—'} &middot;
          dx = {engine.gridstep ?? '—'} &middot;
          Courant = {cflLabel}
        </div>
        <div className="legend">
          <span style={{ background: 'rgb(40,80,200)' }} /> negative pressure
          <span style={{ background: 'rgb(245,245,245)', color: '#333' }} /> zero
          <span style={{ background: 'rgb(220,40,40)' }} /> positive pressure
          <span style={{ background: `rgb(${OBSTACLE_R},${OBSTACLE_G},${OBSTACLE_B})` }} /> obstacle
          <span style={{ background: 'rgba(255,200,0,0.85)' }} /> driver
        </div>
      </footer>
    </div>
  );
}

// =====================================================================
// Waveform spec helper (matches build_waveform on the backend)
// =====================================================================

function buildWaveformSpec(
  type: string,
  amplitude: number,
  frequency: number,
  delay: number,
): Record<string, number | string> {
  const spec: Record<string, number | string> = { type, amplitude, frequency };
  if (type === 'GaussianPulse') {
    spec.center_time = delay;
    spec.width = 1.0 / Math.max(frequency, 1e-6);
  } else if (type === 'RickerWavelet') {
    spec.delay = delay;
  }
  return spec;
}

// =====================================================================
// 2D canvas rendering — composes pressure + obstacles + drivers.
// Unchanged from the prior implementation; only the type signatures
// loosened to match the dim-tagged ObstaclePayload union.
// =====================================================================

let offscreen: HTMLCanvasElement | null = null;
/** Separate offscreen for the sensing panel so its 64x64 blits don't
 *  fight the render loop's pressure-sized buffer. */
let senseOffscreen: HTMLCanvasElement | null = null;

/** Piecewise-linear approximation of matplotlib's viridis, matching the
 *  colormap used by the offline demo (scripts/demo_room_mapping.py). */
function viridis(v: number): [number, number, number] {
  const stops: [number, number, number][] = [
    [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
  ];
  const t = Math.max(0, Math.min(1, v)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  const f = t - i;
  return [
    Math.round(stops[i][0] + (stops[i + 1][0] - stops[i][0]) * f),
    Math.round(stops[i][1] + (stops[i + 1][1] - stops[i][1]) * f),
    Math.round(stops[i][2] + (stops[i + 1][2] - stops[i][2]) * f),
  ];
}

function drawScene(
  canvas: HTMLCanvasElement | null,
  data: SimulationUpdate2D,
  obs: ObstaclePayload2D,
  drivers: DriverInfo[],
  engine: EngineInfo,
): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const { grid, max_val } = data;
  const h = grid.length;
  if (h === 0) return;
  const w = grid[0].length;
  if (w === 0) return;

  const absMax = Math.max(1e-6, max_val);
  const obsAligned =
    Array.isArray(obs.mask) &&
    obs.shape[0] === h &&
    obs.shape[1] === w &&
    obs.mask.length === h;

  const buf = ctx.createImageData(w, h);
  const px = buf.data;
  for (let y = 0; y < h; y++) {
    const row = grid[y];
    const obsRow = obsAligned ? obs.mask[y] : undefined;
    for (let x = 0; x < w; x++) {
      let r: number, g: number, b: number;
      if (obsRow && obsRow[x]) {
        r = OBSTACLE_R;
        g = OBSTACLE_G;
        b = OBSTACLE_B;
      } else {
        const v = Math.max(-1, Math.min(1, row[x] / absMax));
        [r, g, b] = divergingColormap(v);
      }
      const idx = (y * w + x) * 4;
      px[idx] = r;
      px[idx + 1] = g;
      px[idx + 2] = b;
      px[idx + 3] = 255;
    }
  }

  if (offscreen === null) offscreen = document.createElement('canvas');
  if (offscreen.width !== w || offscreen.height !== h) {
    offscreen.width = w;
    offscreen.height = h;
  }
  const offCtx = offscreen.getContext('2d');
  if (!offCtx) return;
  offCtx.putImageData(buf, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);

  const gs = engine.grid_shape;
  if (gs && gs.length === 2 && drivers.length) {
    const [fh, fw] = gs;
    for (const d of drivers) {
      if (d.position.length !== 2) continue;
      const [iy, ix] = d.position;
      const cx = (ix / fw) * canvas.width;
      const cy = (iy / fh) * canvas.height;
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(255,200,0,0.85)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.85)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}

function divergingColormap(v: number): [number, number, number] {
  const negR = 40, negG = 80, negB = 200;
  const zeroR = 245, zeroG = 245, zeroB = 245;
  const posR = 220, posG = 40, posB = 40;
  if (v >= 0) {
    const t = v;
    return [
      Math.round(zeroR + (posR - zeroR) * t),
      Math.round(zeroG + (posG - zeroG) * t),
      Math.round(zeroB + (posB - zeroB) * t),
    ];
  } else {
    const t = -v;
    return [
      Math.round(zeroR + (negR - zeroR) * t),
      Math.round(zeroG + (negG - zeroG) * t),
      Math.round(zeroB + (negB - zeroB) * t),
    ];
  }
}

export default App;
