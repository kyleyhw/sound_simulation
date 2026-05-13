import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

// =====================================================================
// Wire types — must match the payloads emitted by SimulationManager.
// =====================================================================

interface SimulationUpdate {
  grid: number[][];
  max_val: number;
  step: number;
  time: number;
}

interface EngineInfo {
  grid_shape?: [number, number];
  timestep?: number;
  wavespeed?: number;
  gridstep?: number;
  courant?: number;
  downsample?: number;
}

interface ObstaclePayload {
  shape: [number, number];
  downsample: number;
  mask: number[][];
}

interface DriverInfo {
  position: [number, number];
  waveform: { type: string; [k: string]: unknown };
}

interface StatusPayload {
  is_running: boolean;
  engine: EngineInfo;
  config: Record<string, unknown>;
  obstacles?: ObstaclePayload;
  drivers?: DriverInfo[];
}

// Interaction mode for the canvas.
type Mode = 'view' | 'draw-obstacle' | 'erase-obstacle' | 'place-driver' | 'remove-driver';

const CANVAS_PX = 600;
const DRAG_FLUSH_MS = 33; // ~30 Hz batch flush during a brush stroke
const DRIVER_PICK_RADIUS_PX = 12;

// Visual constants. Kept near the top so colour tweaks don't require diving
// into the colormap.
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
  const lastUpdateRef = useRef<SimulationUpdate | null>(null);
  const obstaclesRef = useRef<ObstaclePayload>({ shape: [0, 0], downsample: 1, mask: [] });
  const driversRef = useRef<DriverInfo[]>([]);
  const engineRef = useRef<EngineInfo>({});
  const fpsRef = useRef<{ frames: number; last: number; fps: number }>({
    frames: 0,
    last: performance.now(),
    fps: 0,
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

  // -- Interaction mode + brush -----------------------------------------
  const [mode, setMode] = useState<Mode>('view');
  const [brushRadius, setBrushRadius] = useState(3);

  // -- Parameter form (sent via update_config) --------------------------
  const [waveformType, setWaveformType] = useState('RickerWavelet');
  const [waveformAmp, setWaveformAmp] = useState(5.0);
  const [waveformFreq, setWaveformFreq] = useState(0.1);
  const [waveformDelay, setWaveformDelay] = useState(20.0);
  const [gridW, setGridW] = useState(200);
  const [gridH, setGridH] = useState(200);
  const [courantForm, setCourantForm] = useState(0.5);
  const [downsampleForm, setDownsampleForm] = useState(2);
  const [broadcastHz, setBroadcastHz] = useState(30.0);

  // -- Drag accumulator -------------------------------------------------
  // We collect grid cells touched during a brush stroke and flush them at
  // DRAG_FLUSH_MS cadence rather than emitting one socket event per
  // mousemove. The set is keyed by "i,j" string for cheap dedup.
  const dragRef = useRef<{
    active: boolean;
    value: boolean;             // true = paint obstacle, false = erase
    pending: Set<string>;       // unsent positions, "i,j" strings
    flushTimer: number | null;  // window.setTimeout handle
  }>({ active: false, value: true, pending: new Set(), flushTimer: null });

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
      if (payload.obstacles) {
        obstaclesRef.current = payload.obstacles;
      }
      const drivers = payload.drivers || [];
      driversRef.current = drivers;
      setDriverListUI(drivers);
      // Mirror config waveform into the form on first arrival so the user
      // sees the engine's actual defaults rather than the React initial
      // state.
      const cfg = payload.config || {};
      const wf = (cfg.waveform as Record<string, unknown> | undefined) ?? {};
      if (typeof wf.type === 'string') setWaveformType(wf.type);
      if (typeof wf.amplitude === 'number') setWaveformAmp(wf.amplitude);
      if (typeof wf.frequency === 'number') setWaveformFreq(wf.frequency);
      if (typeof wf.delay === 'number') setWaveformDelay(wf.delay);
      const gs = cfg.grid_shape as number[] | undefined;
      if (Array.isArray(gs) && gs.length === 2) {
        setGridH(gs[0]);
        setGridW(gs[1]);
      }
      if (typeof cfg.courant === 'number') setCourantForm(cfg.courant);
      if (typeof cfg.downsample === 'number') setDownsampleForm(cfg.downsample);
      if (typeof cfg.broadcast_hz === 'number') setBroadcastHz(cfg.broadcast_hz);
    });

    socket.on('simulation_update', (data: SimulationUpdate) => {
      lastUpdateRef.current = data;
      setStep(data.step);
      setSimTime(data.time);
      setMaxVal(data.max_val);
    });

    return () => {
      socket.removeAllListeners();
      socket.disconnect();
      socketRef.current = null;
    };
    // Effect intentionally runs once — socket lifecycle is module-scoped.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ===================================================================
  // Render loop — one paint per rAF, decoupled from wire cadence.
  // ===================================================================
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const data = lastUpdateRef.current;
      const obs = obstaclesRef.current;
      const drivers = driversRef.current;
      const eng = engineRef.current;
      if (data) {
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
  // Socket emit helpers
  // ===================================================================
  const send = useCallback((event: string, payload: unknown = {}) => {
    socketRef.current?.emit(event, payload);
  }, []);

  const flushDrag = useCallback(() => {
    const d = dragRef.current;
    if (d.pending.size === 0) return;
    const positions: [number, number][] = [];
    d.pending.forEach((key) => {
      const [i, j] = key.split(',').map((s) => parseInt(s, 10));
      positions.push([i, j]);
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
  // Coordinate translation: canvas px -> full-grid (i, j).
  //
  // Engine grid_shape is (full_h, full_w). Canvas is CANVAS_PX × CANVAS_PX
  // showing the downsampled view. Mapping is straight ratio in each axis;
  // we clamp to the valid range so a click on the very edge stays in bounds.
  // ===================================================================
  const canvasToGrid = useCallback(
    (cx: number, cy: number): [number, number] | null => {
      const gs = engineRef.current.grid_shape;
      if (!gs) return null;
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

  // Disc of cells within `brushRadius` of (i0, j0), clipped to grid bounds.
  const brushCells = useCallback(
    (i0: number, j0: number): [number, number][] => {
      const gs = engineRef.current.grid_shape;
      if (!gs) return [];
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
  // Canvas mouse handlers — dispatched by current mode.
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
          waveform: buildWaveformSpec(
            waveformType,
            waveformAmp,
            waveformFreq,
            waveformDelay,
          ),
        });
      } else if (mode === 'remove-driver') {
        // Pick the nearest driver within DRIVER_PICK_RADIUS_PX of the click.
        const gs = engineRef.current.grid_shape;
        if (!gs) return;
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
    [
      mode,
      brushCells,
      canvasToGrid,
      eventToCanvasPx,
      scheduleFlush,
      send,
      waveformType,
      waveformAmp,
      waveformFreq,
      waveformDelay,
    ],
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
  // Top-level control buttons
  // ===================================================================
  const handleStart = () => send('start_simulation');
  const handleStop = () => send('stop_simulation');
  const handleReset = () => send('reset_simulation');
  const handleClearObstacles = () => send('clear_obstacles');
  const handleClearDrivers = () => send('clear_drivers');

  const handleApplyConfig = () => {
    send('update_config', {
      grid_shape: [gridH, gridW],
      courant: courantForm,
      downsample: downsampleForm,
      broadcast_hz: broadcastHz,
      waveform: buildWaveformSpec(
        waveformType,
        waveformAmp,
        waveformFreq,
        waveformDelay,
      ),
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
    switch (mode) {
      case 'view':
        return 'default';
      case 'draw-obstacle':
      case 'erase-obstacle':
        return 'crosshair';
      case 'place-driver':
        return 'copy';
      case 'remove-driver':
        return 'pointer';
      default:
        return 'default';
    }
  }, [mode]);

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
          <span className="meta">step {step}</span>
          <span className="meta">t = {simTime.toFixed(2)}</span>
          <span className="meta">max|p| = {maxVal.toExponential(2)}</span>
          <span className="meta">render {fps.toFixed(0)} fps</span>
        </div>
      </header>

      <div className="controls">
        <button onClick={handleStart} disabled={isRunning || !isConnected}>Start</button>
        <button onClick={handleStop} disabled={!isRunning || !isConnected}>Stop</button>
        <button onClick={handleReset} disabled={!isConnected}>Reset</button>
      </div>

      <div className="layout">
        <div className="canvas-container">
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
        </div>

        <aside className="side-panel">
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
              <button onClick={handleClearObstacles} disabled={!isConnected}>
                Clear obstacles
              </button>
              <button onClick={handleClearDrivers} disabled={!isConnected}>
                Clear drivers
              </button>
            </div>
          </section>

          <section className="panel">
            <h2>Waveform</h2>
            <label className="row">
              type
              <select
                value={waveformType}
                onChange={(e) => setWaveformType(e.target.value)}
              >
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
              These values are used both for the parameter <em>Apply</em> below
              (rebuilds the simulation with one driver from this waveform) and
              for every <em>Place driver</em> click.
            </div>
          </section>

          <section className="panel">
            <h2>Geometry &amp; cadence</h2>
            <label className="row">
              grid rows (i)
              <input
                type="number"
                min={16}
                max={1024}
                step={2}
                value={gridH}
                onChange={(e) => setGridH(parseInt(e.target.value || '0', 10))}
              />
            </label>
            <label className="row">
              grid cols (j)
              <input
                type="number"
                min={16}
                max={1024}
                step={2}
                value={gridW}
                onChange={(e) => setGridW(parseInt(e.target.value || '0', 10))}
              />
            </label>
            <label className="row">
              courant
              <input
                type="number"
                min={0.05}
                max={0.7}
                step={0.05}
                value={courantForm}
                onChange={(e) => setCourantForm(parseFloat(e.target.value || '0'))}
              />
              <span className="hint">CFL: σ ≤ 1/√2 ≈ 0.707 (2D)</span>
            </label>
            <label className="row">
              downsample
              <input
                type="number"
                min={1}
                max={8}
                step={1}
                value={downsampleForm}
                onChange={(e) => setDownsampleForm(parseInt(e.target.value || '1', 10))}
              />
              <span className="hint">wire-only, no physical effect</span>
            </label>
            <label className="row">
              broadcast Hz
              <input
                type="number"
                min={1}
                max={120}
                step={1}
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
              Apply discards the current field, drivers, and obstacles — the
              engine is reconstructed with the new geometry.
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
                      [{i}] {d.waveform.type} @ ({d.position[0]}, {d.position[1]})
                    </span>
                    <button
                      onClick={() => send('remove_driver', { index: i })}
                      className="mini"
                    >
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
  if (type === 'RickerWavelet' || type === 'GaussianPulse') {
    // GaussianPulse uses 'center_time' / 'width', not 'delay'/'frequency'.
    // We translate the single "delay" UI control into both shapes so the
    // user does not have to context-switch.
    if (type === 'GaussianPulse') {
      spec.center_time = delay;
      spec.width = 1.0 / Math.max(frequency, 1e-6);
    } else {
      spec.delay = delay;
    }
  }
  return spec;
}

// =====================================================================
// Canvas rendering — composes the pressure field, obstacle overlay, and
// driver markers into a single per-frame draw.
// =====================================================================

let offscreen: HTMLCanvasElement | null = null;

function drawScene(
  canvas: HTMLCanvasElement | null,
  data: SimulationUpdate,
  obs: ObstaclePayload,
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
  // Whether the obstacle payload is shaped like the current pressure view.
  // If they ever disagree (e.g. mid-rebuild) we just skip the overlay this
  // frame; the next status broadcast will line them back up.
  const obsAligned =
    obs.shape[0] === h && obs.shape[1] === w && obs.mask.length === h;

  const buf = ctx.createImageData(w, h);
  const px = buf.data;
  for (let y = 0; y < h; y++) {
    const row = grid[y];
    const obsRow = obsAligned ? obs.mask[y] : undefined;
    for (let x = 0; x < w; x++) {
      let r: number;
      let g: number;
      let b: number;
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

  if (offscreen === null) {
    offscreen = document.createElement('canvas');
  }
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

  // Driver markers — drawn in full canvas coordinates so they stay crisp
  // independent of the downsample factor. We map the *full* grid_shape
  // (not the downsampled view) so positions match exactly what the backend
  // stores. Marker is a filled disc with a thin outline so it's visible
  // against both red and blue extremes of the colormap.
  const gs = engine.grid_shape;
  if (gs && drivers.length) {
    const [fh, fw] = gs;
    for (const d of drivers) {
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
