import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface SimulationUpdate {
  grid: number[][];
  max_val: number;
  step: number;
  time: number;
}

interface EngineInfo {
  grid_shape?: number[];
  timestep?: number;
  wavespeed?: number;
  gridstep?: number;
  courant?: number;
  downsample?: number;
}

interface StatusPayload {
  is_running: boolean;
  engine: EngineInfo;
  config: Record<string, unknown>;
}

const CANVAS_PX = 600;

function App(): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<Socket | null>(null);
  const lastUpdateRef = useRef<SimulationUpdate | null>(null);
  const fpsRef = useRef<{ frames: number; last: number; fps: number }>({
    frames: 0,
    last: performance.now(),
    fps: 0,
  });

  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [engine, setEngine] = useState<EngineInfo>({});
  const [step, setStep] = useState(0);
  const [simTime, setSimTime] = useState(0);
  const [maxVal, setMaxVal] = useState(0);
  const [fps, setFps] = useState(0);

  // Establish the socket once, inside an effect, so React StrictMode can
  // cleanly tear it down on unmount.
  useEffect(() => {
    const socket = io('/', { path: '/socket.io/', transports: ['websocket', 'polling'] });
    socketRef.current = socket;

    socket.on('connect', () => setIsConnected(true));
    socket.on('disconnect', () => setIsConnected(false));
    socket.on('connect_error', (err) => console.warn('connect_error', err.message));

    socket.on('status', (payload: StatusPayload) => {
      setIsRunning(payload.is_running);
      setEngine(payload.engine || {});
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
  }, []);

  // Render loop — one paint per animation frame, regardless of how often
  // the backend sends updates. Decouples wire cadence from screen cadence.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const data = lastUpdateRef.current;
      if (data) {
        drawGrid(canvasRef.current, data);
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

  const send = useCallback((event: string, payload: unknown = {}) => {
    socketRef.current?.emit(event, payload);
  }, []);

  const handleStart = () => send('start_simulation');
  const handleStop = () => send('stop_simulation');
  const handleReset = () => send('reset_simulation');

  const cflLabel = useMemo(() => {
    const c = engine.courant;
    if (c === undefined) return '—';
    return c.toFixed(3);
  }, [engine.courant]);

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

      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={CANVAS_PX}
          height={CANVAS_PX}
          style={{ border: '1px solid #444', imageRendering: 'pixelated' }}
        />
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
        </div>
      </footer>
    </div>
  );
}

// --- Canvas rendering ------------------------------------------------------

// Cached offscreen canvas at native grid resolution; reused across frames
// because putImageData ignores canvas transforms, so we must blit through
// drawImage to scale up to the visible canvas size.
let offscreen: HTMLCanvasElement | null = null;

function drawGrid(canvas: HTMLCanvasElement | null, data: SimulationUpdate): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const { grid, max_val } = data;
  const h = grid.length;
  if (h === 0) return;
  const w = grid[0].length;
  if (w === 0) return;

  const absMax = Math.max(1e-6, max_val);

  // Build a per-pixel ImageData buffer at the grid's resolution. This is
  // dramatically faster than 2,500+ fillRect calls per frame.
  const buf = ctx.createImageData(w, h);
  const px = buf.data;
  for (let y = 0; y < h; y++) {
    const row = grid[y];
    for (let x = 0; x < w; x++) {
      const v = Math.max(-1, Math.min(1, row[x] / absMax));
      const [r, g, b] = divergingColormap(v);
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
}

// Smooth diverging blue -> white -> red colormap.
// v in [-1, 1]. White at v == 0 makes the rest field clearly readable.
function divergingColormap(v: number): [number, number, number] {
  // Endpoints chosen for high contrast on a dark UI:
  // - negative end: deep blue
  // - zero        : near-white (slight off-white so it's visible on dark bg)
  // - positive end: deep red
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
