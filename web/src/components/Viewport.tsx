import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MATERIAL_NAMES } from '../engine/simulation';
import { type Cell, discCells, ellipseCells, lineCells, rectCells } from '../lib/geometry';
import { colormapCss, SEQUENTIAL } from '../render/colormaps';
import { FieldRenderer } from '../render/fieldRenderer';
import type { FrameStats } from '../state/runtime';
import { useApp } from '../state/store';

/** Geometry of the displayed 2D plane (a 3D grid shows one slice). */
interface Plane {
  rows: number;
  cols: number;
  /** Map a displayed (row, col) to a full grid position. */
  toPos: (r: number, c: number) => number[];
  /** Map a full grid position into the plane, or null if off-plane. */
  fromPos: (pos: number[]) => Cell | null;
  /** Flat grid index of a displayed cell. */
  toIndex: (r: number, c: number) => number;
}

function usePlane(): Plane {
  const scene = useApp((s) => s.scene);
  const axis = useApp((s) => s.view.sliceAxis);
  const index = useApp((s) => s.view.sliceIndex);
  return useMemo(() => {
    const [nx, ny, nz] = [scene.params.shape[0], scene.params.shape[1], scene.params.shape[2] ?? 1];
    if (scene.params.dims === 2) {
      return {
        rows: nx,
        cols: ny,
        toPos: (r, c) => [r, c],
        fromPos: (p) => [p[0], p[1]],
        toIndex: (r, c) => r * ny + c,
      };
    }
    const s = Math.max(0, Math.min(index, [nx, ny, nz][axis] - 1));
    if (axis === 2)
      return {
        rows: nx,
        cols: ny,
        toPos: (r, c) => [r, c, s],
        fromPos: (p) => (p[2] === s ? [p[0], p[1]] : null),
        toIndex: (r, c) => (r * ny + c) * nz + s,
      };
    if (axis === 1)
      return {
        rows: nx,
        cols: nz,
        toPos: (r, c) => [r, s, c],
        fromPos: (p) => (p[1] === s ? [p[0], p[2]] : null),
        toIndex: (r, c) => (r * ny + s) * nz + c,
      };
    return {
      rows: ny,
      cols: nz,
      toPos: (r, c) => [s, r, c],
      fromPos: (p) => (p[0] === s ? [p[1], p[2]] : null),
      toIndex: (r, c) => (s * ny + r) * nz + c,
    };
  }, [scene.params.shape, scene.params.dims, axis, index]);
}

function extract(src: Float32Array | Uint8Array, plane: Plane, out: Float32Array | Uint8Array): void {
  for (let r = 0; r < plane.rows; r++) for (let c = 0; c < plane.cols; c++) out[r * plane.cols + c] = src[plane.toIndex(r, c)];
}

export function Viewport() {
  const runtime = useApp((s) => s.runtime);
  const scene = useApp((s) => s.scene);
  const view = useApp((s) => s.view);
  const theme = useApp((s) => s.theme);
  const tool = useApp((s) => s.tool);
  const brushSize = useApp((s) => s.brushSize);
  const selected = useApp((s) => s.selected);
  const zones = useApp((s) => s.zones);
  const zoneKind = useApp((s) => s.zoneKind);
  const plane = usePlane();

  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const arrowRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<FieldRenderer | null>(null);
  const [size, setSize] = useState({ w: 400, h: 400 });
  const [hover, setHover] = useState<Cell | null>(null);
  const [preview, setPreview] = useState<{ a: Cell; b: Cell } | null>(null);
  const [stats, setStats] = useState<FrameStats>(() => runtime.stats());
  const scaleRef = useRef(1);
  const dragRef = useRef<{ kind: 'paint' | 'shape' | 'move'; last?: Cell; start?: Cell; marker?: { kind: 'driver' | 'probe'; id: string } } | null>(null);

  // Fit the canvas to the available area, preserving the grid aspect.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const ro = new ResizeObserver(() => {
      const bw = host.clientWidth - 28;
      const bh = host.clientHeight - 28;
      const aspect = plane.cols / plane.rows;
      let w = bw;
      let h = w / aspect;
      if (h > bh) {
        h = bh;
        w = h * aspect;
      }
      setSize({ w: Math.max(120, Math.floor(w)), h: Math.max(120, Math.floor(h)) });
    });
    ro.observe(host);
    return () => ro.disconnect();
  }, [plane.cols, plane.rows]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!rendererRef.current) rendererRef.current = new FieldRenderer(canvas);
  }, []);

  useEffect(() => runtime.subscribe(setStats), [runtime]);

  // Per-frame draw.
  const buffers = useMemo(
    () => ({ field: new Float32Array(plane.rows * plane.cols), mat: new Uint8Array(plane.rows * plane.cols) }),
    [plane.rows, plane.cols],
  );
  const draw = useCallback(() => {
    const r = rendererRef.current;
    const canvas = canvasRef.current;
    if (!r || !canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(size.w * dpr);
    const H = Math.round(size.h * dpr);
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }
    const sim = runtime.sim;
    let src: Float32Array = runtime.displayField();
    let signed = true;
    if (view.overlay === 'rms' && runtime.scrubIndex === null) {
      const rms = sim.rmsMap();
      if (rms) {
        src = rms;
        signed = false;
      }
    }
    if (src.length !== sim.n) return;
    extract(src, plane, buffers.field);
    extract(sim.material, plane, buffers.mat);
    let peak = 0;
    for (let i = 0; i < buffers.field.length; i++) {
      const v = Math.abs(buffers.field[i]);
      if (v > peak) peak = v;
    }
    if (view.autoScale) {
      const target = Math.max(peak, 1e-9);
      scaleRef.current = target > scaleRef.current ? target : Math.max(target, scaleRef.current * 0.96);
    } else scaleRef.current = view.fixedScale;
    const cmap = signed ? view.colormap : SEQUENTIAL.includes(view.colormap) ? view.colormap : 'magma';
    r.draw(buffers.field, plane.rows, plane.cols, buffers.mat, {
      colormap: cmap,
      mode: view.mode,
      scale: scaleRef.current,
      dbRange: view.dbRange,
      signed,
      showMaterials: view.showMaterials,
      theme,
    });
    drawArrows();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runtime, view, plane, buffers, size, theme]);

  const drawArrows = () => {
    const c = arrowRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(size.w * dpr);
    c.height = Math.round(size.h * dpr);
    ctx.clearRect(0, 0, c.width, c.height);
    const sim = runtime.sim;
    // c(x) regions: translucent tint (blue = slower, amber = faster).
    const speed = scene.speed;
    if (speed && view.showMaterials) {
      const img = ctx.createImageData(plane.cols, plane.rows);
      for (let r = 0; r < plane.rows; r++)
        for (let cc = 0; cc < plane.cols; cc++) {
          const v = speed[plane.toIndex(r, cc)];
          if (v === 1) continue;
          const o = (r * plane.cols + cc) * 4;
          const slow = v < 1;
          img.data[o] = slow ? 90 : 245;
          img.data[o + 1] = slow ? 170 : 180;
          img.data[o + 2] = slow ? 255 : 60;
          img.data[o + 3] = Math.min(150, 40 + Math.abs(1 - v) * 220);
        }
      const off = document.createElement('canvas');
      off.width = plane.cols;
      off.height = plane.rows;
      off.getContext('2d')!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0, c.width, c.height);
    }
    if (view.overlay !== 'intensity' || !sim.intensity || scene.params.dims !== 2) return;
    const [Ir, Ic] = sim.intensity;
    const stride = Math.max(4, Math.round(plane.cols / 28));
    let maxMag = 0;
    for (let r = stride / 2; r < plane.rows; r += stride)
      for (let cc = stride / 2; cc < plane.cols; cc += stride) {
        const idx = plane.toIndex(Math.floor(r), Math.floor(cc));
        maxMag = Math.max(maxMag, Math.hypot(Ir[idx], Ic[idx]));
      }
    if (maxMag <= 0) return;
    const sx = c.width / plane.cols;
    const sy = c.height / plane.rows;
    ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.85)' : 'rgba(10,20,40,0.85)';
    ctx.lineWidth = 1.3 * dpr;
    for (let r = stride / 2; r < plane.rows; r += stride)
      for (let cc = stride / 2; cc < plane.cols; cc += stride) {
        const idx = plane.toIndex(Math.floor(r), Math.floor(cc));
        const m = Math.hypot(Ir[idx], Ic[idx]) / maxMag;
        if (m < 0.04) continue;
        const len = stride * 0.9 * Math.sqrt(m);
        const ang = Math.atan2(Ir[idx], Ic[idx]);
        const x0 = cc * sx;
        const y0 = r * sy;
        const x1 = x0 + Math.cos(ang) * len * sx;
        const y1 = y0 + Math.sin(ang) * len * sy;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        const h = 3.5 * dpr;
        ctx.lineTo(x1 - h * Math.cos(ang - 0.45), y1 - h * Math.sin(ang - 0.45));
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - h * Math.cos(ang + 0.45), y1 - h * Math.sin(ang + 0.45));
        ctx.stroke();
      }
  };

  useEffect(() => {
    draw();
    return runtime.onFrame(draw);
  }, [runtime, draw]);

  // ---------------------------------------------------------------- //
  // Pointer interaction
  // ---------------------------------------------------------------- //

  const cellAt = (e: React.PointerEvent): Cell | null => {
    const el = canvasRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const c = Math.floor(((e.clientX - rect.left) / rect.width) * plane.cols);
    const r = Math.floor(((e.clientY - rect.top) / rect.height) * plane.rows);
    if (r < 0 || c < 0 || r >= plane.rows || c >= plane.cols) return null;
    return [r, c];
  };

  const pickMarker = (cell: Cell): { kind: 'driver' | 'probe'; id: string } | null => {
    const tol = Math.max(3, plane.cols / 40);
    let best: { kind: 'driver' | 'probe'; id: string; d: number } | null = null;
    for (const d of scene.drivers) {
      const q = plane.fromPos(d.pos);
      if (!q) continue;
      const dist = Math.hypot(q[0] - cell[0], q[1] - cell[1]);
      if (dist <= tol && (!best || dist < best.d)) best = { kind: 'driver', id: d.id, d: dist };
    }
    for (const p of scene.probes) {
      const q = plane.fromPos(p.pos);
      if (!q) continue;
      const dist = Math.hypot(q[0] - cell[0], q[1] - cell[1]);
      if (dist <= tol && (!best || dist < best.d)) best = { kind: 'probe', id: p.id, d: dist };
    }
    return best ? { kind: best.kind, id: best.id } : null;
  };

  const paint = (cells: Cell[], material: number) => {
    useApp.getState().paintCells(
      cells.map(([r, c]) => plane.toIndex(r, c)),
      material,
    );
  };

  const onPointerDown = (e: React.PointerEvent) => {
    const cell = cellAt(e);
    if (!cell) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const st = useApp.getState();
    const marker = pickMarker(cell);
    if (tool === 'select' || ((tool === 'driver' || tool === 'probe') && marker)) {
      if (marker) {
        st.select(marker);
        st.setInspectorTab(marker.kind === 'driver' ? 'sources' : 'probes');
        st.checkpoint();
        dragRef.current = { kind: 'move', marker };
      } else st.select(null);
      return;
    }
    if (tool === 'driver') {
      st.addDriver(plane.toPos(cell[0], cell[1]));
      st.setInspectorTab('sources');
      return;
    }
    if (tool === 'probe') {
      st.addProbe(plane.toPos(cell[0], cell[1]));
      st.setInspectorTab('probes');
      return;
    }
    if (tool === 'speed') {
      st.checkpoint();
      st.paintSpeedCells(discCells(cell[0], cell[1], brushSize, plane.rows, plane.cols).map(([r, c]) => plane.toIndex(r, c)), st.paintSpeed);
      dragRef.current = { kind: 'paint', last: cell };
      return;
    }
    if (tool === 'brush' || tool === 'eraser') {
      st.checkpoint();
      const m = tool === 'eraser' ? 0 : st.paintMaterial;
      paint(discCells(cell[0], cell[1], brushSize, plane.rows, plane.cols), m);
      dragRef.current = { kind: 'paint', last: cell };
      return;
    }
    dragRef.current = { kind: 'shape', start: cell };
    setPreview({ a: cell, b: cell });
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const cell = cellAt(e);
    setHover(cell);
    const drag = dragRef.current;
    if (!drag || !cell) return;
    const st = useApp.getState();
    if (drag.kind === 'paint' && drag.last && tool === 'speed') {
      st.paintSpeedCells(lineCells(drag.last, cell, brushSize, plane.rows, plane.cols).map(([r, c]) => plane.toIndex(r, c)), st.paintSpeed);
      drag.last = cell;
    } else if (drag.kind === 'paint' && drag.last) {
      const m = tool === 'eraser' ? 0 : st.paintMaterial;
      paint(lineCells(drag.last, cell, brushSize, plane.rows, plane.cols), m);
      drag.last = cell;
    } else if (drag.kind === 'shape' && drag.start) {
      setPreview({ a: drag.start, b: cell });
    } else if (drag.kind === 'move' && drag.marker) {
      st.moveMarker(drag.marker.kind, drag.marker.id, plane.toPos(cell[0], cell[1]));
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (drag?.kind === 'shape' && drag.start && tool === 'zone') {
      const end = cellAt(e) ?? preview?.b ?? drag.start;
      const st = useApp.getState();
      if (scene.params.dims === 2)
        st.setZone(st.zoneKind, [Math.min(drag.start[0], end[0]), Math.min(drag.start[1], end[1]), Math.max(drag.start[0], end[0]), Math.max(drag.start[1], end[1])]);
      st.setInspectorTab('control');
      setPreview(null);
      return;
    }
    if (drag?.kind === 'shape' && drag.start) {
      const end = cellAt(e) ?? preview?.b ?? drag.start;
      const st = useApp.getState();
      const t = Math.max(1, Math.round(brushSize / 2));
      let cells: Cell[] = [];
      if (tool === 'line') cells = lineCells(drag.start, end, brushSize, plane.rows, plane.cols);
      else if (tool === 'rect') cells = rectCells(drag.start, end, t, e.shiftKey, plane.rows, plane.cols);
      else if (tool === 'ellipse') cells = ellipseCells(drag.start, end, t, e.shiftKey, plane.rows, plane.cols);
      if (cells.length) {
        st.checkpoint();
        paint(cells, st.paintMaterial);
      }
      setPreview(null);
    }
  };

  // ---------------------------------------------------------------- //
  // Overlay (markers, previews, brush cursor)
  // ---------------------------------------------------------------- //

  const mk = Math.max(1.6, plane.cols / 55);
  const vb = `0 0 ${plane.cols} ${plane.rows}`;
  const scaleLabel = scaleRef.current;
  const isRms = view.overlay === 'rms';
  const legendCss = colormapCss(isRms ? (SEQUENTIAL.includes(view.colormap) ? view.colormap : 'magma') : view.colormap);

  const cursor =
    tool === 'select' ? 'default' : tool === 'driver' || tool === 'probe' ? 'copy' : 'crosshair';

  return (
    <div className="viewport" ref={hostRef}>
      <div className="canvas-wrap" style={{ width: size.w, height: size.h, cursor }} data-testid="field">
        <canvas
          ref={canvasRef}
          aria-label="Pressure field"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={() => setHover(null)}
        />
        <canvas className="overlay" ref={arrowRef} />
        <svg className="overlay" viewBox={vb} preserveAspectRatio="none">
          {scene.drivers.map((d) => {
            const q = plane.fromPos(d.pos);
            if (!q) return null;
            const sel = selected?.kind === 'driver' && selected.id === d.id;
            return (
              <g key={d.id} opacity={d.enabled ? 1 : 0.4} data-testid="driver-marker">
                <circle cx={q[1] + 0.5} cy={q[0] + 0.5} r={mk * (sel ? 1.5 : 1.1)} fill="none" stroke="var(--driver)" strokeWidth={mk * 0.35} />
                <circle cx={q[1] + 0.5} cy={q[0] + 0.5} r={mk * 0.45} fill="var(--driver)" />
              </g>
            );
          })}
          {scene.probes.map((p) => {
            const q = plane.fromPos(p.pos);
            if (!q) return null;
            const sel = selected?.kind === 'probe' && selected.id === p.id;
            const x = q[1] + 0.5;
            const y = q[0] + 0.5;
            const s = mk * (sel ? 1.5 : 1.15);
            return (
              <g key={p.id} data-testid="probe-marker">
                <path
                  d={`M ${x} ${y - s} L ${x + s * 0.9} ${y + s * 0.7} L ${x - s * 0.9} ${y + s * 0.7} Z`}
                  fill="none"
                  stroke="var(--probe)"
                  strokeWidth={mk * 0.35}
                />
                <circle cx={x} cy={y} r={mk * 0.3} fill="var(--probe)" />
              </g>
            );
          })}
          {scene.params.dims === 2 &&
            (['bright', 'dark'] as const).map((k) => {
              const z = zones[k];
              if (!z) return null;
              return (
                <rect
                  key={k}
                  data-testid={`zone-${k}`}
                  x={z[1]}
                  y={z[0]}
                  width={z[3] - z[1] + 1}
                  height={z[2] - z[0] + 1}
                  fill={k === 'bright' ? 'rgba(255, 196, 0, 0.12)' : 'rgba(80, 160, 255, 0.12)'}
                  stroke={k === 'bright' ? '#ffc400' : '#50a0ff'}
                  strokeWidth={Math.max(0.5, plane.cols / 300)}
                  strokeDasharray="3 2"
                />
              );
            })}
          {preview && tool === 'zone' && (
            <rect
              x={Math.min(preview.a[1], preview.b[1])}
              y={Math.min(preview.a[0], preview.b[0])}
              width={Math.abs(preview.b[1] - preview.a[1]) + 1}
              height={Math.abs(preview.b[0] - preview.a[0]) + 1}
              fill="none"
              stroke={zoneKind === 'bright' ? '#ffc400' : '#50a0ff'}
              strokeWidth={Math.max(0.6, plane.cols / 250)}
              strokeDasharray="2 1.5"
            />
          )}
          {preview && (tool === 'rect' || tool === 'ellipse' || tool === 'line') && (
            <g fill="none" stroke="var(--accent)" strokeWidth={Math.max(0.6, plane.cols / 250)} strokeDasharray="2 1.5">
              {tool === 'line' && <line x1={preview.a[1] + 0.5} y1={preview.a[0] + 0.5} x2={preview.b[1] + 0.5} y2={preview.b[0] + 0.5} />}
              {tool === 'rect' && (
                <rect
                  x={Math.min(preview.a[1], preview.b[1])}
                  y={Math.min(preview.a[0], preview.b[0])}
                  width={Math.abs(preview.b[1] - preview.a[1]) + 1}
                  height={Math.abs(preview.b[0] - preview.a[0]) + 1}
                />
              )}
              {tool === 'ellipse' && (
                <ellipse
                  cx={(preview.a[1] + preview.b[1]) / 2 + 0.5}
                  cy={(preview.a[0] + preview.b[0]) / 2 + 0.5}
                  rx={Math.abs(preview.b[1] - preview.a[1]) / 2 + 0.5}
                  ry={Math.abs(preview.b[0] - preview.a[0]) / 2 + 0.5}
                />
              )}
            </g>
          )}
          {hover && (tool === 'brush' || tool === 'eraser' || tool === 'speed') && (
            <circle
              cx={hover[1] + 0.5}
              cy={hover[0] + 0.5}
              r={Math.max(0.5, brushSize - 0.5)}
              fill="none"
              stroke={tool === 'eraser' ? 'var(--danger)' : 'var(--accent)'}
              strokeWidth={Math.max(0.4, plane.cols / 300)}
            />
          )}
        </svg>
      </div>

      {stats.unstable && (
        <div className="banner" role="alert">
          The simulation became unstable (CFL violated or runaway source). Reset, or lower the Courant number.
        </div>
      )}

      <div className="hud mono" data-testid="hud">
        <span>step {stats.step}</span>
        <span>t = {formatTime(stats.time, scene.units)}</span>
        <span>{stats.running ? `${stats.fps.toFixed(0)} fps · ${formatRate(stats.stepsPerSecond)}` : stats.scrub !== null ? 'scrubbing' : 'paused'}</span>
        {hover && (
          <span className="dim">
            [{plane.toPos(hover[0], hover[1]).join(', ')}]
            {scene.materials[plane.toIndex(hover[0], hover[1])] ? ` ${MATERIAL_NAMES[scene.materials[plane.toIndex(hover[0], hover[1])]]}` : ''}
          </span>
        )}
      </div>

      <div className="legend" aria-label="Colour scale">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <span>{isRms ? 'RMS pressure' : 'Pressure'}</span>
          <span className="dim">{view.mode === 'db' ? `dB, ${view.dbRange} dB range` : view.autoScale ? 'auto' : 'fixed'}</span>
        </div>
        <div className="bar" style={{ background: legendCss }} />
        <div className="ticks mono">
          {view.mode === 'db' ? (
            <>
              <span>{isRms ? `−${view.dbRange}` : '−0 dB'}</span>
              <span>{isRms ? '' : `−${view.dbRange}`}</span>
              <span>0 dB</span>
            </>
          ) : (
            <>
              <span>{isRms ? '0' : `−${fmt(scaleLabel)}`}</span>
              <span>{isRms ? fmt(scaleLabel / 2) : '0'}</span>
              <span>{fmt(scaleLabel)}</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 100 || a < 0.01) return v.toExponential(1);
  return v.toPrecision(2);
}

export function formatTime(t: number, units: 'grid' | 'si'): string {
  if (units === 'si') {
    if (t < 1e-3) return `${(t * 1e6).toFixed(1)} µs`;
    if (t < 1) return `${(t * 1e3).toFixed(2)} ms`;
    return `${t.toFixed(3)} s`;
  }
  return t.toFixed(1);
}

function formatRate(sps: number): string {
  return sps >= 1000 ? `${(sps / 1000).toFixed(1)}k steps/s` : `${sps.toFixed(0)} steps/s`;
}
