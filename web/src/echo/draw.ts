/**
 * Echo vision (#/echo): canvas drawing for the room, the live field and the
 * result maps. Every panel is the 100 x 100 grid drawn as blocky cells at
 * `SCALE` pixels per cell, with vector overlays (cell-edge outlines, the
 * speaker bar, the drawable area) on top.
 */

import { colormapLut, type ColormapName } from '../render/colormaps';
import type { CellRect } from './room';

/** Canvas pixels per grid cell. */
export const SCALE = 8;

/** A straight run of cell edges: from (x0, y0) to (x1, y1) in cell units. */
export type Segment = [number, number, number, number];

/**
 * The outline of a mask as merged straight segments along cell edges (a
 * boundary lies between a cell in the mask and one outside it or off-grid).
 */
export function maskEdges(mask: ArrayLike<number>, rows: number, cols: number): Segment[] {
  const on = (i: number, j: number) => i >= 0 && j >= 0 && i < rows && j < cols && mask[i * cols + j] !== 0;
  const out: Segment[] = [];
  // Horizontal edges: the line y = i between rows i - 1 and i.
  for (let i = 0; i <= rows; i++) {
    let start = -1;
    for (let j = 0; j <= cols; j++) {
      const edge = j < cols && on(i - 1, j) !== on(i, j);
      if (edge && start < 0) start = j;
      if (!edge && start >= 0) {
        out.push([start, i, j, i]);
        start = -1;
      }
    }
  }
  // Vertical edges: the line x = j between columns j - 1 and j.
  for (let j = 0; j <= cols; j++) {
    let start = -1;
    for (let i = 0; i <= rows; i++) {
      const edge = i < rows && on(i, j - 1) !== on(i, j);
      if (edge && start < 0) start = i;
      if (!edge && start >= 0) {
        out.push([j, start, j, i]);
        start = -1;
      }
    }
  }
  return out;
}

export const css = (name: string, fallback: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

function rgb(color: string): [number, number, number] {
  const m = /^#([0-9a-f]{6})$/i.exec(color.trim());
  if (!m) return [128, 128, 128];
  const v = parseInt(m[1], 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** Fixed overlay colours (they sit on the dark colormaps in both themes). */
export const ESTIMATE_COLOR = '#38bdf8';
export const TRUTH_COLOR = '#4ade80';

export interface Layer {
  kind: 'room' | 'field' | 'map';
  /** 'field': signed pressure; 'map': values in [0, 1]. */
  values?: Float32Array;
  /** Obstacles to fill (room and field). */
  walls?: Uint8Array | null;
  colormap?: ColormapName;
  /** Signed field scale (the field is shown as sign(p) sqrt(|p| / scale)). */
  scale?: number;
}

export interface Overlay {
  outlines?: { mask: Uint8Array; color: string; dash?: boolean; width?: number }[];
  array?: number[][];
  /** Index into `array` of the speaker that is pinging (drawn with a ring). */
  active?: number | null;
  area?: CellRect | null;
  preview?: { rect: CellRect; erase: boolean } | null;
}

/** Draw one panel: the cell layer, then the overlays. */
export function drawPanel(cv: HTMLCanvasElement, n: number, layer: Layer, ov: Overlay = {}): void {
  const W = n * SCALE;
  if (cv.width !== W || cv.height !== W) {
    cv.width = W;
    cv.height = W;
  }
  const ctx = cv.getContext('2d');
  if (!ctx) return;
  const img = new ImageData(n, n);
  const floor = rgb(css('--panel-2', '#1b2130'));
  const wall = rgb(css('--text-2', '#a7afc2'));
  const lut = layer.colormap ? colormapLut(layer.colormap) : null;
  for (let q = 0; q < n * n; q++) {
    let c: [number, number, number] | number[] = floor;
    if (layer.kind === 'map' && layer.values && lut) {
      const k = Math.round(Math.min(1, Math.max(0, layer.values[q])) * 255) * 4;
      c = [lut[k], lut[k + 1], lut[k + 2]];
    } else if (layer.kind === 'field' && layer.values && lut) {
      const v = layer.values[q] / (layer.scale || 1);
      const s = Math.sign(v) * Math.sqrt(Math.min(1, Math.abs(v)));
      const k = Math.round((0.5 + 0.5 * s) * 255) * 4;
      c = [lut[k], lut[k + 1], lut[k + 2]];
    }
    if (layer.walls?.[q]) c = wall;
    img.data[q * 4] = c[0];
    img.data[q * 4 + 1] = c[1];
    img.data[q * 4 + 2] = c[2];
    img.data[q * 4 + 3] = 255;
  }
  // Scale the cells up without smoothing (an offscreen canvas holds the n x n image).
  const off = document.createElement('canvas');
  off.width = n;
  off.height = n;
  off.getContext('2d')!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, W, W);

  if (ov.area) {
    const a = ov.area;
    ctx.save();
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = css('--border-strong', '#37405a');
    ctx.lineWidth = 2;
    ctx.strokeRect(a.c0 * SCALE, a.r0 * SCALE, (a.c1 - a.c0 + 1) * SCALE, (a.r1 - a.r0 + 1) * SCALE);
    ctx.restore();
  }
  for (const o of ov.outlines ?? []) {
    ctx.save();
    ctx.strokeStyle = o.color;
    ctx.lineWidth = o.width ?? 5;
    ctx.lineCap = 'square';
    if (o.dash) ctx.setLineDash([10, 7]);
    ctx.beginPath();
    for (const [x0, y0, x1, y1] of maskEdges(o.mask, n, n)) {
      ctx.moveTo(x0 * SCALE, y0 * SCALE);
      ctx.lineTo(x1 * SCALE, y1 * SCALE);
    }
    ctx.stroke();
    ctx.restore();
  }
  if (ov.preview) {
    const r = ov.preview.rect;
    ctx.save();
    ctx.fillStyle = ov.preview.erase ? 'rgba(244, 63, 94, 0.25)' : 'rgba(56, 189, 248, 0.35)';
    ctx.strokeStyle = ov.preview.erase ? '#f43f5e' : ESTIMATE_COLOR;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    const [x, y, w, h] = [r.c0 * SCALE, r.r0 * SCALE, (r.c1 - r.c0 + 1) * SCALE, (r.r1 - r.r0 + 1) * SCALE];
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
    ctx.restore();
  }
  if (ov.array) {
    ctx.save();
    ctx.fillStyle = css('--driver', '#fb7185');
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.lineWidth = 1.5;
    for (const [r, c] of ov.array) {
      ctx.beginPath();
      ctx.arc((c + 0.5) * SCALE, (r + 0.5) * SCALE, SCALE * 0.95, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
    if (ov.active != null && ov.array[ov.active]) {
      const [r, c] = ov.array[ov.active];
      ctx.strokeStyle = css('--driver', '#fb7185');
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc((c + 0.5) * SCALE, (r + 0.5) * SCALE, SCALE * 2.2, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.restore();
  }
}
