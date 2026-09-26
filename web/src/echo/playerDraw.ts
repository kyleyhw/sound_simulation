/**
 * Echo vision (#/echo): canvas drawing for the playback. Everything draws
 * into canvases sized to their CSS box x devicePixelRatio; the 100 x 100
 * cell layers are drawn once into a small offscreen canvas and scaled up
 * smoothly, and the geometry (walls, speakers, rings, outlines) is vector on
 * top, so the waves look smooth and the edges stay crisp. Nothing here
 * allocates per frame except what the canvas API itself does.
 */

import { colormapLut, type ColormapName } from '../render/colormaps';
import { maskEdges } from './draw';

/** A canvas with an n x n ImageData, for one cell layer. */
export class CellLayer {
  readonly canvas: HTMLCanvasElement;
  readonly img: ImageData;
  constructor(readonly n: number) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = n;
    this.canvas.height = n;
    this.img = new ImageData(n, n);
  }
  commit(): HTMLCanvasElement {
    this.canvas.getContext('2d')!.putImageData(this.img, 0, 0);
    return this.canvas;
  }
}

/** The colour of zero in a diverging map (the playback background). */
export function zeroColor(map: ColormapName): string {
  const l = colormapLut(map);
  return `rgb(${l[512]}, ${l[513]}, ${l[514]})`;
}

/** Display exponent of the scattered field: sign(v) (|v| / P)^DISPLAY_GAMMA against the display peak P. */
export const DISPLAY_GAMMA = 0.8;

const tabA = new Float32Array(256);
const tabB = new Float32Array(256);
/** Display value of every 8-bit code of a frame whose own peak is `gain` x the display peak. */
function codeTable(tab: Float32Array, gain: number, frameGamma: number) {
  for (let c = -128; c < 128; c++) {
    const a = Math.min(1, (Math.abs(c) / 127) ** (1 / frameGamma) * gain);
    tab[c + 128] = Math.sign(c) * a ** DISPLAY_GAMMA;
  }
}

/**
 * Write a scattered-field frame into `layer`, blending frames `fa` and `fb`
 * (weight w on fb). Each frame is gamma-coded against its own peak S
 * (sense.ts encodeFrame); `gain` is S / P, P the display peak.
 */
export function paintFrame(layer: CellLayer, frames: Int8Array, fa: number, fb: number, w: number, gainA: number, gainB: number, map: ColormapName, frameGamma: number): HTMLCanvasElement {
  const n2 = layer.n * layer.n;
  const lut = colormapLut(map);
  const d = layer.img.data;
  const oa = fa * n2;
  const ob = fb * n2;
  codeTable(tabA, gainA, frameGamma);
  codeTable(tabB, gainB, frameGamma);
  for (let q = 0; q < n2; q++) {
    const v = (1 - w) * tabA[frames[oa + q] + 128] + w * tabB[frames[ob + q] + 128];
    const k = Math.round((0.5 + 0.5 * v) * 255) * 4;
    const o = q * 4;
    d[o] = lut[k];
    d[o + 1] = lut[k + 1];
    d[o + 2] = lut[k + 2];
    d[o + 3] = 255;
  }
  return layer.commit();
}

/** Write values in [0, 1] into `layer` with a sequential map. */
export function paintMap(layer: CellLayer, values: Float32Array, map: ColormapName): HTMLCanvasElement {
  const lut = colormapLut(map);
  const d = layer.img.data;
  for (let q = 0; q < values.length; q++) {
    const k = Math.round(Math.min(1, Math.max(0, values[q])) * 255) * 4;
    d[q * 4] = lut[k];
    d[q * 4 + 1] = lut[k + 1];
    d[q * 4 + 2] = lut[k + 2];
    d[q * 4 + 3] = 255;
  }
  return layer.commit();
}

/**
 * An energy image as display amplitude: sqrt(I / (headroom x max)), clipped
 * to 1, the max taken away from the device's near field. A headroom above 1
 * leaves the image dimmer (the picture of the first pings, which grows).
 */
export function amplitudeImage(image: Float32Array, near: Uint8Array, headroom = 1): Float32Array {
  let max = 0;
  for (let q = 0; q < image.length; q++) if (!near[q]) max = Math.max(max, image[q]);
  const out = new Float32Array(image.length);
  if (max > 0) for (let q = 0; q < image.length; q++) out[q] = Math.min(1, Math.sqrt(Math.max(0, image[q]) / (headroom * max)));
  return out;
}

/** Values in [0, 1] as one colour with that opacity (an overlay). */
export function paintTint(layer: CellLayer, values: Float32Array, [r, g, b]: [number, number, number]): HTMLCanvasElement {
  const d = layer.img.data;
  for (let q = 0; q < values.length; q++) {
    d[q * 4] = r;
    d[q * 4 + 1] = g;
    d[q * 4 + 2] = b;
    d[q * 4 + 3] = Math.round(255 * Math.min(1, Math.max(0, values[q])));
  }
  return layer.commit();
}

/** The obstacle cells as one path in cell units (row runs merged). */
export function wallPath(mask: Uint8Array, n: number): Path2D {
  const p = new Path2D();
  for (let i = 0; i < n; i++) {
    let j = 0;
    while (j < n) {
      if (!mask[i * n + j]) {
        j++;
        continue;
      }
      const j0 = j;
      while (j < n && mask[i * n + j]) j++;
      p.rect(j0, i, j - j0, 1);
    }
  }
  return p;
}

/** A mask's outline as one path in cell units. */
export function outlinePath(mask: Uint8Array, n: number): Path2D {
  const p = new Path2D();
  for (const [x0, y0, x1, y1] of maskEdges(mask, n, n)) {
    p.moveTo(x0, y0);
    p.lineTo(x1, y1);
  }
  return p;
}

/** Size a canvas's backing store to its CSS box (x devicePixelRatio); returns [w, h] in device pixels. */
export function fitCanvas(cv: HTMLCanvasElement, square: boolean): [number, number] {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(cv.clientWidth * dpr));
  const h = square ? w : Math.max(1, Math.round(cv.clientHeight * dpr));
  if (cv.width !== w || cv.height !== h) {
    cv.width = w;
    cv.height = h;
  }
  return [w, h];
}

/** Draw a cell layer over the whole square canvas, smoothed. */
export function blit(ctx: CanvasRenderingContext2D, src: CanvasImageSource, W: number, alpha = 1) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'low';
  ctx.drawImage(src, 0, 0, W, W);
  ctx.restore();
}

export function fillPath(ctx: CanvasRenderingContext2D, path: Path2D, cell: number, color: string, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.scale(cell, cell);
  ctx.fillStyle = color;
  ctx.fill(path);
  ctx.restore();
}

export function strokePath(ctx: CanvasRenderingContext2D, path: Path2D, cell: number, color: string, width: number, alpha = 1, dash: number[] | null = null) {
  // Scale the path, not the context, so the line width stays in device pixels.
  const p = new Path2D();
  p.addPath(path, new DOMMatrix().scale(cell));
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'square';
  if (dash) ctx.setLineDash(dash);
  ctx.stroke(p);
  ctx.restore();
}

/**
 * The direct click as a thin expanding ring around each emitting speaker
 * (radius in cells), with a faint band about half a wavelength wide.
 */
export function drawRing(ctx: CanvasRenderingContext2D, cell: number, centres: number[][], radius: number, band: number, color: string) {
  if (radius <= 0) return;
  const fade = Math.min(1, 6 / Math.sqrt(radius + 4));
  ctx.save();
  ctx.strokeStyle = color;
  for (const [r, c] of centres) {
    const x = (c + 0.5) * cell;
    const y = (r + 0.5) * cell;
    ctx.globalAlpha = 0.09 * fade;
    ctx.lineWidth = band * cell;
    ctx.beginPath();
    ctx.arc(x, y, radius * cell, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.globalAlpha = 0.55 * fade;
    ctx.lineWidth = Math.max(1, cell * 0.28);
    ctx.setLineDash([cell * 0.9, cell * 0.7]);
    ctx.beginPath();
    ctx.arc(x, y, radius * cell, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

/**
 * The device: speakers as dots (emitting ones ringed), microphones that are
 * not speakers as small squares, and a halo on each microphone whose size
 * follows the echo arriving there (`level` in [0, 1] per mic).
 */
export function drawDevice(
  ctx: CanvasRenderingContext2D,
  cell: number,
  speakers: number[][],
  mics: number[][],
  active: number[],
  level: ArrayLike<number> | null,
  { driver, ink }: { driver: string; ink: string },
) {
  ctx.save();
  if (level) {
    ctx.strokeStyle = ink;
    for (let m = 0; m < mics.length; m++) {
      const a = Math.min(1, level[m]);
      if (a < 0.1) continue;
      const [r, c] = mics[m];
      ctx.globalAlpha = 0.85 * a;
      ctx.lineWidth = Math.max(1.5, cell * 0.35);
      ctx.beginPath();
      ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, cell * (1.4 + 2.2 * a), 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = driver;
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.lineWidth = Math.max(1, cell * 0.18);
  const isSpeaker = new Set(speakers.map((p) => `${p[0]},${p[1]}`));
  for (const [r, c] of speakers) {
    ctx.beginPath();
    ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, cell * 0.95, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
  for (const [r, c] of mics) {
    if (isSpeaker.has(`${r},${c}`)) continue;
    ctx.fillRect((c + 0.1) * cell, (r + 0.1) * cell, cell * 0.8, cell * 0.8);
  }
  ctx.strokeStyle = driver;
  ctx.lineWidth = Math.max(2, cell * 0.38);
  for (const s of active) {
    const [r, c] = speakers[s];
    ctx.beginPath();
    ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, cell * 2.3, 0, 2 * Math.PI);
    ctx.stroke();
  }
  ctx.restore();
}

/** Compression of the echo traces and the echo halos. */
export const TRACE_GAMMA = 0.6;

/**
 * The echo-only recordings of one emission as stacked traces (one lane per
 * microphone), over steps [from, to]. Amplitudes are compressed
 * (|v|^TRACE_GAMMA against the loudest echo), so faint echoes show. Drawn into `cv` at its full size.
 */
export function drawTraces(cv: HTMLCanvasElement, residual: Float32Array, mics: number, steps: number, from: number, to: number, own: number[], color: string, ownColor: string) {
  const ctx = cv.getContext('2d')!;
  const W = cv.width;
  const H = cv.height;
  ctx.clearRect(0, 0, W, H);
  let R = 0;
  for (let m = 0; m < mics; m++) for (let k = from; k <= to; k++) R = Math.max(R, Math.abs(residual[m * steps + k]));
  if (!(R > 0)) R = 1;
  const lane = H / mics;
  const amp = lane * 0.8;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  ctx.lineWidth = 1.2 * dpr;
  ctx.lineJoin = 'round';
  for (let m = 0; m < mics; m++) {
    const y0 = lane * (m + 0.5);
    ctx.strokeStyle = own.includes(m) ? ownColor : color;
    ctx.beginPath();
    for (let k = from; k <= to; k++) {
      const v = residual[m * steps + k] / R;
      const x = ((k - from) / (to - from)) * W;
      const y = y0 - Math.sign(v) * Math.abs(v) ** TRACE_GAMMA * amp;
      if (k === from) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  return R;
}
