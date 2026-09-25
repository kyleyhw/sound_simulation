/**
 * Dynamic demo scenario for the closed loop (plan 8.4): a 2D room with
 * partially absorbing walls, an 8-speaker bar, and four epochs in which the
 * listener moves, an obstacle moves, and a partition with a door appears.
 */

import type { Geometry, Rect } from '../control/soundfield';
import { DEFAULT_PARAMS, type SimParams } from '../engine/simulation';
import type { Epoch } from './closedLoop';

export interface LoopScenario {
  params: SimParams;
  array: number[][];
  frequency: number;
  epochs: Epoch[];
}

function block(mat: Uint8Array, cols: number, r0: number, r1: number, c0: number, c1: number, id = 2): void {
  for (let i = r0; i <= r1; i++) for (let j = c0; j <= c1; j++) mat[i * cols + j] = id;
}

export function demoScenario(n = 100): LoopScenario {
  const s = n / 100;
  const S = (v: number) => Math.round(v * s);
  const params: SimParams = { ...DEFAULT_PARAMS, shape: [n, n], outer: 'cpml', cpmlCells: 12 };
  const geo = (fill: (m: Uint8Array) => void): Geometry => {
    const m = new Uint8Array(n * n);
    fill(m);
    return { params, materials: m, speed: null };
  };
  const array = Array.from({ length: 8 }, (_, k) => [S(86), S(31) + Math.round(k * 4 * s)]);
  const obstacleA = (m: Uint8Array) => block(m, n, S(46), S(52), S(58), S(72));
  const obstacleB = (m: Uint8Array) => block(m, n, S(50), S(56), S(20), S(34));
  const brightA: Rect = [S(18), S(20), S(30), S(32)];
  const brightB: Rect = [S(34), S(16), S(46), S(28)];
  const dark: Rect = [S(18), S(62), S(30), S(74)];
  return {
    params,
    array,
    frequency: 0.05,
    epochs: [
      { label: 'Initial room', truth: geo(obstacleA), bright: brightA, dark },
      { label: 'Listener moves', truth: geo(obstacleA), bright: brightB, dark },
      { label: 'Obstacle moves', truth: geo(obstacleB), bright: brightB, dark },
      {
        label: 'Partition with a door',
        truth: geo((m) => {
          obstacleB(m);
          block(m, n, S(8), S(40), S(48), S(49));
          block(m, n, S(22), S(28), S(48), S(49), 0); // the open door
        }),
        bright: brightB,
        dark,
      },
    ],
  };
}

/** Seeded uniform PRNG in [0, 1) (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Bright and dark zones (13 x 13 cells, as in the demo) clear of the obstacles by 2 cells, centres >= 28 cells apart. */
function pickZones(mat: Uint8Array, n: number, rnd: () => number): { bright: Rect; dark: Rect } {
  const s = n / 100;
  const S = (v: number) => Math.round(v * s);
  const ri = (a: number, b: number) => a + Math.floor(rnd() * (b - a + 1));
  const Z = S(12);
  const clear = (z: Rect) => {
    for (let i = Math.max(0, z[0] - 2); i <= Math.min(n - 1, z[2] + 2); i++) for (let j = Math.max(0, z[1] - 2); j <= Math.min(n - 1, z[3] + 2); j++) if (mat[i * n + j]) return false;
    return true;
  };
  const zone = (): Rect => {
    const r0 = ri(S(12), S(54));
    const c0 = ri(S(12), S(75));
    return [r0, c0, r0 + Z, c0 + Z];
  };
  let bright: Rect = [S(18), S(20), S(30), S(32)];
  let dark: Rect = [S(18), S(62), S(30), S(74)];
  for (let t = 0; t < 2000; t++) {
    const b = zone();
    const d = zone();
    if (!clear(b) || !clear(d)) continue;
    if (Math.hypot((b[0] + b[2] - d[0] - d[2]) / 2, (b[1] + b[3] - d[1] - d[3]) / 2) < S(28)) continue;
    bright = b;
    dark = d;
    break;
  }
  return { bright, dark };
}

/** One rigid object of a random loop scene: a block or a partition with a door. */
export type LoopObject =
  | { kind: 'block'; r0: number; r1: number; c0: number; c1: number }
  | { kind: 'partition'; axis: 0 | 1; r0: number; r1: number; c0: number; c1: number; door: [number, number] }
  | { kind: 'disc' | 'L' | 'diagonal'; r0: number; r1: number; c0: number; c1: number }; // out-of-family shapes (bounding box)

export interface RandomLoopScene {
  seed: number;
  objects: LoopObject[];
  truth: Geometry;
  bright: Rect;
  dark: Rect;
}

/**
 * A random room in the style of the demo (plan 8.4 family): the demo's
 * grid, CPML and speaker bar, with 1-3 rigid objects (blocks of varied size
 * and thin partitions with a door gap, some running from the top edge like
 * the demo's), kept at least 12 cells from the array, and a bright and a dark
 * zone (13 x 13 cells, as in the demo) that avoid the objects.
 */
export function randomLoopScene(seed: number, n = 100): RandomLoopScene {
  const rnd = mulberry32(seed * 2654435761 + 12345);
  const s = n / 100;
  const S = (v: number) => Math.round(v * s);
  const ri = (a: number, b: number) => a + Math.floor(rnd() * (b - a + 1)); // inclusive
  const params: SimParams = { ...DEFAULT_PARAMS, shape: [n, n], outer: 'cpml', cpmlCells: 12 };
  const rowMin = S(8);
  const rowMax = S(74);
  const colMin = S(8);
  const colMax = S(91);
  const occupied = new Uint8Array(n * n); // objects dilated by 3 cells
  const mat = new Uint8Array(n * n);
  const fits = (r0: number, r1: number, c0: number, c1: number) => {
    if (r0 < rowMin || r1 > rowMax || c0 < colMin || c1 > colMax) return false;
    for (let i = r0; i <= r1; i++) for (let j = c0; j <= c1; j++) if (occupied[i * n + j]) return false;
    return true;
  };
  const occupy = (r0: number, r1: number, c0: number, c1: number) => {
    for (let i = Math.max(0, r0 - 3); i <= Math.min(n - 1, r1 + 3); i++) for (let j = Math.max(0, c0 - 3); j <= Math.min(n - 1, c1 + 3); j++) occupied[i * n + j] = 1;
  };
  const objects: LoopObject[] = [];
  const count = ri(1, 3);
  let tries = 0;
  while (objects.length < count && tries++ < 200) {
    if (rnd() < 0.3) {
      const axis: 0 | 1 = rnd() < 0.6 ? 0 : 1;
      const L = ri(S(18), S(36));
      const g = ri(S(5), S(9));
      const off = ri(S(3), L - g - S(3));
      let r0: number;
      let c0: number;
      if (axis === 0) {
        r0 = rnd() < 0.5 ? rowMin : ri(rowMin, rowMax - L + 1);
        c0 = ri(colMin, colMax - 1);
      } else {
        r0 = ri(rowMin, rowMax - 1);
        c0 = ri(colMin, colMax - L + 1);
      }
      const [r1, c1] = axis === 0 ? [r0 + L - 1, c0 + 1] : [r0 + 1, c0 + L - 1];
      if (!fits(r0, r1, c0, c1)) continue;
      const door: [number, number] = axis === 0 ? [r0 + off, r0 + off + g - 1] : [c0 + off, c0 + off + g - 1];
      block(mat, n, r0, r1, c0, c1);
      if (axis === 0) block(mat, n, door[0], door[1], c0, c1, 0);
      else block(mat, n, r0, r1, door[0], door[1], 0);
      occupy(r0, r1, c0, c1);
      objects.push({ kind: 'partition', axis, r0, r1, c0, c1, door });
    } else {
      const h = ri(S(4), S(12));
      const w = ri(S(4), S(18));
      const r0 = ri(rowMin, rowMax - h + 1);
      const c0 = ri(colMin, colMax - w + 1);
      const r1 = r0 + h - 1;
      const c1 = c0 + w - 1;
      if (!fits(r0, r1, c0, c1)) continue;
      block(mat, n, r0, r1, c0, c1);
      occupy(r0, r1, c0, c1);
      objects.push({ kind: 'block', r0, r1, c0, c1 });
    }
  }
  const { bright, dark } = pickZones(mat, n, rnd);
  return { seed, objects, truth: { params, materials: mat, speed: null }, bright, dark };
}

/**
 * Shapes outside the training family (robustness check of the learned room
 * estimate; never trained on): 1-3 rigid discs, L-shapes or diagonal walls
 * on the 100 x 100 demo grid, with zones chosen as in randomLoopScene.
 */
export function oodLoopScene(seed: number): RandomLoopScene {
  const n = 100;
  const rnd = mulberry32(seed * 2654435761 + 777);
  const ri = (a: number, b: number) => a + Math.floor(rnd() * (b - a + 1));
  const params: SimParams = { ...DEFAULT_PARAMS, shape: [n, n], outer: 'cpml', cpmlCells: 12 };
  const mat = new Uint8Array(n * n);
  const occupied = new Uint8Array(n * n);
  const objects: LoopObject[] = [];
  const count = ri(1, 3);
  for (let t = 0; t < 300 && objects.length < count; t++) {
    const cells: number[] = [];
    const kind = ['disc', 'L', 'diagonal'][ri(0, 2)];
    const r0 = ri(10, 64);
    const c0 = ri(10, 80);
    if (kind === 'disc') {
      const R = ri(3, 8);
      for (let i = r0 - R; i <= r0 + R; i++) for (let j = c0 - R; j <= c0 + R; j++) if ((i - r0) ** 2 + (j - c0) ** 2 <= R * R) cells.push(i * n + j);
    } else if (kind === 'L') {
      const a = ri(8, 16);
      const b = ri(8, 16);
      const th = ri(3, 5);
      const sr = rnd() < 0.5 ? 1 : -1;
      const sc = rnd() < 0.5 ? 1 : -1;
      for (let k = 0; k < a; k++) for (let w = 0; w < th; w++) cells.push((r0 + sr * k) * n + c0 + sc * w);
      for (let k = 0; k < b; k++) for (let w = 0; w < th; w++) cells.push((r0 + sr * w) * n + c0 + sc * k);
    } else {
      const L = ri(14, 26);
      const sg = rnd() < 0.5 ? 1 : -1;
      for (let k = 0; k < L; k++) for (let w = 0; w < 2; w++) cells.push((r0 + k) * n + c0 + sg * (k + w));
    }
    const ok = cells.every((q) => {
      const i = Math.floor(q / n);
      const j = q % n;
      return i >= 8 && i <= 74 && j >= 8 && j <= 91 && !occupied[q];
    });
    if (!ok) continue;
    let r1 = -1;
    let c1 = -1;
    let rr0 = n;
    let cc0 = n;
    for (const q of cells) {
      mat[q] = 2;
      const i = Math.floor(q / n);
      const j = q % n;
      [rr0, r1, cc0, c1] = [Math.min(rr0, i), Math.max(r1, i), Math.min(cc0, j), Math.max(c1, j)];
      for (let a = Math.max(0, i - 3); a <= Math.min(n - 1, i + 3); a++) for (let b = Math.max(0, j - 3); b <= Math.min(n - 1, j + 3); b++) occupied[a * n + b] = 1;
    }
    objects.push({ kind: kind as 'disc' | 'L' | 'diagonal', r0: rr0, r1, c0: cc0, c1 });
  }
  const { bright, dark } = pickZones(mat, n, rnd);
  return { seed, objects, truth: { params, materials: mat, speed: null }, bright, dark };
}
