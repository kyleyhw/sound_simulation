/**
 * Echo vision (#/echo): the room model of the toy page. A room is the loop's
 * 100 x 100 material map (0 = air, 2 = rigid), edited with rectangle drags
 * and checked against the family the learned room estimate was trained on
 * (scenarios.randomLoopScene: 1-3 axis-aligned blocks and thin partitions
 * with a door gap, kept inside `AREA`, away from the speaker bar).
 */

import { demoScenario, randomLoopScene } from '../loop/scenarios';

/** Grid side (the learned model is trained for 100 x 100 only). */
export const N = 100;
/** Material id of a drawn obstacle (rigid, as in randomLoopScene). */
export const RIGID = 2;
/** Where obstacles may go (inclusive; randomLoopScene's rowMin..rowMax, colMin..colMax). */
export const AREA = { r0: 8, r1: 74, c0: 8, c1: 91 } as const;

export type Tool = 'look' | 'block' | 'wall' | 'erase';

/** An inclusive cell rectangle. */
export interface CellRect {
  r0: number;
  r1: number;
  c0: number;
  c1: number;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Cell under a point of a square canvas of CSS size w x h (clamped to the grid). */
export function pointToCell(x: number, y: number, w: number, h: number, n = N): [number, number] {
  return [clamp(Math.floor((y / h) * n), 0, n - 1), clamp(Math.floor((x / w) * n), 0, n - 1)];
}

/** Clamp a cell into the drawable area. */
export function clampToArea([r, c]: [number, number]): [number, number] {
  return [clamp(Math.round(r), AREA.r0, AREA.r1), clamp(Math.round(c), AREA.c0, AREA.c1)];
}

/** Wall thickness in cells (the training partitions are 2 cells thick). */
export const WALL = 2;

/**
 * The rectangle a drag from `a` to `b` covers, clamped to `AREA`. A block or
 * an erase covers the box spanned by the two cells; a wall snaps to the
 * dominant axis of the drag and is `WALL` cells thick.
 */
export function dragRect(tool: Tool, a: [number, number], b: [number, number]): CellRect | null {
  if (tool === 'look') return null;
  const [ar, ac] = clampToArea(a);
  const [br, bc] = clampToArea(b);
  if (tool === 'wall') {
    if (Math.abs(br - ar) >= Math.abs(bc - ac)) {
      const c0 = Math.min(ac, AREA.c1 - WALL + 1);
      return { r0: Math.min(ar, br), r1: Math.max(ar, br), c0, c1: c0 + WALL - 1 };
    }
    const r0 = Math.min(ar, AREA.r1 - WALL + 1);
    return { r0, r1: r0 + WALL - 1, c0: Math.min(ac, bc), c1: Math.max(ac, bc) };
  }
  return { r0: Math.min(ar, br), r1: Math.max(ar, br), c0: Math.min(ac, bc), c1: Math.max(ac, bc) };
}

/** A copy of `mat` with `rect` set to `value` (RIGID to draw, 0 to erase). */
export function paintRect(mat: Uint8Array, rect: CellRect, value: number, n = N): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(mat);
  for (let i = rect.r0; i <= rect.r1; i++) for (let j = rect.c0; j <= rect.c1; j++) out[i * n + j] = value;
  return out;
}

/** A connected (4-neighbour) obstacle with its bounding box. */
export interface RoomObject extends CellRect {
  cells: number;
  /** The object fills its bounding box. */
  rectangular: boolean;
}

/** The 4-connected obstacles of a material map. */
export function roomObjects(mat: Uint8Array, n = N): RoomObject[] {
  const seen = new Uint8Array(mat.length);
  const out: RoomObject[] = [];
  for (let q0 = 0; q0 < mat.length; q0++) {
    if (!mat[q0] || seen[q0]) continue;
    const o = { r0: n, r1: -1, c0: n, c1: -1, cells: 0, rectangular: false };
    const stack = [q0];
    seen[q0] = 1;
    while (stack.length) {
      const q = stack.pop()!;
      const i = Math.floor(q / n);
      const j = q % n;
      o.cells++;
      o.r0 = Math.min(o.r0, i);
      o.r1 = Math.max(o.r1, i);
      o.c0 = Math.min(o.c0, j);
      o.c1 = Math.max(o.c1, j);
      const nb = [i > 0 ? q - n : -1, i < n - 1 ? q + n : -1, j > 0 ? q - 1 : -1, j < n - 1 ? q + 1 : -1];
      for (const qq of nb) {
        if (qq >= 0 && mat[qq] && !seen[qq]) {
          seen[qq] = 1;
          stack.push(qq);
        }
      }
    }
    o.rectangular = o.cells === (o.r1 - o.r0 + 1) * (o.c1 - o.c0 + 1);
    out.push(o);
  }
  return out;
}

/**
 * Whether a room looks like the training family: every object an
 * axis-aligned box no thicker than 12 cells and no longer than 36 (the
 * largest block is 12 x 18, the longest partition 36), and at most 6 of them
 * (3 objects, a partition with a door counting as two). Returns why not.
 */
export function familyCheck(mat: Uint8Array, n = N): { inFamily: boolean; reason: 'shape' | 'size' | 'count' | null } {
  const objs = roomObjects(mat, n);
  if (objs.some((o) => !o.rectangular)) return { inFamily: false, reason: 'shape' };
  if (objs.some((o) => Math.min(o.r1 - o.r0, o.c1 - o.c0) + 1 > 12 || Math.max(o.r1 - o.r0, o.c1 - o.c0) + 1 > 36)) return { inFamily: false, reason: 'size' };
  if (objs.length > 6) return { inFamily: false, reason: 'count' };
  return { inFamily: true, reason: null };
}

/** A random room from the training family (never a training seed: those are 1e6 + i and up). */
export function randomRoom(seed: number): Uint8Array {
  return randomLoopScene(seed, N).truth.materials;
}

export interface Example {
  id: string;
  label: string;
  /** Outside the training family (the network is expected to struggle). */
  hard?: boolean;
  build: () => Uint8Array;
}

function disc(r: number, c: number, R: number): Uint8Array {
  const m = new Uint8Array(N * N);
  for (let i = r - R; i <= r + R; i++) for (let j = c - R; j <= c + R; j++) if ((i - r) ** 2 + (j - c) ** 2 <= R * R) m[i * N + j] = RIGID;
  return m;
}

function diagonal(r: number, c: number, L: number): Uint8Array {
  const m = new Uint8Array(N * N);
  for (let k = 0; k < L; k++) for (let w = 0; w < 2; w++) m[(r + k) * N + c + k + w] = RIGID;
  return m;
}

const demo = () => demoScenario(N).epochs;

/** Named rooms: the loop demo's rooms and two shapes the network was never trained on. */
export const EXAMPLES: Example[] = [
  { id: 'box', label: 'A box', build: () => demo()[0].truth.materials.slice() },
  { id: 'box-left', label: 'A box on the left', build: () => demo()[2].truth.materials.slice() },
  { id: 'door', label: 'A partition with a door', build: () => demo()[3].truth.materials.slice() },
  { id: 'pillar', label: 'A round pillar (hard case)', hard: true, build: () => disc(42, 50, 8) },
  { id: 'diagonal', label: 'A diagonal wall (hard case)', hard: true, build: () => diagonal(24, 30, 26) },
];
