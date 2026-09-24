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
