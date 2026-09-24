/**
 * Room twin (plan 9.5): rebuild a measured shoebox room in the 3D FDTD
 * engine and compute its impulse response and T60, for comparison with
 * the measurement and with Sabine/Eyring.
 *
 * Wall absorption: the measured/entered random-incidence coefficient alpha
 * is mapped to the locally reacting wall admittance beta by inverting the
 * 3D random-incidence (Paris) average of the angle-dependent absorption,
 *   alpha_rand(beta) = int_0^{pi/2} (1 - |R(theta)|^2) sin(2 theta) dtheta,
 *   R(theta) = (cos theta - beta) / (cos theta + beta).
 */

import { Simulation } from '../engine/simulation';
import { decayMetrics, type Shoebox } from './measure';

export function alphaRandom3d(beta: number): number {
  const n = 2000;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const th = ((i + 0.5) / n) * (Math.PI / 2);
    const c = Math.cos(th);
    const r = (c - beta) / (c + beta);
    acc += (1 - r * r) * Math.sin(2 * th);
  }
  return (acc * (Math.PI / 2)) / n;
}

export function betaForAlpha(alpha: number): number {
  let lo = 1e-4;
  let hi = 1;
  for (let k = 0; k < 60; k++) {
    const mid = 0.5 * (lo + hi);
    if (alphaRandom3d(mid) < alpha) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

export interface TwinResult {
  ir: Float32Array;
  sampleRate: number;
  t30: number | null;
  t20: number | null;
  dx: number;
  cells: number[];
}

/**
 * Simulate the room's impulse response at a co-located source/mic position.
 * Runs in chunks via the `yieldEvery` callback so the UI stays responsive.
 */
export async function simulateShoebox(
  room: Shoebox,
  pos: [number, number, number],
  opts: { maxCells?: number; seconds?: number; onProgress?: (f: number) => void } = {},
): Promise<TwinResult> {
  const c = 343;
  const maxCells = opts.maxCells ?? 64;
  const longest = Math.max(room.lx, room.ly, room.lz);
  const dx = longest / (maxCells - 2);
  const cells = [room.lx, room.ly, room.lz].map((l) => Math.max(8, Math.round(l / dx) + 2));
  const beta = betaForAlpha(room.alpha);
  const sim = new Simulation({ dims: 3, shape: cells, c, dx, courant: 0.5, outer: 'absorb', outerBeta: beta });
  const cell = pos.map((v, a) => Math.max(1, Math.min(cells[a] - 2, Math.round(v / dx))));
  const mic = [cell[0], cell[1], Math.min(cells[2] - 2, cell[2] + 1)];
  // Band-limited pulse well inside the grid's resolved band (8 cells/lambda).
  const fmax = c / (8 * dx);
  sim.setDrivers([{ id: 'src', pos: cell, waveform: { type: 'ricker', amplitude: 1, frequency: fmax / 2.5, delay: 2.5 / fmax }, enabled: true }]);
  sim.setProbes([{ id: 'mic', pos: mic }]);
  const steps = Math.round((opts.seconds ?? 0.6) / sim.dt);
  for (let s = 0; s < steps; s++) {
    sim.step();
    if (s % 200 === 0) {
      opts.onProgress?.(s / steps);
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  const ir = sim.probeSeries('mic');
  const fs = 1 / sim.dt;
  const m = decayMetrics(ir, fs, Math.round(0.005 * fs));
  return { ir, sampleRate: fs, t30: m.t30, t20: m.t20, dx, cells };
}
