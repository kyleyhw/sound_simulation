/**
 * Sound-field control in the browser (plan 7.7): measure speaker-to-zone
 * transfer functions with the FDTD engine, design array weights, and apply
 * them as per-driver gain and delay. It mirrors the Python `control/`
 * package on a single frequency.
 *
 * Transfer functions. Each speaker s in turn plays cos(w t). After the
 * transient has left (or decayed in) the room, the steady-state pressure at
 * point m is Re(H_ms e^{jwt}). H_ms is read by a DFT over a whole number of
 * periods, H = (2/W) sum_t p(t) e^{-jwt}.
 *
 * Weights. A complex weight w_s = g_s e^{-j w d_s} drives speaker s with
 * gain g_s and delay d_s, because cos(w(t - d)) = Re(e^{-jwd} e^{jwt}). The
 * zone pressures are then p = H w. The designs are:
 *   delay-and-sum  w_s = e^{+jk r_s}   (free-field alignment at the bright centre)
 *   focus          w = conj(h_c)       (matched filter / time reversal to the bright centre)
 *   pressure match (H^H H + lambda I) w = H_b^H 1   (bright zone to unit amplitude, dark to 0)
 *   contrast (ACC) w = argmax (w^H R_b w)/(w^H (R_d + delta I) w), the principal
 *                  generalised eigenvector (Choi & Kim 2002)
 * Every design is scaled to the same array effort, ||w||^2 = N.
 * Contrast = 10 log10(mean_bright |p|^2 / mean_dark |p|^2).
 */

import { type DriverSpec, type SimParams, Simulation } from '../engine/simulation';
import {
  addDiag,
  cabs2,
  cadd,
  type CMat,
  type CVec,
  cconj,
  cexp,
  cmul,
  cscale,
  cx,
  gram,
  matvec,
  norm2,
  principalGeneralized,
  solve,
  trace,
} from './complex';

export type Rect = [number, number, number, number]; // r0, c0, r1, c1 (inclusive, 2D cells)
export type Design = 'das' | 'focus' | 'pm' | 'acc';

export const DESIGN_LABELS: Record<Design, string> = {
  das: 'Delay-and-sum',
  focus: 'Focus (time reversal)',
  pm: 'Pressure matching',
  acc: 'Acoustic contrast (ACC)',
};

export interface Geometry {
  params: SimParams;
  materials: Uint8Array;
  speed: Float32Array | null;
}

export interface Transfer {
  f: number;
  bright: number[][]; // cell positions
  dark: number[][];
  Hb: CMat; // [point][speaker]
  Hd: CMat;
}

/** Up to `max` cells of a rectangle on an even sub-grid. */
export function zonePoints(r: Rect, max = 40): number[][] {
  const [r0, c0, r1, c1] = [Math.min(r[0], r[2]), Math.min(r[1], r[3]), Math.max(r[0], r[2]), Math.max(r[1], r[3])];
  const h = r1 - r0 + 1;
  const w = c1 - c0 + 1;
  const step = Math.max(1, Math.ceil(Math.sqrt((h * w) / max)));
  const out: number[][] = [];
  for (let i = r0; i <= r1; i += step) for (let j = c0; j <= c1; j += step) out.push([i, j]);
  return out;
}

export function rectCentre(r: Rect): number[] {
  return [(r[0] + r[2]) / 2, (r[1] + r[3]) / 2];
}

function buildSim(g: Geometry): Simulation {
  const sim = new Simulation(g.params);
  sim.setMaterialMap(g.materials);
  if (g.speed) sim.setSpeedMap(g.speed);
  return sim;
}

/** Settling time: three domain diagonals plus five periods (grid time units). */
export function settleTime(params: SimParams, f: number): number {
  const diag = Math.hypot(...params.shape) * params.dx;
  return (3 * diag) / params.c + 5 / f;
}

/**
 * Measure H for every speaker at frequency f (one steady-state run per
 * speaker). `onProgress` receives a fraction in [0, 1]; the loop yields to
 * the event loop so the UI stays responsive.
 */
export async function measureTransfer(
  g: Geometry,
  speakers: number[][],
  bright: Rect,
  dark: Rect,
  f: number,
  opts: { periods?: number; settle?: number; onProgress?: (x: number) => void; yieldEvery?: number } = {},
): Promise<Transfer> {
  const sim = buildSim(g);
  const bp = zonePoints(bright);
  const dp = zonePoints(dark);
  const idx = [...bp, ...dp].map((p) => sim.index(p));
  const dt = sim.dt;
  const settleSteps = Math.round((opts.settle ?? settleTime(g.params, f)) / dt);
  const W = Math.max(8, Math.round((opts.periods ?? 4) / (f * dt)));
  const w = 2 * Math.PI * f;
  const H: CMat = idx.map(() => speakers.map(() => cx(0)));
  const total = speakers.length * (settleSteps + W);
  let done = 0;
  for (let s = 0; s < speakers.length; s++) {
    sim.reset();
    sim.setDrivers([{ id: 'tx', pos: speakers[s], waveform: { type: 'cosine', amplitude: 1, frequency: f }, enabled: true }]);
    for (let k = 0; k < settleSteps + W; k++) {
      sim.step();
      if (k >= settleSteps) {
        const t = sim.time;
        const c = Math.cos(w * t);
        const sn = Math.sin(w * t);
        for (let m = 0; m < idx.length; m++) {
          const v = sim.p[idx[m]];
          H[m][s][0] += (2 / W) * v * c;
          H[m][s][1] -= (2 / W) * v * sn;
        }
      }
      done++;
      if (done % (opts.yieldEvery ?? 400) === 0) {
        opts.onProgress?.(done / total);
        await new Promise((r) => setTimeout(r, 0));
      }
    }
  }
  opts.onProgress?.(1);
  return { f, bright: bp, dark: dp, Hb: H.slice(0, bp.length), Hd: H.slice(bp.length) };
}

function normalise(w: CVec): CVec {
  const n = w.length;
  const s = Math.sqrt(n / (norm2(w) || 1));
  return w.map((v) => cscale(v, s));
}

/** Array weights for a design (effort-normalised, ||w||^2 = N). */
export function design(kind: Design, T: Transfer, speakers: number[][], params: SimParams, bright: Rect): CVec {
  const n = speakers.length;
  const k = (2 * Math.PI * T.f) / params.c;
  const centre = rectCentre(bright);
  if (kind === 'das') {
    return normalise(speakers.map((s) => cexp(k * params.dx * Math.hypot(s[0] - centre[0], s[1] - centre[1]))));
  }
  if (kind === 'focus') {
    let best = 0;
    let bd = Infinity;
    T.bright.forEach((p, i) => {
      const d = Math.hypot(p[0] - centre[0], p[1] - centre[1]);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    return normalise(T.Hb[best].map(cconj));
  }
  const Rb = gram(T.Hb);
  const Rd = gram(T.Hd);
  if (kind === 'pm') {
    const A = gram([...T.Hb, ...T.Hd]);
    const m = T.Hb.length + T.Hd.length;
    const lam = (1e-3 * trace(A)) / n;
    // (H^H H / m + lam I) w = H_b^H 1 / m
    const rhs: CVec = Array.from({ length: n }, (_, j) => T.Hb.reduce<[number, number]>((acc, row) => cadd(acc, cconj(row[j])), cx(0))).map((v) => cscale(v, 1 / m));
    return normalise(solve(addDiag(A, lam), rhs));
  }
  const delta = (1e-4 * (trace(Rd) + trace(Rb))) / n;
  return normalise(principalGeneralized(Rb, addDiag(Rd, delta)));
}

/** Predicted zone contrast (dB) of weights w. */
export function contrastDb(T: Transfer, w: CVec): number {
  const e = (H: CMat) => H.reduce((s, row) => s + cabs2(row.reduce<[number, number]>((a, h, j) => cadd(a, cmul(h, w[j])), cx(0))), 0) / Math.max(1, H.length);
  return 10 * Math.log10(e(T.Hb) / Math.max(e(T.Hd), 1e-30));
}

/** Mean |p|^2 in the bright zone relative to a single unit-weight speaker (dB). */
export function brightGainDb(T: Transfer, w: CVec): number {
  const pb = T.Hb.map((row) => matvec([row], w)[0]);
  const single = T.Hb.reduce((s, row) => s + cabs2(row[0]), 0) / T.Hb.length;
  const e = pb.reduce((s, v) => s + cabs2(v), 0) / pb.length;
  return 10 * Math.log10(e / Math.max(single, 1e-30));
}

/** Drivers playing cos(w (t - d_s)) * g_s for weights w_s = g_s e^{-j w d_s}. */
export function weightedDrivers(speakers: number[][], w: CVec, f: number, amplitude = 1, ids?: string[]): DriverSpec[] {
  const period = 1 / f;
  return speakers.map((pos, s) => {
    const phase = Math.atan2(w[s][1], w[s][0]);
    let delay = -phase / (2 * Math.PI * f);
    delay = ((delay % period) + period) % period;
    return {
      id: ids?.[s] ?? `arr-${s}`,
      pos: [...pos],
      waveform: { type: 'cosine', amplitude, frequency: f },
      enabled: true,
      gain: Math.sqrt(cabs2(w[s])),
      delay,
    };
  });
}

/** Measured contrast (dB) from a time-averaged |p|^2 map over two zones. */
export function zoneContrastFromEnergy(energy: Float32Array, index: (p: number[]) => number, bright: Rect, dark: Rect): number {
  const mean = (r: Rect) => {
    const pts = zonePoints(r, 400);
    return pts.reduce((s, p) => s + energy[index(p)], 0) / pts.length;
  };
  return 10 * Math.log10(mean(bright) / Math.max(mean(dark), 1e-30));
}

/** Run weighted drivers to steady state and return the measured contrast (dB). */
export function verifyContrast(g: Geometry, drivers: DriverSpec[], bright: Rect, dark: Rect, f: number, periods = 6): number {
  const sim = buildSim(g);
  sim.setDrivers(drivers);
  const settle = Math.round(settleTime(g.params, f) / sim.dt);
  for (let k = 0; k < settle; k++) sim.step();
  const W = Math.round(periods / (f * sim.dt));
  const e = new Float32Array(sim.n);
  for (let k = 0; k < W; k++) {
    sim.step();
    for (let i = 0; i < sim.n; i++) e[i] += sim.p[i] * sim.p[i];
  }
  return zoneContrastFromEnergy(e, (p) => sim.index(p), bright, dark);
}

/** A straight line array: `count` speakers centred on `centre`, `spacing` cells apart. */
export function lineArray(centre: number[], count: number, spacing: number, axis: 0 | 1, shape: number[]): number[][] {
  const out: number[][] = [];
  for (let s = 0; s < count; s++) {
    const off = (s - (count - 1) / 2) * spacing;
    const p = [...centre];
    p[axis] = Math.round(centre[axis] + off);
    p[0] = Math.max(1, Math.min(shape[0] - 2, p[0]));
    p[1] = Math.max(1, Math.min(shape[1] - 2, p[1]));
    out.push(p);
  }
  return out;
}
