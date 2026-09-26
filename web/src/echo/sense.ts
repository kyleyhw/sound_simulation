/**
 * Echo vision (#/echo): one sensing sweep of the closed loop's room estimate
 * (loop/closedLoop.ts senseRoom, loop/learnedSensing.ts), split so the page
 * can animate it.
 *
 * - `listenSweep` runs the room and the empty room in lockstep, emission by
 *   emission. The engine is linear, so room - empty is the scattered field:
 *   only the sound that bounced off something. It returns the recordings
 *   (identical to the loop's `pingRecordings`; tests/unit/echo.test.ts
 *   checks this exactly) and hands over each emission's scattered-field
 *   frames, gamma-coded to 8 bits, for playback.
 * - `cumulativeImages` / `singleImages` are the loop's `migrationImages`
 *   restricted to the first k pings / to ping k (the other pings get a zero
 *   residual), so the page can show the picture building up. With every
 *   ping included it is `migrationImages` itself, bit for bit.
 * - `analyseImages` is the rest of senseRoom: back-projection and the U-Net.
 */

import type { Geometry } from '../control/soundfield';
import { Simulation } from '../engine/simulation';
import { backProjectionEstimate, type LearnedEstimator, maskIou, type MigrationImages, migrationImages, senseSteps } from '../loop/closedLoop';
import { demoScenario } from '../loop/scenarios';
import { BAR_F0, type Device } from './device';
import { N } from './room';

/** Ricker ping centre frequency (cycles per unit time), as in the loop's sensing. */
export const F0 = BAR_F0;

/** The loop's sensing setup: 100 x 100 grid, CPML walls, the 8-speaker bar. */
export function echoSetup() {
  const { params, array } = demoScenario(N);
  return { params, array, steps: senseSteps(params) };
}

export function geometry(materials: Uint8Array): Geometry {
  return { params: echoSetup().params, materials, speed: null };
}

/**
 * Every speaker pings in turn while every bar position records (the loop's
 * pingRecordings). `onFrame(ping, step, p)` sees the live pressure field
 * after each step; `p` is the simulation's buffer, so copy it to keep it.
 */
export function recordPings(g: Geometry, array: number[][], f0: number, steps: number, onFrame?: (ping: number, step: number, p: Float32Array) => void): Float32Array[][] {
  const sim = new Simulation(g.params);
  sim.setMaterialMap(g.materials);
  if (g.speed) sim.setSpeedMap(g.speed);
  const delay = 1.5 / f0;
  const idx = array.map((p) => sim.index(p));
  const out: Float32Array[][] = [];
  for (let s = 0; s < array.length; s++) {
    sim.reset();
    sim.setDrivers([{ id: 'ping', pos: array[s], waveform: { type: 'ricker', amplitude: 1, frequency: f0, delay }, enabled: true }]);
    const rec = array.map(() => new Float32Array(steps));
    for (let k = 0; k < steps; k++) {
      sim.step();
      for (let m = 0; m < idx.length; m++) rec[m][k] = sim.p[idx[m]];
      onFrame?.(s, k, sim.p);
    }
    out.push(rec);
  }
  return out;
}

// ---- scattered-field frames ---------------------------------------------

/** Frames are kept at steps from, from + every, ... <= to. */
export interface FrameWindow {
  from: number;
  to: number;
  every: number;
}

export const frameCount = (w: FrameWindow) => Math.floor((w.to - w.from) / w.every) + 1;

/**
 * The steps worth showing for a room: from just before the click leaves the
 * speaker until the first-order echo of the farthest obstacle cell has
 * passed every microphone (plus one pulse period). Later steps hold only
 * weak multiple bounces. Clamped to the recording.
 */
export function echoWindow(materials: Uint8Array, device: Device, params: Geometry['params'], steps: number, every = 2): FrameWindow {
  const [rows, cols] = params.shape;
  const dt = new Simulation(params).dt;
  let far = 0;
  for (let q = 0; q < rows * cols; q++) {
    if (!materials[q]) continue;
    const i = Math.floor(q / cols);
    const j = q % cols;
    let ds = 0;
    let dm = 0;
    for (const s of device.speakers) ds = Math.max(ds, Math.hypot(i - s[0], j - s[1]));
    for (const m of device.mics) dm = Math.max(dm, Math.hypot(i - m[0], j - m[1]));
    far = Math.max(far, ds + dm);
  }
  const from = Math.max(0, Math.floor((device.delay - 1 / device.f0) / dt));
  const end = Math.ceil((device.delay + (far * params.dx) / params.c + 1 / device.f0) / dt);
  const to = Math.min(steps - 1, Math.max(from + 4 * every, end));
  return { from, to: from + Math.floor((to - from) / every) * every, every };
}

/** Frames are stored as round(127 sign(v) (|v| / S)^GAMMA), S the frame's peak |v|. */
export const FRAME_GAMMA = 0.5;

/** Gamma-code one field into `out` at `offset`; returns its scale S (peak |v|). */
export function encodeFrame(v: Float32Array, out: Int8Array, offset: number): number {
  let s = 0;
  for (let q = 0; q < v.length; q++) s = Math.max(s, Math.abs(v[q]));
  if (!(s > 0)) {
    out.fill(0, offset, offset + v.length);
    return 0;
  }
  const inv = 1 / s;
  for (let q = 0; q < v.length; q++) {
    const a = v[q] * inv;
    out[offset + q] = Math.round(127 * Math.sign(a) * Math.sqrt(Math.abs(a)));
  }
  return s;
}

/** Inverse of `encodeFrame` for one value. */
export const decodeValue = (q: number, scale: number) => Math.sign(q) * scale * (Math.abs(q) / 127) ** (1 / FRAME_GAMMA);

/** One emission of a sweep, as handed to `listenSweep`'s callback. */
export interface EmissionRecord {
  emission: number;
  /** Recordings at every microphone, room and empty room (`steps` samples each). */
  recRoom: Float32Array[];
  recEmpty: Float32Array[];
  /** Gamma-coded scattered-field frames (frameCount x cells) and their scales. */
  frames: Int8Array;
  scales: Float32Array;
}

/**
 * Run each emission in the room and in the empty room side by side. Returns
 * every recording; `onEmission` gets each emission's recordings and (if a
 * window is given) its scattered-field frames as soon as it is done.
 */
export function listenSweep(
  room: Geometry,
  device: Device,
  steps: number,
  win: FrameWindow | null,
  onEmission?: (r: EmissionRecord) => void,
): { recRoom: Float32Array[][]; recEmpty: Float32Array[][] } {
  const make = (materials: Uint8Array) => {
    const sim = new Simulation(room.params);
    sim.setMaterialMap(materials);
    if (room.speed) sim.setSpeedMap(room.speed);
    return sim;
  };
  const simR = make(room.materials);
  const simE = make(new Uint8Array(room.materials.length));
  const mic = device.mics.map((p) => simR.index(p));
  const cells = simR.n;
  const scratch = new Float32Array(cells);
  const recRoom: Float32Array[][] = [];
  const recEmpty: Float32Array[][] = [];
  device.emissions.forEach((em, e) => {
    const drivers = em.drivers.map((d, k) => ({ id: `ping${k}`, pos: device.speakers[d.speaker], waveform: d.waveform, enabled: true }));
    for (const sim of [simR, simE]) {
      sim.reset();
      sim.setDrivers(drivers);
    }
    const rr = device.mics.map(() => new Float32Array(steps));
    const re = device.mics.map(() => new Float32Array(steps));
    const nf = win ? frameCount(win) : 0;
    const frames = new Int8Array(nf * cells);
    const scales = new Float32Array(nf);
    for (let k = 0; k < steps; k++) {
      simR.step();
      simE.step();
      for (let m = 0; m < mic.length; m++) {
        rr[m][k] = simR.p[mic[m]];
        re[m][k] = simE.p[mic[m]];
      }
      if (win && k >= win.from && k <= win.to && (k - win.from) % win.every === 0) {
        const f = (k - win.from) / win.every;
        const a = simR.p;
        const b = simE.p;
        for (let q = 0; q < cells; q++) scratch[q] = a[q] - b[q];
        scales[f] = encodeFrame(scratch, frames, f * cells);
      }
    }
    recRoom.push(rr);
    recEmpty.push(re);
    onEmission?.({ emission: e, recRoom: rr, recEmpty: re, frames, scales });
  });
  return { recRoom, recEmpty };
}

/** Residual (echo-only) recordings of one emission, flattened mic by mic. */
export function residuals(recRoom: Float32Array[], recEmpty: Float32Array[]): Float32Array {
  const steps = recRoom[0].length;
  const out = new Float32Array(recRoom.length * steps);
  for (let m = 0; m < recRoom.length; m++) for (let k = 0; k < steps; k++) out[m * steps + k] = recRoom[m][k] - recEmpty[m][k];
  return out;
}

// ---- the picture, ping by ping ------------------------------------------

/**
 * `migrationImages` with only the pings for which `include(s)` holds: the
 * others get a zero residual. Pings not recorded yet may be missing.
 */
export function partialImages(
  recRoom: (Float32Array[] | undefined)[],
  recEmpty: (Float32Array[] | undefined)[],
  array: number[][],
  params: Geometry['params'],
  f0: number,
  include: (s: number) => boolean,
): MigrationImages {
  const known = recRoom.find((r) => r) ?? recEmpty.find((r) => r);
  if (!known) throw new Error('no recordings');
  const zero = known.map((r) => new Float32Array(r.length));
  const room = array.map((_, s) => (include(s) && recRoom[s] ? recRoom[s]! : (recEmpty[s] ?? zero)));
  const empty = array.map((_, s) => recEmpty[s] ?? zero);
  return migrationImages(room, empty, array, params, f0);
}

/** The images of the first k + 1 pings (with every ping: `migrationImages` exactly). */
export const cumulativeImages = (recRoom: (Float32Array[] | undefined)[], recEmpty: (Float32Array[] | undefined)[], array: number[][], params: Geometry['params'], f0: number, k: number) =>
  partialImages(recRoom, recEmpty, array, params, f0, (s) => s <= k);

/** The images of ping k alone (their coherent parts sum to the full coherent image). */
export const singleImages = (recRoom: (Float32Array[] | undefined)[], recEmpty: (Float32Array[] | undefined)[], array: number[][], params: Geometry['params'], f0: number, k: number) =>
  partialImages(recRoom, recEmpty, array, params, f0, (s) => s === k);

// ---- the estimates -------------------------------------------------------

export interface EchoResult {
  /** Back-projected echo energy (the image back-projection thresholds). */
  image: Float32Array;
  /** Back-projection estimate: the brightest blob of `image`. */
  backprojection: Uint8Array;
  /** U-Net obstacle probability (0 outside its crop). */
  probability: Float32Array;
  /** U-Net estimate: probability above the model's threshold. */
  learned: Uint8Array;
  iouBackprojection: number;
  iouLearned: number;
}

/** Back-projection and the network on a sweep's migration images, both scored against the truth. */
export function analyseImages(images: MigrationImages, array: number[][], truth: Geometry, model: LearnedEstimator): EchoResult {
  const backprojection = backProjectionEstimate(images.image, array, truth.params.shape);
  const { estimate: learned, probability } = model.estimate(images, array);
  return {
    image: images.image,
    backprojection,
    probability,
    learned,
    iouBackprojection: maskIou(backprojection, truth.materials),
    iouLearned: maskIou(learned, truth.materials),
  };
}

/** Everything senseRoom does after the pings, with both estimates scored against the truth. */
export function analyseEchoes(recRoom: Float32Array[][], recEmpty: Float32Array[][], array: number[][], truth: Geometry, model: LearnedEstimator): EchoResult {
  return analyseImages(migrationImages(recRoom, recEmpty, array, truth.params, F0), array, truth, model);
}
