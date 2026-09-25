/**
 * Closed loop in simulation (plan Phase 8): sense -> digital twin -> design
 * -> act, run against a "true" room that can change between epochs.
 *
 * Sensing (8.1). The speaker array also records: every speaker in turn emits
 * a Ricker pulse, and every array position records. The residual
 * r_sm(t) = rec_room(t) - rec_empty(t) removes the direct path and the
 * device's own response. The empty-room reference is what a device measures
 * once in open space. The residual is back-projected with delay-and-sum
 * (Kirchhoff) migration:
 *
 *   I(x) = sum_{s,m} env(r_sm)( t0 + (|x - x_s| + |x - x_m|) dx / c ),
 *
 * where t0 is the pulse delay. The sum is coherent: raw residuals are summed
 * and the image is the squared sum smoothed over the pulse length. An
 * envelope (incoherent) sum only resolves range and paints whole ellipse
 * arcs; the coherent sum also uses the phase across the 64 speaker-mic
 * pairs, which narrows the arc to the reflector. Pixels above a fraction of
 * the image maximum, away from the array, form candidate cells. The largest
 * connected blob, dilated by one cell, is the obstacle estimate. The twin is the true room's outer boundary plus these cells,
 * modelled as rigid.
 *
 * Control. ACC weights (control/soundfield.ts) designed on the twin are
 * applied in the true room, and the steady-state zone contrast is measured
 * there. A crude twin can be worse than assuming an empty room. The
 * *guarded* loop therefore also closes on measurement: a few monitor mics
 * per zone (five cells around each zone centre, e.g. a phone at the
 * listener) score the twin design and the empty-room design in the true
 * room, and the better one is kept. Only the monitor points take part in
 * that choice. The reported contrast is always measured over the whole
 * zones. Two references run on the same epoch: a static controller
 * designed once at epoch 0, and an oracle designed on the true room (8.5).
 */

import { contrastDb, design, type Geometry, measureTransfer, type Rect, verifyContrast, weightedDrivers } from '../control/soundfield';
import { Simulation } from '../engine/simulation';
import type { CVec } from '../control/complex';

export interface SenseResult {
  image: Float32Array; // back-projected energy, row-major 2D
  estimate: Uint8Array; // 1 = estimated obstacle (from the active estimator)
  iou: number | null; // against the true obstacle cells (if given)
  estimator: EstimatorName; // which estimator formed `estimate`
  images: MigrationImages;
  backprojection: Uint8Array; // the back-projection estimate (always computed; it is cheap)
  probability: Float32Array | null; // learned obstacle probability (learned estimator only)
  learnedMs: number; // time spent in the learned estimator (0 for back-projection)
}

function buildSim(g: Geometry): Simulation {
  const sim = new Simulation(g.params);
  sim.setMaterialMap(g.materials);
  if (g.speed) sim.setSpeedMap(g.speed);
  return sim;
}

/** Record every array position while each speaker emits a Ricker pulse. */
export function pingRecordings(g: Geometry, array: number[][], f0: number, steps: number): Float32Array[][] {
  const sim = buildSim(g);
  const delay = 1.5 / f0;
  const out: Float32Array[][] = [];
  const idx = array.map((p) => sim.index(p));
  for (let s = 0; s < array.length; s++) {
    sim.reset();
    sim.setDrivers([{ id: 'ping', pos: array[s], waveform: { type: 'ricker', amplitude: 1, frequency: f0, delay }, enabled: true }]);
    const rec = array.map(() => new Float32Array(steps));
    for (let k = 0; k < steps; k++) {
      sim.step();
      for (let m = 0; m < idx.length; m++) rec[m][k] = sim.p[idx[m]];
    }
    out.push(rec);
  }
  return out;
}

/** The 4-connected blob of `mask` with the largest summed `weight`. */
function largestComponent(mask: Uint8Array, rows: number, cols: number, weight: Float32Array): Uint8Array {
  const label = new Int32Array(mask.length).fill(-1);
  let best = -1;
  let bestW = -1;
  let next = 0;
  for (let q0 = 0; q0 < mask.length; q0++) {
    if (!mask[q0] || label[q0] >= 0) continue;
    const stack = [q0];
    label[q0] = next;
    let w = 0;
    while (stack.length) {
      const q = stack.pop()!;
      w += weight[q];
      const i = Math.floor(q / cols);
      const j = q % cols;
      for (const [a, b] of [
        [i - 1, j],
        [i + 1, j],
        [i, j - 1],
        [i, j + 1],
      ]) {
        if (a < 0 || b < 0 || a >= rows || b >= cols) continue;
        const qq = a * cols + b;
        if (mask[qq] && label[qq] < 0) {
          label[qq] = next;
          stack.push(qq);
        }
      }
    }
    if (w > bestW) {
      bestW = w;
      best = next;
    }
    next++;
  }
  const out = new Uint8Array(mask.length);
  for (let q = 0; q < mask.length; q++) if (label[q] === best && best >= 0) out[q] = 1;
  return out;
}

function dilate(mask: Uint8Array, rows: number, cols: number): Uint8Array {
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      if (!mask[i * cols + j]) continue;
      for (let a = Math.max(1, i - 1); a <= Math.min(rows - 2, i + 1); a++)
        for (let b = Math.max(1, j - 1); b <= Math.min(cols - 2, j + 1); b++) out[a * cols + b] = 1;
    }
  return out;
}

/** The back-projected images of one sensing sweep (all row-major, rows x cols). */
export interface MigrationImages {
  /** Signed coherent delay-and-sum over all speaker-mic pairs. */
  coherent: Float32Array;
  /** Signed coherent sum over the pairs whose source is in the left / right half of the array. */
  left: Float32Array;
  right: Float32Array;
  /** Energy of the coherent image, box-smoothed over about half a wavelength (the back-projection image). */
  image: Float32Array;
  /** Incoherent sum of squared residuals at the travel lag, box-smoothed the same way. */
  incoherent: Float32Array;
}

/** Number of recorded steps for a sensing sweep (2.2 domain diagonals). */
export function senseSteps(params: Geometry['params']): number {
  const [rows, cols] = params.shape;
  return Math.round((2.2 * Math.hypot(rows, cols) * params.dx) / params.c / (0.5 * params.dx / params.c));
}

function boxSmooth(src: Float32Array, rows: number, cols: number, r: number, square: boolean): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      let acc = 0;
      let n = 0;
      for (let a = Math.max(0, i - r); a <= Math.min(rows - 1, i + r); a++)
        for (let b = Math.max(0, j - r); b <= Math.min(cols - 1, j + r); b++) {
          const v = src[a * cols + b];
          acc += square ? v ** 2 : v;
          n++;
        }
      out[i * cols + j] = acc / n;
    }
  return out;
}

/**
 * Back-project the residual scattered field r_sm = rec_room - rec_empty by
 * delay-and-sum migration (coherent, per half-array, and incoherent).
 */
export function migrationImages(recRoom: Float32Array[][], recEmpty: Float32Array[][], array: number[][], params: Geometry['params'], f0: number): MigrationImages {
  const [rows, cols] = params.shape;
  const steps = recRoom[0][0].length;
  const dt = new Simulation(params).dt;
  const delay = 1.5 / f0;
  const res = recRoom.map((rs, s) => rs.map((r, m) => r.map((v, k) => v - recEmpty[s][m][k])));
  const half = array.length / 2;
  const coherent = new Float32Array(rows * cols);
  const left = new Float32Array(rows * cols);
  const right = new Float32Array(rows * cols);
  const inco = new Float32Array(rows * cols);
  const cdx = params.dx / params.c;
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      const d = array.map((a) => Math.hypot(i - a[0], j - a[1]) * cdx);
      let acc = 0;
      let accL = 0;
      let acc2 = 0;
      for (let s = 0; s < array.length; s++)
        for (let m = 0; m < array.length; m++) {
          const k = Math.round((delay + d[s] + d[m]) / dt);
          if (k < steps) {
            const v = res[s][m][k];
            acc += v;
            if (s < half) accL += v;
            acc2 += v * v;
          }
        }
      coherent[i * cols + j] = acc;
      left[i * cols + j] = accL;
      right[i * cols + j] = acc - accL;
      inco[i * cols + j] = acc2;
    }
  // Energy of the coherent image, smoothed over about half a wavelength.
  const r = Math.max(1, Math.round(0.25 / f0 / params.dx * params.c));
  return { coherent, left, right, image: boxSmooth(coherent, rows, cols, r, true), incoherent: boxSmooth(inco, rows, cols, r, false) };
}

/** True where a cell lies within `exclude` cells of an array element (its near field). */
export function nearArray(array: number[][], rows: number, cols: number, exclude = 6): Uint8Array {
  const out = new Uint8Array(rows * cols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) if (array.some((a) => Math.hypot(i - a[0], j - a[1]) < exclude)) out[i * cols + j] = 1;
  return out;
}

/**
 * The back-projection estimate: cells above `threshold` x the image maximum
 * (away from the array), the largest blob, dilated by one cell.
 */
export function backProjectionEstimate(image: Float32Array, array: number[][], shape: number[], opts: { threshold?: number; exclude?: number; dilate?: boolean } = {}): Uint8Array {
  const [rows, cols] = shape;
  const near = nearArray(array, rows, cols, opts.exclude ?? 6);
  let max = 0;
  for (let q = 0; q < rows * cols; q++) if (!near[q]) max = Math.max(max, image[q]);
  const th = (opts.threshold ?? 0.7) * max;
  const cand = new Uint8Array(rows * cols);
  for (let i = 1; i < rows - 1; i++)
    for (let j = 1; j < cols - 1; j++) if (!near[i * cols + j] && image[i * cols + j] >= th) cand[i * cols + j] = 1;
  const blob = largestComponent(cand, rows, cols, image);
  return opts.dilate === false ? blob : dilate(blob, rows, cols);
}

/** Intersection over union of an estimate against the true obstacle cells (nonzero). */
export function maskIou(estimate: Uint8Array, truth: Uint8Array): number {
  let inter = 0;
  let uni = 0;
  for (let q = 0; q < estimate.length; q++) {
    const a = estimate[q] !== 0;
    const b = truth[q] !== 0;
    if (a && b) inter++;
    if (a || b) uni++;
  }
  return uni ? inter / uni : 1;
}

/** Turns the migration images of a sweep into an obstacle estimate (e.g. the learned U-Net). */
export interface LearnedEstimator {
  /** Grid shape the estimator was trained for; other shapes fall back to back-projection. */
  readonly shape: number[];
  estimate(images: MigrationImages, array: number[][]): { estimate: Uint8Array; probability: Float32Array };
}

export type EstimatorName = 'backprojection' | 'learned';

/**
 * Sense the room: back-project the residual scattered field and form the
 * obstacle estimate, by thresholding (back-projection) or with a learned
 * estimator on the same images. `truthObstacles` (optional) only scores the
 * estimate; it is never used to form it.
 */
export function senseRoom(
  room: Geometry,
  empty: Geometry,
  array: number[][],
  opts: {
    f0?: number;
    steps?: number;
    threshold?: number;
    exclude?: number;
    truthObstacles?: Uint8Array;
    dilate?: boolean;
    learned?: LearnedEstimator | null;
    recEmpty?: Float32Array[][];
  } = {},
): SenseResult {
  const f0 = opts.f0 ?? 0.08;
  const steps = opts.steps ?? senseSteps(room.params);
  const recRoom = pingRecordings(room, array, f0, steps);
  const recEmpty = opts.recEmpty ?? pingRecordings(empty, array, f0, steps);
  const images = migrationImages(recRoom, recEmpty, array, room.params, f0);
  const bp = backProjectionEstimate(images.image, array, room.params.shape, opts);
  const useLearned = !!opts.learned && opts.learned.shape.every((v, i) => v === room.params.shape[i]);
  const tl = performance.now();
  const learned = useLearned ? opts.learned!.estimate(images, array) : null;
  const learnedMs = learned ? performance.now() - tl : 0;
  const estimate = learned ? learned.estimate : bp;
  return {
    image: images.image,
    estimate,
    iou: opts.truthObstacles ? maskIou(estimate, opts.truthObstacles) : null,
    estimator: learned ? 'learned' : 'backprojection',
    images,
    backprojection: bp,
    probability: learned?.probability ?? null,
    learnedMs,
  };
}

/** Twin geometry: the true outer boundary and the estimated obstacles (rigid). */
export function twinGeometry(truth: Geometry, estimate: Uint8Array): Geometry {
  const materials = new Uint8Array(estimate.length);
  for (let q = 0; q < estimate.length; q++) if (estimate[q]) materials[q] = 2;
  return { params: truth.params, materials, speed: null };
}

export interface Epoch {
  label: string;
  truth: Geometry;
  bright: Rect;
  dark: Rect;
}

export interface EpochResult {
  label: string;
  guarded: number; // better of twin / empty-room design, chosen by the monitor mics
  guardChoice: 'twin' | 'empty';
  sensedIou: number | null;
  estimator: EstimatorName; // which estimator formed the twin
  adaptive: number; // measured contrast in the true room, controller designed on the twin
  static: number; // controller designed once at epoch 0
  oracle: number; // controller designed on the true room
  naive: number; // controller designed on the empty room (no sensing)
  predictedTwin: number;
  latencyMs: { sense: number; estimate: number; twin: number; design: number; act: number }; // sense includes estimate
  estimate: Uint8Array;
  image: Float32Array;
  probability: Float32Array | null; // learned obstacle probability (learned estimator only)
}

/** Five monitor cells around a zone centre (a plus shape, 3 cells apart). */
export function monitorPoints(r: Rect): number[][] {
  const c = [Math.round((r[0] + r[2]) / 2), Math.round((r[1] + r[3]) / 2)];
  return [c, [c[0] - 3, c[1]], [c[0] + 3, c[1]], [c[0], c[1] - 3], [c[0], c[1] + 3]];
}

/** Steady-state contrast (dB) between two point sets for the given drivers. */
export function pointContrast(g: Geometry, drivers: ReturnType<typeof weightedDrivers>, bright: number[][], dark: number[][], f: number, periods = 6): number {
  const sim = buildSim(g);
  sim.setDrivers(drivers);
  const settle = Math.round(((3 * Math.hypot(...g.params.shape) * g.params.dx) / g.params.c + 5 / f) / sim.dt);
  for (let k = 0; k < settle; k++) sim.step();
  const W = Math.round(periods / (f * sim.dt));
  const bi = bright.map((p) => sim.index(p));
  const di = dark.map((p) => sim.index(p));
  let eb = 0;
  let ed = 0;
  for (let k = 0; k < W; k++) {
    sim.step();
    for (const q of bi) eb += sim.p[q] ** 2;
    for (const q of di) ed += sim.p[q] ** 2;
  }
  return 10 * Math.log10(eb / bi.length / Math.max(ed / di.length, 1e-30));
}

export async function accWeights(g: Geometry, array: number[][], bright: Rect, dark: Rect, f: number): Promise<{ w: CVec; predicted: number }> {
  const T = await measureTransfer(g, array, bright, dark, f, { yieldEvery: 1e9 });
  const w = design('acc', T, array, g.params, bright);
  return { w, predicted: contrastDb(T, w) };
}

/**
 * Run the loop over a sequence of epochs (the true room or zones change
 * between them). `onEpoch` is called as each epoch finishes.
 */
export async function runLoop(
  epochs: Epoch[],
  array: number[][],
  f: number,
  opts: { onEpoch?: (r: EpochResult, i: number) => void; senseThreshold?: number; dilate?: boolean; learned?: LearnedEstimator | null } = {},
): Promise<EpochResult[]> {
  const results: EpochResult[] = [];
  let staticW: CVec | null = null;
  const tick = () => new Promise((r) => setTimeout(r, 0));
  for (let e = 0; e < epochs.length; e++) {
    const ep = epochs[e];
    const empty: Geometry = { params: ep.truth.params, materials: new Uint8Array(ep.truth.materials.length), speed: null };
    const t0 = performance.now();
    const sensed = senseRoom(ep.truth, empty, array, { truthObstacles: ep.truth.materials, threshold: opts.senseThreshold, dilate: opts.dilate, learned: opts.learned });
    const t1 = performance.now();
    const twin = twinGeometry(ep.truth, sensed.estimate);
    const t2 = performance.now();
    await tick();
    const adaptive = await accWeights(twin, array, ep.bright, ep.dark, f);
    const t3 = performance.now();
    const act = (w: CVec) => verifyContrast(ep.truth, weightedDrivers(array, w, f), ep.bright, ep.dark, f);
    const measured = act(adaptive.w);
    const t4 = performance.now();
    await tick();
    if (!staticW) staticW = adaptive.w;
    const oracle = await accWeights(ep.truth, array, ep.bright, ep.dark, f);
    const naive = await accWeights(empty, array, ep.bright, ep.dark, f);
    // Guard: choose between the twin and the empty-room design with the monitor mics only.
    const mb = monitorPoints(ep.bright);
    const md = monitorPoints(ep.dark);
    const monTwin = pointContrast(ep.truth, weightedDrivers(array, adaptive.w, f), mb, md, f);
    const monEmpty = pointContrast(ep.truth, weightedDrivers(array, naive.w, f), mb, md, f);
    const naiveMeasured = act(naive.w);
    const r: EpochResult = {
      label: ep.label,
      guarded: monTwin >= monEmpty ? measured : naiveMeasured,
      guardChoice: monTwin >= monEmpty ? 'twin' : 'empty',
      sensedIou: sensed.iou,
      estimator: sensed.estimator,
      adaptive: measured,
      static: act(staticW),
      oracle: act(oracle.w),
      naive: naiveMeasured,
      predictedTwin: adaptive.predicted,
      latencyMs: { sense: t1 - t0, estimate: sensed.learnedMs, twin: t2 - t1, design: t3 - t2, act: t4 - t3 },
      estimate: sensed.estimate,
      image: sensed.image,
      probability: sensed.probability,
    };
    results.push(r);
    opts.onEpoch?.(r, e);
    await tick();
  }
  return results;
}
