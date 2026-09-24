/**
 * A sensing session in the browser (plan 4.3.5, 6.6.3, 6.7.1): acquire poses
 * in a 64 x 64 room, run the model, fuse, and suggest where to move next.
 */
import { chirp, protocolSim, randomPose, recordPose } from './acquire';
import { fuse, type SkipModel } from './model';
import type { Simulation } from '../engine/simulation';

export interface Pose {
  driver: number[];
  mics: number[][];
  logits: Float32Array;
}

/** Nearest-neighbour resample of any 2D material map to an n x n obstacle mask. */
export function maskFromScene(materials: Uint8Array, rows: number, cols: number, n = 64): Uint8Array {
  const out = new Uint8Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const r = Math.min(rows - 1, Math.floor(((i + 0.5) * rows) / n));
      const c = Math.min(cols - 1, Math.floor(((j + 0.5) * cols) / n));
      out[i * n + j] = materials[r * cols + c] ? 1 : 0;
    }
  // The outer ring is the room wall (p = 0 edge) in the protocol, not an obstacle.
  for (let k = 0; k < n; k++) out[k] = out[(n - 1) * n + k] = out[k * n] = out[k * n + n - 1] = 0;
  return out;
}

/**
 * A random room in the spirit of the v2 "mixed" family: 1-6 shapes drawn from
 * rectangles, discs, thin walls and L-shapes. This is not the training
 * generator's exact distribution.
 */
export function randomRoom(n = 64, rnd: () => number = Math.random): Uint8Array {
  const m = new Uint8Array(n * n);
  const set = (i: number, j: number) => {
    if (i > 2 && j > 2 && i < n - 3 && j < n - 3) m[i * n + j] = 1;
  };
  const k = 1 + Math.floor(rnd() * 6);
  for (let s = 0; s < k; s++) {
    const ci = 8 + Math.floor(rnd() * (n - 16));
    const cj = 8 + Math.floor(rnd() * (n - 16));
    const a = 4 + Math.floor(rnd() * 10);
    const b = 4 + Math.floor(rnd() * 10);
    const kind = Math.floor(rnd() * 4);
    if (kind === 0) for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) set(ci + i - (a >> 1), cj + j - (b >> 1));
    else if (kind === 1) for (let i = -a; i <= a; i++) for (let j = -a; j <= a; j++) (i * i + j * j <= (a * a) / 2 ? set(ci + i, cj + j) : 0);
    else if (kind === 2) {
      const th = rnd() * Math.PI;
      for (let t = -2 * a; t <= 2 * a; t++) set(Math.round(ci + t * Math.cos(th)), Math.round(cj + t * Math.sin(th)));
    } else {
      for (let i = 0; i < 2 * a; i++) set(ci + i - a, cj - a);
      for (let j = 0; j < 2 * b; j++) set(ci + a - 1, cj + j - a);
    }
  }
  return m;
}

export class SensingSession {
  readonly model: SkipModel;
  readonly mask: Uint8Array;
  readonly n: number;
  readonly sim: Simulation;
  readonly src: Float32Array;
  poses: Pose[] = [];

  constructor(model: SkipModel, mask: Uint8Array) {
    this.model = model;
    this.mask = mask;
    const p = model.manifest.protocol;
    this.n = p.grid;
    this.sim = protocolSim(p, mask);
    this.src = chirp(p, this.sim.dt);
  }

  /** Acquire one pose (random, or with the driver at `at`) and run the model. */
  addPose(at?: number[]): Pose {
    const p = this.model.manifest.protocol;
    let pose = randomPose(this.mask, this.n, p.mic_spacing);
    if (at) {
      // Keep a random mic pair but move the device to the suggested spot.
      for (let t = 0; t < 200; t++) {
        const cand = randomPose(this.mask, this.n, p.mic_spacing);
        const c = [(cand.mics[0][0] + cand.mics[1][0]) / 2, (cand.mics[0][1] + cand.mics[1][1]) / 2];
        if (Math.hypot(c[0] - at[0], c[1] - at[1]) < 6) {
          pose = cand;
          break;
        }
      }
    }
    const [a, b] = recordPose(this.sim, p, this.src, pose.driver, pose.mics);
    const logits = this.model.forward(a, b, this.src);
    const out = { ...pose, logits };
    this.poses.push(out);
    return out;
  }

  fused(): Float32Array | null {
    return this.poses.length ? fuse(this.poses.map((p) => p.logits), this.model.manifest.calibration) : null;
  }

  /**
   * "Move here next" (6.6.3, heuristic): the free cell whose 12-cell
   * neighbourhood holds the most binary entropy of the fused map. A pose
   * mostly hears what is near it, so it is sent to where the map is least
   * certain. This is not an expected-information-gain computation.
   */
  suggestNext(fused: Float32Array): number[] {
    const n = this.n;
    const H = fused.map((q) => {
      const p = Math.min(1 - 1e-6, Math.max(1e-6, q));
      return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
    });
    let best = [n >> 1, n >> 1];
    let bestV = -1;
    for (let i = 6; i < n - 6; i += 3)
      for (let j = 6; j < n - 6; j += 3) {
        if (this.mask[i * n + j]) continue;
        let v = 0;
        for (let a = -12; a <= 12; a += 2)
          for (let b = -12; b <= 12; b += 2) {
            const r = i + a;
            const c = j + b;
            if (r >= 0 && c >= 0 && r < n && c < n && a * a + b * b <= 144) v += H[r * n + c];
          }
        if (v > bestV) {
          bestV = v;
          best = [i, j];
        }
      }
    return best;
  }
}

export function iou(pred: ArrayLike<boolean | number>, truth: Uint8Array): number {
  let inter = 0;
  let uni = 0;
  for (let q = 0; q < truth.length; q++) {
    const a = !!pred[q];
    const b = truth[q] !== 0;
    if (a && b) inter++;
    if (a || b) uni++;
  }
  return uni ? inter / uni : 1;
}
