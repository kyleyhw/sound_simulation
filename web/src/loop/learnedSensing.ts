/**
 * Learned room estimate for the closed loop (loop sensing study,
 * tests/reports/loop_sensing_2026_09_25.md).
 *
 * The loop's back-projection thresholds the coherent migration image and
 * keeps the largest blob. A linear array sees mostly the front faces of
 * obstacles, so that estimate is a thin arc. Here a small U-Net maps the same
 * grid-aligned migration images (plus geometry channels) to an obstacle
 * probability per cell, and the estimate is the cells above a threshold
 * chosen on validation scenes. The network is trained by
 * scripts/train_loop_sensing.py on random loop scenes simulated with this
 * code (web/scripts/loop_sensing_data.ts); the feature code below is mirrored
 * line by line in `features()` there, and tests/unit/loopSensing.test.ts
 * checks the two against each other.
 *
 * Input channels (on the central m x m crop, m = 8 floor(n / 8)):
 *   0-2  coherent, left-half and right-half coherent images / max |coherent|
 *   3    sqrt(coherent energy / its max)            (the back-projection image)
 *   4    sqrt(incoherent energy / its max)
 *   5    standardised log10 of the coherent-energy max (a constant plane)
 *   6-7  row and column coordinates in [-1, 1]
 *   8    distance to the array centre / n
 *   9    the array's near field (1 within 6 cells of an element)
 *   10   training prior logit / 5
 * Maxima are taken outside the near field. The output logit is the prior
 * logit plus the U-Net output.
 */

import type { LearnedEstimator, MigrationImages } from './closedLoop';
import { nearArray } from './closedLoop';
import { concat, conv2d, maxPool2, relu, type Tensor, tensor, upsampleNearest2 } from '../sensing/nn';

export interface LoopModelManifest {
  format: 'loop-unet-v1';
  grid: number; // full grid side n
  crop: number; // model side m
  offset: number; // crop offset (rows and columns)
  width: number;
  channels: number;
  norm: { logMaxMean: number; logMaxStd: number; exclude: number; clip: number };
  threshold: number; // on the probability, chosen on validation scenes
  prior_logit: number[]; // m x m
  tensors: { name: string; shape: number[]; offset: number }[];
  training?: Record<string, unknown>;
}

export const LOOP_FEATURE_CHANNELS = 11;

/** Model input features for one sweep (see the module comment). */
export function loopFeatures(im: MigrationImages, array: number[][], n: number, man: Pick<LoopModelManifest, 'crop' | 'offset' | 'norm' | 'prior_logit'>): Tensor {
  const { crop: m, offset: o, norm } = man;
  const near = nearArray(array, n, n, norm.exclude);
  let sC = 0;
  let sE = 0;
  let sI = 0;
  for (let q = 0; q < n * n; q++) {
    if (near[q]) continue;
    sC = Math.max(sC, Math.abs(im.coherent[q]));
    sE = Math.max(sE, im.image[q]);
    sI = Math.max(sI, im.incoherent[q]);
  }
  sC = Math.max(sC, 1e-12);
  sE = Math.max(sE, 1e-24);
  sI = Math.max(sI, 1e-24);
  const clip = norm.clip;
  const cl = (v: number, lo: number) => Math.min(clip, Math.max(lo, v));
  const logMax = (Math.log10(sE) - norm.logMaxMean) / norm.logMaxStd;
  const ac = [array.reduce((s, a) => s + a[0], 0) / array.length, array.reduce((s, a) => s + a[1], 0) / array.length];
  const x = tensor(LOOP_FEATURE_CHANNELS, m, m);
  const P = m * m;
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) {
      const gi = i + o;
      const gj = j + o;
      const g = gi * n + gj;
      const q = i * m + j;
      x.d[q] = cl(im.coherent[g] / sC, -clip);
      x.d[P + q] = cl(im.left[g] / sC, -clip);
      x.d[2 * P + q] = cl(im.right[g] / sC, -clip);
      x.d[3 * P + q] = cl(Math.sqrt(Math.max(0, im.image[g]) / sE), 0);
      x.d[4 * P + q] = cl(Math.sqrt(Math.max(0, im.incoherent[g]) / sI), 0);
      x.d[5 * P + q] = logMax;
      x.d[6 * P + q] = (2 * gi) / (n - 1) - 1;
      x.d[7 * P + q] = (2 * gj) / (n - 1) - 1;
      x.d[8 * P + q] = Math.hypot(gi - ac[0], gj - ac[1]) / n;
      x.d[9 * P + q] = near[g];
      x.d[10 * P + q] = man.prior_logit[q] / 5;
    }
  return x;
}

/** The compact U-Net of imaging/models.py (BatchNorm folded into the convolutions). */
export class LoopUNet implements LearnedEstimator {
  readonly manifest: LoopModelManifest;
  readonly shape: number[];
  private w = new Map<string, Float32Array>();
  private prior: Float32Array;

  constructor(manifest: LoopModelManifest, weights: ArrayBuffer) {
    if (manifest.format !== 'loop-unet-v1') throw new Error(`unknown loop model format ${manifest.format}`);
    this.manifest = manifest;
    this.shape = [manifest.grid, manifest.grid];
    const all = new Float32Array(weights);
    for (const t of manifest.tensors) {
      const k = t.shape.reduce((a, b) => a * b, 1);
      this.w.set(t.name, all.subarray(t.offset, t.offset + k));
    }
    this.prior = Float32Array.from(manifest.prior_logit);
  }

  static async load(base: string): Promise<LoopUNet> {
    const [m, b] = await Promise.all([fetch(`${base}.json`).then((r) => r.json()), fetch(`${base}.bin`).then((r) => r.arrayBuffer())]);
    return new LoopUNet(m as LoopModelManifest, b);
  }

  private conv(x: Tensor, name: string, k = 3): Tensor {
    const b = this.w.get(`${name}.bias`);
    const W = this.w.get(`${name}.weight`);
    if (!b || !W) throw new Error(`missing weight ${name}`);
    return conv2d(x, W, b, b.length, k, k === 3 ? 1 : 0);
  }

  private block(x: Tensor, name: string): Tensor {
    return relu(this.conv(relu(this.conv(x, `${name}.0`)), `${name}.1`));
  }

  /** Output logits (m x m) for the given input features. */
  forward(x: Tensor): Float32Array {
    const e1 = this.block(x, 'e1');
    const e2 = this.block(maxPool2(e1), 'e2');
    const e3 = this.block(maxPool2(e2), 'e3');
    const b = this.block(maxPool2(e3), 'bott');
    const d3 = this.block(concat(upsampleNearest2(b), e3), 'd3');
    const d2 = this.block(concat(upsampleNearest2(d3), e2), 'd2');
    const d1 = this.block(concat(upsampleNearest2(d2), e1), 'd1');
    const out = this.conv(d1, 'out', 1).d;
    for (let q = 0; q < out.length; q++) out[q] += this.prior[q];
    return out;
  }

  /** Obstacle probability over the full grid (0 outside the crop). */
  probability(images: MigrationImages, array: number[][]): Float32Array {
    const { grid: n, crop: m, offset: o } = this.manifest;
    const logits = this.forward(loopFeatures(images, array, n, this.manifest));
    const p = new Float32Array(n * n);
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) p[(i + o) * n + j + o] = 1 / (1 + Math.exp(-logits[i * m + j]));
    return p;
  }

  estimate(images: MigrationImages, array: number[][]): { estimate: Uint8Array; probability: Float32Array } {
    const probability = this.probability(images, array);
    const estimate = new Uint8Array(probability.length);
    for (let q = 0; q < probability.length; q++) if (probability[q] >= this.manifest.threshold) estimate[q] = 1;
    return { estimate, probability };
  }
}
