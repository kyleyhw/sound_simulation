/**
 * The loop's learned room estimate (loop sensing study, 2026-09-25) against
 * PyTorch: the browser re-simulates held-out scenes, computes the migration
 * images and features, runs the U-Net, and must reproduce the training
 * script's features and logits (fixtures from
 * scripts/train_loop_sensing.py export --fixtures).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { Geometry } from '../../src/control/soundfield';
import { migrationImages, pingRecordings, runLoop, senseSteps } from '../../src/loop/closedLoop';
import { loopFeatures, type LoopModelManifest, LoopUNet } from '../../src/loop/learnedSensing';
import { demoScenario, randomLoopScene } from '../../src/loop/scenarios';
import { tensor, upsampleNearest2 } from '../../src/sensing/nn';

const root = new URL('../../public/models/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('loop_unet.json', root), 'utf8')) as LoopModelManifest;
const buf = readFileSync(new URL('loop_unet.bin', root));
const model = new LoopUNet(manifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const fx = JSON.parse(readFileSync(new URL('../fixtures/loop_sensing_parity.json', import.meta.url), 'utf8')) as {
  stride: number;
  rooms: { split: string; index: number; seed: number; raw_sum: number[]; raw_absmax: number[]; feature_sample: number[]; logits: number[] }[];
};

describe('nn: nearest upsampling', () => {
  it('matches F.interpolate(scale_factor=2, mode="nearest")', () => {
    const x = tensor(2, 2, 3, Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
    const y = upsampleNearest2(x);
    expect([y.c, y.h, y.w]).toEqual([2, 4, 6]);
    for (let c = 0; c < 2; c++) for (let i = 0; i < 4; i++) for (let j = 0; j < 6; j++) expect(y.d[(c * 4 + i) * 6 + j]).toBe(x.d[(c * 2 + (i >> 1)) * 3 + (j >> 1)]);
  });
});

describe('learned loop sensing matches PyTorch', () => {
  const demo = demoScenario(manifest.grid);
  const { array, params } = demo;
  const n = manifest.grid;
  const steps = senseSteps(params);
  const empty: Geometry = { params, materials: new Uint8Array(n * n), speed: null };
  const recEmpty = pingRecordings(empty, array, 0.08, steps);

  for (const room of fx.rooms) {
    it(`${room.split} scene ${room.seed}: simulation, features and U-Net logits`, () => {
      const truth = room.split === 'demo' ? demo.epochs[room.index].truth : randomLoopScene(room.seed, n).truth;
      const im = migrationImages(pingRecordings(truth, array, 0.08, steps), recEmpty, array, params, 0.08);
      // 1. The migration images equal the training data's (same code, so to rounding).
      const raw = [im.coherent, im.left, im.right, im.image, im.incoherent];
      raw.forEach((a, c) => {
        let s = 0;
        let mx = 0;
        for (const v of a) {
          s += v;
          mx = Math.max(mx, Math.abs(v));
        }
        expect(Math.abs(mx - room.raw_absmax[c])).toBeLessThan(1e-5 * room.raw_absmax[c]);
        expect(Math.abs(s - room.raw_sum[c])).toBeLessThan(1e-4 * n * n * room.raw_absmax[c]);
      });
      // 2. Features: the TypeScript mirror of the Python feature code.
      const x = loopFeatures(im, array, n, manifest);
      let ferr = 0;
      room.feature_sample.forEach((v, k) => (ferr = Math.max(ferr, Math.abs(x.d[k * fx.stride] - v))));
      expect(ferr).toBeLessThan(2e-5);
      // 3. Logits: the TypeScript U-Net (BatchNorm folded) against PyTorch.
      const logits = model.forward(x);
      let err = 0;
      let scale = 0;
      room.logits.forEach((v, q) => {
        err = Math.max(err, Math.abs(logits[q] - v));
        scale = Math.max(scale, Math.abs(v));
      });
      console.log(`${room.split} ${room.seed}: max |feature err| ${ferr.toExponential(1)}, max |logit err| ${err.toExponential(1)} (logit scale ${scale.toFixed(1)})`);
      expect(err).toBeLessThan(2e-3 * Math.max(1, scale));
    }, 60_000);
  }

  it('the loop uses the learned estimate at the model grid and reports it', async () => {
    const [r] = await runLoop([demo.epochs[0]], array, demo.frequency, { learned: model });
    expect(r.estimator).toBe('learned');
    expect(r.sensedIou!).toBeGreaterThan(0.8);
    expect(r.guarded).toBeGreaterThanOrEqual(10);
    expect(r.oracle).toBeGreaterThanOrEqual(r.guarded - 1);
  }, 120_000);

  it('falls back to back-projection on other grids', async () => {
    const sc = demoScenario(60);
    const [r] = await runLoop([sc.epochs[0]], sc.array, sc.frequency, { learned: model });
    expect(r.estimator).toBe('backprojection');
  }, 120_000);
});
