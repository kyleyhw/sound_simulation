/**
 * Training data for the loop's learned room estimate (loop sensing study,
 * 2026-09-25). Runs the closed loop's own sensing code in Node, so the
 * images are exactly what the browser computes:
 *
 *   npm run loop:data -- --split train --count 2000 [--out ../data/loop_sensing] [--n 100]
 *
 * For each random scene (scenarios.randomLoopScene, seed = split base + i)
 * every speaker of the 8-speaker bar pings and every bar position records;
 * the residual against the empty-room recordings (computed once) is migrated
 * by closedLoop.migrationImages. Per scene it appends to <out>/<split>.*:
 *
 *   .images.f32  5 x n x n float32: coherent, left half, right half,
 *                smoothed coherent energy (the back-projection image),
 *                smoothed incoherent energy
 *   .masks.u8    n x n true obstacle mask (1 = rigid)
 *   .bp.u8       n x n back-projection estimate (the loop's current method)
 *   .jsonl       seed, objects, zones, back-projection IoU, sensing time
 *
 * The split "demo" holds the four epochs of the demo scenario instead, and
 * the split "ood" (robustness check only, never trained on) draws shapes
 * outside the training family: discs, L-shapes and diagonal walls.
 * `--snr <dB>` adds white noise to the room recordings (the empty-room
 * reference stays clean, as if averaged), with standard deviation
 * 10^(-snr/20) x the scene's peak residual; `--name` names the output files
 * (default: the split). The script resumes: scenes already listed in the
 * .jsonl are skipped.
 */

import { appendFileSync, existsSync, mkdirSync, readFileSync, truncateSync } from 'node:fs';
import { backProjectionEstimate, maskIou, migrationImages, pingRecordings, senseSteps } from '../src/loop/closedLoop';
import { demoScenario, mulberry32, oodLoopScene, randomLoopScene } from '../src/loop/scenarios';
import type { Geometry, Rect } from '../src/control/soundfield';

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : def;
}

const SPLIT_BASE: Record<string, number> = { train: 1_000_000, val: 2_000_000, test: 3_000_000, ood: 4_000_000 };

const split = arg('split', 'train');
const count = Number(arg('count', '10'));
const n = Number(arg('n', '100'));
const out = arg('out', '../data/loop_sensing');
const name = arg('name', split);
const snr = process.argv.includes('--snr') ? Number(arg('snr', '0')) : null;
const f0 = 0.08;
mkdirSync(out, { recursive: true });

const demo = demoScenario(n);
const array = demo.array;
const params = demo.params;
const steps = senseSteps(params);
const empty: Geometry = { params, materials: new Uint8Array(n * n), speed: null };
const recEmpty = pingRecordings(empty, array, f0, steps);

const base = `${out}/${name}`;
const done = existsSync(`${base}.jsonl`) ? readFileSync(`${base}.jsonl`, 'utf8').split('\n').filter((l) => l.trim()).length : 0;
// Keep the binary files consistent with the index (a killed run may have written a partial scene).
const trunc = (ext: string, bytes: number) => existsSync(`${base}.${ext}`) && truncateSync(`${base}.${ext}`, done * bytes);
trunc('images.f32', 5 * n * n * 4);
trunc('masks.u8', n * n);
trunc('bp.u8', n * n);

interface Item {
  seed: number;
  label?: string;
  truth: Geometry;
  bright: Rect;
  dark: Rect;
  objects?: unknown;
}
const items = (): Item[] => {
  if (split === 'demo') return demo.epochs.map((e, i) => ({ seed: -1 - i, label: e.label, truth: e.truth, bright: e.bright, dark: e.dark }));
  return Array.from({ length: count }, (_, i) => {
    const sc = split === 'ood' ? oodLoopScene(SPLIT_BASE.ood + i) : randomLoopScene(SPLIT_BASE[split] + i, n);
    return { seed: sc.seed, truth: sc.truth, bright: sc.bright, dark: sc.dark, objects: sc.objects };
  });
};

/** Standard normal samples (Box-Muller) from a seeded uniform generator. */
function gauss(rnd: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
}

const all = items();
console.log(`${split}: ${all.length} scenes, ${done} done, ${steps} steps per ping`);
const t00 = performance.now();
for (let i = done; i < all.length; i++) {
  const it = all[i];
  const t0 = performance.now();
  const rec = pingRecordings(it.truth, array, f0, steps);
  let sigma = 0;
  if (snr !== null) {
    let peak = 0;
    rec.forEach((rs, a) => rs.forEach((r, m) => r.forEach((v, k) => (peak = Math.max(peak, Math.abs(v - recEmpty[a][m][k]))))));
    sigma = peak * 10 ** (-snr / 20);
    const rnd = mulberry32(it.seed * 31 + 7);
    for (const rs of rec) for (const r of rs) for (let k = 0; k < r.length; k++) r[k] += sigma * gauss(rnd);
  }
  const im = migrationImages(rec, recEmpty, array, params, f0);
  const bp = backProjectionEstimate(im.image, array, params.shape);
  const ms = performance.now() - t0;
  const buf = new Float32Array(5 * n * n);
  [im.coherent, im.left, im.right, im.image, im.incoherent].forEach((a, c) => buf.set(a, c * n * n));
  const mask = new Uint8Array(n * n);
  for (let q = 0; q < n * n; q++) mask[q] = it.truth.materials[q] ? 1 : 0;
  appendFileSync(`${base}.images.f32`, Buffer.from(buf.buffer));
  appendFileSync(`${base}.masks.u8`, Buffer.from(mask.buffer));
  appendFileSync(`${base}.bp.u8`, Buffer.from(bp.buffer));
  const meta = { index: i, seed: it.seed, label: it.label, n, objects: it.objects, bright: it.bright, dark: it.dark, bp_iou: maskIou(bp, mask), snr, sigma, sense_ms: Math.round(ms) };
  appendFileSync(`${base}.jsonl`, `${JSON.stringify(meta)}\n`);
  if ((i + 1) % 25 === 0 || i === all.length - 1) {
    const rate = (performance.now() - t00) / (i + 1 - done);
    console.log(`${split} ${i + 1}/${all.length}  ${(rate / 1000).toFixed(2)} s/scene  eta ${(((all.length - i - 1) * rate) / 60000).toFixed(1)} min`);
  }
}
console.log('done');
