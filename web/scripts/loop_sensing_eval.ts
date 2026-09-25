/**
 * Paired closed-loop evaluation of the room estimators (loop sensing study,
 * 2026-09-25):
 *
 *   npm run loop:eval -- --count 80 [--model public/models/loop_unet] [--out ../data/loop_sensing/eval.jsonl]
 *   npm run loop:eval -- --family ood --count 40 --out ../data/loop_sensing/eval_ood.jsonl
 *
 * For each held-out random scene (scenarios.randomLoopScene, test seeds
 * 3e6 + i, disjoint from the training and validation seeds) and each of the
 * four demo epochs, the bar senses the room once, and the same recordings
 * feed both estimators. Then, all measured in the true room:
 *
 *   bp       ACC designed on the back-projection twin
 *   learned  ACC designed on the learned (U-Net) twin
 *   empty    ACC designed on the empty room (no sensing)
 *   oracle   ACC designed on the true room
 *   guarded_bp / guarded_learned
 *            the monitor-mic guard between that twin's design and the
 *            empty-room design (as in runLoop)
 *
 * One JSON line per scene; the script resumes from an existing file.
 */

import { appendFileSync, existsSync, readFileSync } from 'node:fs';
import type { Geometry, Rect } from '../src/control/soundfield';
import { verifyContrast, weightedDrivers } from '../src/control/soundfield';
import { accWeights, maskIou, monitorPoints, pingRecordings, pointContrast, senseRoom, senseSteps, twinGeometry } from '../src/loop/closedLoop';
import { type LoopModelManifest, LoopUNet } from '../src/loop/learnedSensing';
import { demoScenario, oodLoopScene, randomLoopScene } from '../src/loop/scenarios';

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : def;
}

const count = Number(arg('count', '80'));
const family = arg('family', 'random'); // 'random' (test seeds 3e6 + i, plus the demo) or 'ood' (seeds 4e6 + i)
const modelBase = arg('model', 'public/models/loop_unet');
const out = arg('out', '../data/loop_sensing/eval.jsonl');
const n = 100;
const f0 = 0.08;

const buf = readFileSync(`${modelBase}.bin`);
const model = new LoopUNet(JSON.parse(readFileSync(`${modelBase}.json`, 'utf8')) as LoopModelManifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const demo = demoScenario(n);
const { array, params, frequency: f } = demo;
const empty: Geometry = { params, materials: new Uint8Array(n * n), speed: null };
const recEmpty = pingRecordings(empty, array, f0, senseSteps(params));

interface Item {
  kind: 'random' | 'demo' | 'ood';
  seed: number;
  label?: string;
  truth: Geometry;
  bright: Rect;
  dark: Rect;
}
const items: Item[] =
  family === 'ood'
    ? Array.from({ length: count }, (_, i) => {
        const sc = oodLoopScene(4_000_000 + i);
        return { kind: 'ood' as const, seed: sc.seed, truth: sc.truth, bright: sc.bright, dark: sc.dark };
      })
    : [
  ...demo.epochs.map((e, i) => ({ kind: 'demo' as const, seed: -1 - i, label: e.label, truth: e.truth, bright: e.bright, dark: e.dark })),
  ...Array.from({ length: count }, (_, i) => {
    const sc = randomLoopScene(3_000_000 + i, n);
    return { kind: 'random' as const, seed: sc.seed, truth: sc.truth, bright: sc.bright, dark: sc.dark };
  }),
      ];
const done = new Set(existsSync(out) ? readFileSync(out, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l).seed as number) : []);
console.log(`${items.length} scenes, ${done.size} done`);

const t00 = performance.now();
let k = 0;
for (const it of items) {
  if (done.has(it.seed)) continue;
  const t0 = performance.now();
  const sensed = senseRoom(it.truth, empty, array, { learned: model, recEmpty, truthObstacles: it.truth.materials });
  const tSense = performance.now() - t0;
  const tl0 = performance.now();
  model.estimate(sensed.images, array);
  const tLearned = performance.now() - tl0;
  const estimates = { bp: sensed.backprojection, learned: sensed.estimate };
  const designs: Record<string, Awaited<ReturnType<typeof accWeights>>> = {};
  for (const [name, est] of Object.entries(estimates)) designs[name] = await accWeights(twinGeometry(it.truth, est), array, it.bright, it.dark, f);
  designs.empty = await accWeights(empty, array, it.bright, it.dark, f);
  designs.oracle = await accWeights(it.truth, array, it.bright, it.dark, f);
  const measured: Record<string, number> = {};
  for (const [name, d] of Object.entries(designs)) measured[name] = verifyContrast(it.truth, weightedDrivers(array, d.w, f), it.bright, it.dark, f);
  const mb = monitorPoints(it.bright);
  const md = monitorPoints(it.dark);
  const monitor: Record<string, number> = {};
  for (const name of ['bp', 'learned', 'empty']) monitor[name] = pointContrast(it.truth, weightedDrivers(array, designs[name].w, f), mb, md, f);
  const guard = (name: string) => (monitor[name] >= monitor.empty ? { value: measured[name], choice: 'twin' } : { value: measured.empty, choice: 'empty' });
  const gb = guard('bp');
  const gl = guard('learned');
  const cells = (m: Uint8Array) => m.reduce((s, v) => s + (v ? 1 : 0), 0);
  const rec = {
    kind: it.kind,
    seed: it.seed,
    label: it.label,
    bright: it.bright,
    dark: it.dark,
    iou: { bp: maskIou(estimates.bp, it.truth.materials), learned: maskIou(estimates.learned, it.truth.materials) },
    cells: { truth: cells(it.truth.materials), bp: cells(estimates.bp), learned: cells(estimates.learned) },
    contrast: { ...measured, guarded_bp: gb.value, guarded_learned: gl.value },
    guard_choice: { bp: gb.choice, learned: gl.choice },
    monitor,
    predicted: Object.fromEntries(Object.entries(designs).map(([k2, d]) => [k2, d.predicted])),
    ms: { sense_total: Math.round(tSense), learned_inference: Math.round(tLearned), scene: Math.round(performance.now() - t0) },
    estimates: { bp: Array.from(estimates.bp.entries()).filter(([, v]) => v).map(([q]) => q), learned: Array.from(estimates.learned.entries()).filter(([, v]) => v).map(([q]) => q) },
  };
  appendFileSync(out, `${JSON.stringify(rec)}\n`);
  k++;
  const rate = (performance.now() - t00) / k;
  console.log(
    `${it.label ?? it.seed}: IoU bp ${rec.iou.bp.toFixed(2)} learned ${rec.iou.learned.toFixed(2)} | dB bp ${measured.bp.toFixed(1)} learned ${measured.learned.toFixed(1)} empty ${measured.empty.toFixed(1)} oracle ${measured.oracle.toFixed(1)} | ${(rate / 1000).toFixed(1)} s/scene`,
  );
}
console.log('done');
