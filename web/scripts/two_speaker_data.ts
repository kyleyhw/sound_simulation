/**
 * Training and test data for the two-speaker study (tests/reports/
 * two_speaker_2026_09_26.md). Runs web/src/twospeaker (the loop's FDTD
 * engine, the emission schedules, separation and migration) in Node:
 *
 *   npx vite build --ssr scripts/two_speaker_data.ts --outDir .render/ts-data --emptyOutDir --logLevel warn
 *   node .render/ts-data/two_speaker_data.js --split train --count 1000 \
 *     --schemes seq,wide_seq,sum,band,code,seq_k2,seq_k4,seq_rigid,seq_rigid_img [--out ../data/two_speaker]
 *
 * Scenes are the loop's (scenarios.randomLoopScene, seed = split base + i:
 * train 1e6, val 2e6, test 3e6), so every scheme sees the same rooms as the
 * bar data in data/loop_sensing. Per scheme and scene it appends to
 * <out>/<scheme>/<name>.* the same files as loop_sensing_data.ts:
 *
 *   .images.f32  5 x n x n: coherent, left, right, smoothed coherent energy, smoothed incoherent energy
 *   .masks.u8    n x n true obstacle mask
 *   .bp.u8       n x n back-projection estimate (threshold 0.7 x max, largest blob, dilated)
 *   .jsonl       seed, back-projection IoU, measurement steps, separation SDR vs the sequential pings, noise
 *
 * and writes <out>/<scheme>/device.json (element positions, schedule).
 * Every shot is simulated once per scene and room and shared between the
 * schemes that play it (the sequential pings of seq, seq_k2, seq_k4).
 * `--snr <dB>` adds white noise to every room recording with standard
 * deviation 10^(-snr/20) x the peak residual of the narrow device's two
 * sequential pings (the same absolute noise floor for every scheme); the
 * empty-room references stay clean. The script resumes per scheme.
 */

import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  truncateSync,
  writeFileSync,
} from "node:fs";
import type { Geometry } from "../src/control/soundfield";
import { randomLoopScene } from "../src/loop/scenarios";
import {
  placementShots,
  type Scheme,
  scheme,
  schemeElements,
  schemeSteps,
} from "../src/twospeaker/device";
import {
  inSchemeRoom,
  type Recorder,
  recordScheme,
  residualRecordings,
  type SchemeRecordings,
  senseTwoSpeaker,
  simulateShot,
} from "../src/twospeaker/sensing";
import { sdrDb } from "../src/twospeaker/separation";

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : def;
}

const SPLIT_BASE: Record<string, number> = {
  train: 1_000_000,
  val: 2_000_000,
  test: 3_000_000,
};
const split = arg("split", "train");
const count = Number(arg("count", "10"));
const out = arg("out", "../data/two_speaker");
const name = arg("name", split);
const snr = process.argv.includes("--snr") ? Number(arg("snr", "0")) : null;
const schemes: Scheme[] = arg("schemes", "seq")
  .split(",")
  .map((s) => scheme(s));
const n = 100;

const scene0 = randomLoopScene(SPLIT_BASE[split], n);
const emptyOf = (g: Geometry): Geometry => ({
  params: g.params,
  materials: new Uint8Array(n * n),
  speed: null,
});

// All mic positions of all schemes (and of the narrow reference): every shot records them all once.
const seqRef = scheme("seq");
const allMics = [
  ...new Map(
    [...schemes, seqRef]
      .flatMap((s) => s.placements.flatMap((p) => p.mics))
      .map((m) => [m.join(","), m]),
  ).values(),
];
const micIndex = new Map(allMics.map((m, i) => [m.join(","), i]));

/** A recorder that simulates each (room, shot) once, recording every mic of every scheme. */
function cachedRecorder(): Recorder {
  const cache = new Map<string, Float32Array[]>();
  return (g, shot, mics) => {
    const key = `${g.params.outer}|${shot.key}`;
    let rec = cache.get(key);
    if (!rec) {
      rec = simulateShot(g, shot, allMics);
      cache.set(key, rec);
    }
    return mics.map((m) => rec![micIndex.get(m.join(","))!]);
  };
}

// Empty-room references (clean), per room kind, shared across schemes.
const emptyRecorder = cachedRecorder();
const empties = new Map<string, SchemeRecordings>();
for (const s of schemes)
  empties.set(
    s.name,
    recordScheme(inSchemeRoom(s, emptyOf(scene0.truth)), s, {
      recorder: emptyRecorder,
    }),
  );
const seqEmpty = recordScheme(emptyOf(scene0.truth), seqRef, {
  recorder: emptyRecorder,
});

const base = (s: Scheme) => `${out}/${s.name}/${name}`;
const IMG = 5 * n * n * 4;
const doneOf = (s: Scheme) =>
  existsSync(`${base(s)}.jsonl`)
    ? readFileSync(`${base(s)}.jsonl`, "utf8")
        .split("\n")
        .filter((l) => l.trim()).length
    : 0;
for (const s of schemes) {
  mkdirSync(`${out}/${s.name}`, { recursive: true });
  const d = doneOf(s);
  const trunc = (ext: string, bytes: number) =>
    existsSync(`${base(s)}.${ext}`) &&
    truncateSync(`${base(s)}.${ext}`, d * bytes);
  trunc("images.f32", IMG);
  trunc("masks.u8", n * n);
  trunc("bp.u8", n * n);
  writeFileSync(
    `${out}/${s.name}/device.json`,
    JSON.stringify(
      {
        scheme: s,
        elements: schemeElements(s),
        steps: schemeSteps(s),
        shots: s.placements.map((p) =>
          placementShots(s, p).map((sh) => ({
            key: sh.key,
            steps: sh.steps,
            drivers: sh.drivers.map((dr) => dr.pos),
          })),
        ),
      },
      null,
      1,
    ),
  );
}
const start = Math.min(...schemes.map(doneOf));
console.log(
  `${split}/${name}: ${count} scenes from ${start}; schemes ${schemes.map((s) => s.name).join(", ")}; mics ${allMics.length}`,
);

const t00 = performance.now();
for (let i = start; i < count; i++) {
  const sc = randomLoopScene(SPLIT_BASE[split] + i, n);
  const recorder = cachedRecorder();
  const t0 = performance.now();
  // The sequential pings of the narrow device: the separation reference and the noise scale.
  const seqRes = residualRecordings(
    recordScheme(sc.truth, seqRef, { recorder }),
    seqEmpty,
  )[0];
  let sigma = 0;
  if (snr !== null) {
    let peak = 0;
    for (const sh of seqRes)
      for (const r of sh) for (const v of r) peak = Math.max(peak, Math.abs(v));
    sigma = peak * 10 ** (-snr / 20);
  }
  const noise = snr !== null ? { sigma, seed: sc.seed } : undefined;
  const mask = new Uint8Array(n * n);
  for (let q = 0; q < n * n; q++) mask[q] = sc.truth.materials[q] ? 1 : 0;
  const times: Record<string, number> = {};
  for (const s of schemes) {
    if (i < doneOf(s)) continue;
    const ts = performance.now();
    const r = senseTwoSpeaker(sc.truth, s, {
      empty: empties.get(s.name),
      recorder,
      noise,
      truth: mask,
    });
    // Separation quality of simultaneous schemes against the (clean) sequential pings.
    let sdr: number | null = null;
    if (
      (s.kind === "code" || s.kind === "beams") &&
      s.placements.length === 1 &&
      s.room !== "rigid"
    ) {
      const v = r.traces.map((t) =>
        sdrDb(
          t.data,
          seqRes[t.speaker][
            s.placements[0].mics.findIndex((m) => m[1] === t.mic[1])
          ],
        ),
      );
      sdr = v.reduce((a, b) => a + b, 0) / v.length;
    }
    const buf = new Float32Array(5 * n * n);
    [
      r.images.coherent,
      r.images.left,
      r.images.right,
      r.images.image,
      r.images.incoherent,
    ].forEach((a, c) => buf.set(a, c * n * n));
    appendFileSync(`${base(s)}.images.f32`, Buffer.from(buf.buffer));
    appendFileSync(`${base(s)}.masks.u8`, Buffer.from(mask.buffer));
    appendFileSync(`${base(s)}.bp.u8`, Buffer.from(r.backprojection.buffer));
    const ms = Math.round(performance.now() - ts);
    times[s.name] = ms;
    const meta = {
      index: i,
      seed: sc.seed,
      n,
      scheme: s.name,
      bp_iou: r.iou!.backprojection,
      steps: r.steps,
      sdr_db: sdr,
      snr,
      sigma,
      ms,
    };
    appendFileSync(`${base(s)}.jsonl`, `${JSON.stringify(meta)}\n`);
  }
  if ((i + 1) % 10 === 0 || i === count - 1) {
    const rate = (performance.now() - t00) / (i + 1 - start);
    console.log(
      `${name} ${i + 1}/${count}  ${(rate / 1000).toFixed(2)} s/scene  eta ${(((count - i - 1) * rate) / 60000).toFixed(1)} min  last ${Math.round(performance.now() - t0)} ms ${JSON.stringify(times)}`,
    );
  }
}
console.log("done");
