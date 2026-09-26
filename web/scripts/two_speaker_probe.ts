/**
 * Two-speaker study probe: timing, linearity and separation quality on a few
 * random loop scenes (test seeds 3e6 + i):
 *
 *   npx vite build --ssr scripts/two_speaker_probe.ts --outDir .render/ts-probe --emptyOutDir --logLevel warn \
 *     && node .render/ts-probe/two_speaker_probe.js [--count 4] [--out ../data/two_speaker/probe.json]
 *
 * For each scene: the separated traces of every simultaneous scheme are
 * scored against the ideal ones from sequential pings (SDR, dB), for codes of
 * several lengths, with joint least squares and the naive inverse filter.
 */

import { writeFileSync } from "node:fs";
import type { Geometry } from "../src/control/soundfield";
import { randomLoopScene } from "../src/loop/scenarios";
import {
  LISTEN,
  placementShots,
  scheme,
  schemeCodes,
} from "../src/twospeaker/device";
import {
  placementTraces,
  recordScheme,
  residualRecordings,
  simulateShot,
} from "../src/twospeaker/sensing";
import {
  lsSeparate,
  mfSeparate,
  sdrDb,
  toRicker,
} from "../src/twospeaker/separation";

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : def;
}
const count = Number(arg("count", "4"));
const out = arg("out", "../data/two_speaker/probe.json");
const codeLens = arg("codes", "311,622,1244").split(",").map(Number);
const iterList = arg("iters", "200").split(",").map(Number);

const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const results: Record<string, number[]> = {};
const push = (k: string, v: number) => (results[k] ??= []).push(v);

const seq = scheme("seq");
const empty0 = (g: Geometry): Geometry => ({
  params: g.params,
  materials: new Uint8Array(g.materials.length),
  speed: null,
});
const t0 = performance.now();
const sc0 = randomLoopScene(3_000_000, 100);
const eSeq = recordScheme(empty0(sc0.truth), seq);
console.log(`empty seq (2 shots): ${(performance.now() - t0).toFixed(0)} ms`);
const emptyCache = new Map<string, ReturnType<typeof recordScheme>>();
const emptyOf = (name: string, s: ReturnType<typeof scheme>, g: Geometry) => {
  if (!emptyCache.has(name)) emptyCache.set(name, recordScheme(empty0(g), s));
  return emptyCache.get(name)!;
};

for (let i = 0; i < count; i++) {
  const sc = randomLoopScene(3_000_000 + i, 100);
  const tA = performance.now();
  const rSeq = residualRecordings(recordScheme(sc.truth, seq), eSeq)[0];
  push("ms_seq_2shots", performance.now() - tA);
  const ideal = [rSeq[0], rSeq[1]]; // [speaker][mic]
  // sum: linearity check (simultaneous same pulse = sum of the two pings)
  const sSum = scheme("sum");
  const rSum = residualRecordings(
    recordScheme(sc.truth, sSum),
    emptyOf("sum", sSum, sc.truth),
  )[0][0];
  for (let m = 0; m < 2; m++) {
    let e = 0;
    let s = 0;
    for (let k = 0; k < LISTEN; k++) {
      e = Math.max(e, Math.abs(rSum[m][k] - ideal[0][m][k] - ideal[1][m][k]));
      s = Math.max(s, Math.abs(ideal[0][m][k] + ideal[1][m][k]));
    }
    push("linearity_rel_maxerr", e / s);
  }
  // band: ideal = each band pulse emitted alone
  const sBand = scheme("band");
  const shotB = placementShots(sBand, sBand.placements[0])[0];
  const eB = emptyOf("band", sBand, sc.truth)[0][0];
  const rBand = residualRecordings(
    recordScheme(sc.truth, sBand),
    emptyOf("band", sBand, sc.truth),
  )[0];
  const tr = placementTraces(sBand, sBand.placements[0], rBand);
  const alone = shotB.drivers.map((d, sp) => {
    const sh = { drivers: [d], steps: shotB.steps, key: `alone${sp}` };
    const room = simulateShot(sc.truth, sh, sBand.placements[0].mics);
    const emp = simulateShot(empty0(sc.truth), sh, sBand.placements[0].mics);
    return room.map((r, m) => r.map((v, k) => v - emp[m][k]));
  });
  void eB;
  for (const t of tr) {
    const m = sBand.placements[0].mics.findIndex((q) => q[1] === t.mic[1]);
    push(
      `sdr_band_${t.speaker === 0 ? "low" : "high"}`,
      sdrDb(t.data, alone[t.speaker][m]),
    );
  }
  // beams
  const sBeam = scheme("beams");
  const rBeam = residualRecordings(
    recordScheme(sc.truth, sBeam),
    emptyOf("beams", sBeam, sc.truth),
  )[0];
  for (const t of placementTraces(sBeam, sBeam.placements[0], rBeam)) {
    const m = sBeam.placements[0].mics.findIndex((q) => q[1] === t.mic[1]);
    push("sdr_beams", sdrDb(t.data, ideal[t.speaker][m]));
  }
  // codes
  for (const L of codeLens)
    for (const code of ["noise", "chirp"] as const) {
      const sCode = { ...scheme("code", L), code };
      const tc = performance.now();
      const rCode = residualRecordings(
        recordScheme(sc.truth, sCode),
        emptyOf(`${code}${L}`, sCode, sc.truth),
      )[0][0];
      push(`ms_sim_${code}${L}`, performance.now() - tc);
      const codes = schemeCodes(sCode);
      for (let m = 0; m < 2; m++) {
        for (const iters of iterList) {
          const tl = performance.now();
          const ls = lsSeparate(rCode[m], codes, LISTEN + 64, { iters });
          push(`ms_ls${iters}_${code}${L}`, performance.now() - tl);
          push(`relres_ls${iters}_${code}${L}`, ls.relResidual);
          for (let sp = 0; sp < 2; sp++)
            push(
              `sdr_ls${iters}_${code}${L}`,
              sdrDb(toRicker(ls.u[sp]), ideal[sp][m]),
            );
        }
        const mf = mfSeparate(rCode[m], codes);
        for (let sp = 0; sp < 2; sp++)
          push(`sdr_mf_${code}${L}`, sdrDb(mf[sp], ideal[sp][m]));
      }
    }
  console.log(
    `scene ${i}: ${Object.entries(results)
      .map(
        ([k, v]) =>
          `${k}=${v[v.length - 1].toFixed(k.startsWith("ms") ? 0 : 2)}`,
      )
      .join(" ")}`,
  );
}
const summary = Object.fromEntries(
  Object.entries(results).map(([k, v]) => [
    k,
    { mean: mean(v), min: Math.min(...v), max: Math.max(...v), n: v.length },
  ]),
);
console.log(JSON.stringify(summary, null, 1));
writeFileSync(out, JSON.stringify({ count, summary, results }, null, 1));
