/**
 * Two-speaker sensing module (web/src/twospeaker): device geometry, the
 * emission schedules, separation of simultaneous emissions, linearity of the
 * engine, and the generalised migration against the loop's.
 */
import { describe, expect, it } from "vitest";
import type { Geometry } from "../../src/control/soundfield";
import {
  migrationImages,
  pingRecordings,
  senseSteps,
} from "../../src/loop/closedLoop";
import {
  demoScenario,
  mulberry32,
  randomLoopScene,
} from "../../src/loop/scenarios";
import {
  bandPulses,
  BEAM_SHOTS,
  chirpCode,
  laptop,
  LISTEN,
  noiseCode,
  placementShots,
  rickerSamples,
  scheme,
  schemeElements,
  schemeSteps,
} from "../../src/twospeaker/device";
import {
  pairMigration,
  recordScheme,
  residualRecordings,
  schemePaths,
  placementTraces,
  wallImages,
} from "../../src/twospeaker/sensing";
import {
  bandSplit,
  beamSeparate,
  lsSeparate,
  sdrDb,
  toRicker,
} from "../../src/twospeaker/separation";

/** Causal convolution truncated to len. */
function conv(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  len: number,
): Float64Array {
  const out = new Float64Array(len);
  for (let i = 0; i < a.length; i++)
    if (a[i])
      for (let j = 0; j < b.length && i + j < len; j++)
        out[i + j] += a[i] * b[j];
  return out;
}

/** A sparse random impulse response (a few echoes) of length L. */
function sparseIr(L: number, seed: number): Float64Array {
  const rnd = mulberry32(seed);
  const g = new Float64Array(L);
  for (let e = 0; e < 12; e++)
    g[40 + Math.floor(rnd() * (L - 60))] += rnd() * 2 - 1;
  return g;
}

describe("two-speaker device and schedules", () => {
  it("places the laptop-like device on the bar row", () => {
    expect(laptop(45)).toEqual({
      speakers: [
        [86, 39],
        [86, 51],
      ],
      mics: [
        [86, 41],
        [86, 49],
      ],
    });
    expect(laptop(45, 28)).toEqual({
      speakers: [
        [86, 31],
        [86, 59],
      ],
      mics: [
        [86, 33],
        [86, 57],
      ],
    });
    expect(
      scheme("seq_k4").placements.map((p) => p.speakers.map((s) => s[1])),
    ).toEqual([
      [21, 33],
      [33, 45],
      [45, 57],
      [57, 69],
    ]);
    expect(schemeElements(scheme("seq"))).toHaveLength(4);
  });

  it("has the documented measurement times (steps)", () => {
    expect(schemeSteps(scheme("bar8"))).toBe(8 * LISTEN);
    expect(schemeSteps(scheme("seq"))).toBe(2 * LISTEN);
    expect(schemeSteps(scheme("sum"))).toBe(LISTEN);
    expect(schemeSteps(scheme("band"))).toBe(LISTEN + 83);
    expect(schemeSteps(scheme("code"))).toBe(1244 + LISTEN);
    expect(schemeSteps(scheme("seq_k4"))).toBe(8 * LISTEN);
    expect(placementShots(scheme("beams"), laptop(45))).toHaveLength(4);
  });

  it("reproduces the engine Ricker and splits it into disjoint bands", () => {
    const r = rickerSamples(4);
    const a = Math.PI * 0.08 * -18.75;
    expect(r[0]).toBeCloseTo((1 - 2 * a * a) * Math.exp(-a * a), 15);
    const { low, high } = bandPulses();
    // Each band pulse passes its own separation filter and is (nearly) rejected by the other.
    const lo = bandSplit(low);
    const hi = bandSplit(high);
    expect(sdrDb(lo.low, low)).toBeGreaterThan(40);
    expect(sdrDb(hi.high, high)).toBeGreaterThan(40);
  });

  it("codes are unit peak and the two noise codes are nearly uncorrelated", () => {
    const a = noiseCode(1244, 1001);
    const b = noiseCode(1244, 2002);
    expect(Math.max(...a.map(Math.abs))).toBeCloseTo(1, 12);
    let ab = 0;
    let aa = 0;
    for (let k = 0; k < a.length; k++) {
      ab += a[k] * b[k];
      aa += a[k] * a[k];
    }
    expect(Math.abs(ab) / aa).toBeLessThan(0.15);
    expect(Math.max(...chirpCode(622).map(Math.abs))).toBeLessThanOrEqual(1);
  });
});

describe("separation of simultaneous emissions (synthetic LTI system)", () => {
  const L = 400;
  const gA = sparseIr(L, 1);
  const gB = sparseIr(L, 2);
  const ric = rickerSamples(L);
  const ideal = [conv(ric, gA, L), conv(ric, gB, L)];

  it("joint least squares recovers both responses when T_code + L >= 2 L_u, not when shorter", () => {
    const run = (Tc: number) => {
      const ca = noiseCode(Tc, 11);
      const cb = noiseCode(Tc, 22);
      const T = Tc + L;
      const y = conv(ca, gA, T);
      const yb = conv(cb, gB, T);
      for (let k = 0; k < T; k++) y[k] += yb[k];
      const { u } = lsSeparate(y, [ca, cb], L, { iters: 300, lambdaRel: 1e-6 });
      return u.map((x, s) => sdrDb(toRicker(x, L), ideal[s]));
    };
    const long = run(2 * L);
    const short = run(L / 4);
    expect(Math.min(...long)).toBeGreaterThan(25);
    expect(Math.max(...short)).toBeLessThan(Math.min(...long) - 10);
  });

  it("beam shots (same pulse, phase / delay steering) separate exactly into the single-speaker traces", () => {
    const len = L + 12;
    const ys = BEAM_SHOTS.map((b) => {
      const y = new Float64Array(len);
      for (let k = 0; k < len; k++)
        y[k] =
          b.a[0] * (ideal[0][k - b.a[1]] ?? 0) +
          b.b[0] * (ideal[1][k - b.b[1]] ?? 0);
      return y;
    });
    const [rA, rB] = beamSeparate(ys, BEAM_SHOTS, L);
    expect(sdrDb(rA, ideal[0])).toBeGreaterThan(60);
    expect(sdrDb(rB, ideal[1])).toBeGreaterThan(60);
  });
});

describe("two-speaker sensing in the loop room", () => {
  const sc = randomLoopScene(3_000_000, 100);
  const empty: Geometry = {
    params: sc.truth.params,
    materials: new Uint8Array(100 * 100),
    speed: null,
  };

  it("the engine is linear: both speakers at once record the sum of each alone (float32 round-off)", () => {
    const seq = recordScheme(sc.truth, scheme("seq"))[0];
    const sum = recordScheme(sc.truth, scheme("sum"))[0][0];
    for (let m = 0; m < 2; m++) {
      let e = 0;
      let s = 0;
      for (let k = 0; k < LISTEN; k++) {
        e = Math.max(e, Math.abs(sum[m][k] - seq[0][m][k] - seq[1][m][k]));
        s = Math.max(s, Math.abs(sum[m][k]));
      }
      expect(e / s).toBeLessThan(1e-5);
    }
  }, 60_000);

  it("pairMigration reproduces closedLoop.migrationImages for the bar exactly", () => {
    const demo = demoScenario(100);
    const steps = senseSteps(demo.params);
    const ref = migrationImages(
      pingRecordings(sc.truth, demo.array, 0.08, steps),
      pingRecordings(empty, demo.array, 0.08, steps),
      demo.array,
      demo.params,
      0.08,
    );
    const s = scheme("bar8");
    const res = residualRecordings(
      recordScheme(sc.truth, s),
      recordScheme(empty, s),
    );
    const im = pairMigration(
      schemePaths(s, placementTraces(s, s.placements[0], res[0]), [100, 100]),
      demo.params,
    );
    for (const k of [
      "coherent",
      "left",
      "right",
      "image",
      "incoherent",
    ] as const)
      expect(Array.from(im[k])).toEqual(Array.from(ref[k]));
  }, 120_000);

  it("image sources mirror across walls half a cell outside the grid", () => {
    expect(wallImages([86, 39], [100, 100])).toEqual([
      [-87, 39],
      [113, 39],
      [86, -40],
      [86, 160],
    ]);
    const s = scheme("seq_rigid_img");
    const tr = [
      {
        src: [86, 39],
        mic: [86, 41],
        data: new Float32Array(1),
        t0: 0,
        speaker: 0,
      },
    ];
    expect(schemePaths(s, tr, [100, 100])).toHaveLength(9);
  });
});
