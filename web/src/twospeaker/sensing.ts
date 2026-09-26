/**
 * Two-speaker room sensing, end to end (study tests/reports/
 * two_speaker_2026_09_26.md): emission schedule -> recordings (the loop's
 * FDTD engine) -> residual against the empty room -> separation of
 * simultaneous emissions -> grid-aligned migration images -> U-Net.
 *
 * The images have the loop's five-image layout (closedLoop.MigrationImages)
 * so the loop's U-Net recipe and TypeScript runtime (learnedSensing.LoopUNet)
 * apply unchanged:
 *   coherent    signed delay-and-sum over all (source, mic) paths
 *   left/right  the same over the paths of the left / right half of the
 *               sources (by column); with image sources: direct paths /
 *               first-order wall-image paths
 *   image       coherent energy, box-smoothed (the back-projection image)
 *   incoherent  sum of squared residuals at the travel lag, box-smoothed
 * `pairMigration` reproduces closedLoop.migrationImages exactly for the bar
 * (tests/unit/twoSpeaker.test.ts).
 */

import type { Geometry } from "../control/soundfield";
import { Simulation, type SimParams } from "../engine/simulation";
import {
  backProjectionEstimate,
  maskIou,
  type MigrationImages,
} from "../loop/closedLoop";
import type { LoopUNet } from "../loop/learnedSensing";
import { mulberry32 } from "../loop/scenarios";
import {
  BAND_DELAY,
  BEAM_SHOTS,
  DT,
  F0,
  LISTEN,
  type Placement,
  placementShots,
  RICKER_DELAY,
  type Scheme,
  scheme,
  schemeCodes,
  schemeElements,
  type Shot,
} from "./device";
import {
  bandSplit,
  beamSeparate,
  lsSeparate,
  mfSeparate,
  toRicker,
} from "./separation";

/** Records the given mics while a shot plays in a room. */
export type Recorder = (
  g: Geometry,
  shot: Shot,
  mics: number[][],
) => Float32Array[];

/** Simulate one shot with the loop's engine (samples waveforms at the simulation rate, soft sources). */
export const simulateShot: Recorder = (g, shot, mics) => {
  const sim = new Simulation(g.params);
  sim.setMaterialMap(g.materials);
  if (g.speed) sim.setSpeedMap(g.speed);
  const rate = 1 / sim.dt;
  sim.setDrivers(
    shot.drivers.map((d, i) => ({
      id: `d${i}`,
      pos: d.pos,
      // 'samples' returns data[k] at step k exactly (x = k, zero weight on k + 1); the trailing 0 keeps the last sample.
      waveform: {
        type: "samples" as const,
        amplitude: 1,
        rate,
        delay: 0,
        data: [...Array.from(d.data), 0],
      },
      enabled: true,
    })),
  );
  const idx = mics.map((p) => sim.index(p));
  const rec = mics.map(() => new Float32Array(shot.steps));
  for (let k = 0; k < shot.steps; k++) {
    sim.step();
    for (let m = 0; m < idx.length; m++) rec[m][k] = sim.p[idx[m]];
  }
  return rec;
};

/** The scheme's room: the loop's parameters, with rigid outer walls for the '*_rigid' schemes. */
export function schemeParams(s: Scheme, base: SimParams): SimParams {
  return s.room === "rigid" ? { ...base, outer: "rigid" } : base;
}

/** The geometry `g` moved into the scheme's room (same obstacles). */
export function inSchemeRoom(s: Scheme, g: Geometry): Geometry {
  return { ...g, params: schemeParams(s, g.params) };
}

/** Recordings [placement][shot][mic] of a scheme's whole emission schedule. */
export type SchemeRecordings = Float32Array[][][];

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++)
    h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

/** Deterministic white Gaussian noise for one (shot, mic) recording. */
export function addNoise(
  x: Float32Array,
  sigma: number,
  seed: number,
  key: string,
): Float32Array {
  const out = Float32Array.from(x);
  if (!(sigma > 0)) return out;
  const rnd = mulberry32(hash(`${seed}|${key}`));
  for (let k = 0; k < out.length; k++)
    out[k] +=
      sigma *
      Math.sqrt(-2 * Math.log(1 - rnd())) *
      Math.cos(2 * Math.PI * rnd());
  return out;
}

export function recordScheme(
  g: Geometry,
  s: Scheme,
  opts: { recorder?: Recorder; noise?: { sigma: number; seed: number } } = {},
): SchemeRecordings {
  const rec = opts.recorder ?? simulateShot;
  return s.placements.map((p) =>
    placementShots(s, p).map((shot) => {
      const r = rec(g, shot, p.mics);
      return opts.noise
        ? r.map((x, m) =>
            addNoise(
              x,
              opts.noise!.sigma,
              opts.noise!.seed,
              `${shot.key}|${p.mics[m].join(",")}`,
            ),
          )
        : r;
    }),
  );
}

/** Residual recordings (room minus empty room, rounded to float32 as in the loop). */
export function residualRecordings(
  room: SchemeRecordings,
  empty: SchemeRecordings,
): SchemeRecordings {
  return room.map((pl, a) =>
    pl.map((sh, b) => sh.map((r, m) => r.map((v, k) => v - empty[a][b][m][k]))),
  );
}

/** A separated trace: what `mic` records from `src` alone (Ricker-equivalent, or band-limited), pulse centre at t0 (time units). */
export interface Trace {
  src: number[];
  mic: number[];
  data: Float32Array;
  t0: number;
  /** Speaker index within the placement (0 = A, 1 = B). */
  speaker: number;
}

/** Separate one placement's residual shots into per-(speaker, mic) traces. */
export function placementTraces(
  s: Scheme,
  p: Placement,
  res: Float32Array[][],
): Trace[] {
  const out: Trace[] = [];
  const add = (sp: number, m: number, data: Float32Array, t0 = RICKER_DELAY) =>
    out.push({ src: p.speakers[sp], mic: p.mics[m], data, t0, speaker: sp });
  switch (s.kind) {
    case "seq":
      p.speakers.forEach((_, sp) =>
        p.mics.forEach((_, m) => add(sp, m, res[sp][m])),
      );
      break;
    case "sum":
      // Not separable: migrate the summed trace under both source hypotheses.
      p.speakers.forEach((_, sp) =>
        p.mics.forEach((_, m) => add(sp, m, res[0][m])),
      );
      break;
    case "band": {
      const split = p.mics.map((_, m) => bandSplit(res[0][m]));
      p.mics.forEach((_, m) => add(0, m, split[m].low, BAND_DELAY * DT));
      p.mics.forEach((_, m) => add(1, m, split[m].high, BAND_DELAY * DT));
      break;
    }
    case "code": {
      const codes = schemeCodes(s);
      const sep = p.mics.map((_, m) =>
        s.separation === "mf"
          ? mfSeparate(res[0][m], codes)
          : lsSeparate(res[0][m], codes, LISTEN + 64).u.map((u) => toRicker(u)),
      );
      p.speakers.forEach((_, sp) =>
        p.mics.forEach((_, m) => add(sp, m, sep[m][sp])),
      );
      break;
    }
    case "beams": {
      const sep = p.mics.map((_, m) =>
        beamSeparate(
          res.map((sh) => sh[m]),
          BEAM_SHOTS,
        ),
      );
      p.speakers.forEach((_, sp) =>
        p.mics.forEach((_, m) => add(sp, m, sep[m][sp])),
      );
      break;
    }
  }
  return out;
}

/** One migration path: a trace migrated from `src` to `mic` (possibly wall images), summed into image `group` (0 left, 1 right). */
export interface Path {
  src: number[];
  mic: number[];
  data: Float32Array;
  t0: number;
  group: 0 | 1;
}

/** First-order images of a point across the four rigid outer walls (half a cell outside the edge cells). */
export function wallImages(p: number[], shape: number[]): number[][] {
  const [rows, cols] = shape;
  return [
    [-1 - p[0], p[1]],
    [2 * rows - 1 - p[0], p[1]],
    [p[0], -1 - p[1]],
    [p[0], 2 * cols - 1 - p[1]],
  ];
}

/** Migration paths of a scheme's traces: groups by source column (left / right half), or direct / image paths. */
export function schemePaths(
  s: Scheme,
  traces: Trace[],
  shape: number[],
): Path[] {
  if (s.imageSources) {
    const out: Path[] = [];
    for (const t of traces) {
      out.push({ ...t, group: 0 });
      for (const si of wallImages(t.src, shape))
        out.push({ ...t, src: si, group: 1 });
      for (const mi of wallImages(t.mic, shape))
        out.push({ ...t, mic: mi, group: 1 });
    }
    return out;
  }
  const cols = [...new Set(traces.map((t) => t.src[1]))].sort((a, b) => a - b);
  const half = cols.length / 2;
  return traces.map((t) => ({
    ...t,
    group: cols.indexOf(t.src[1]) < half ? 0 : 1,
  }));
}

function boxSmooth(
  src: Float32Array,
  rows: number,
  cols: number,
  r: number,
  square: boolean,
): Float32Array {
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
 * Delay-and-sum migration over explicit paths (a generalisation of
 * closedLoop.migrationImages, which it reproduces for the bar):
 * I(x) = sum_paths data(t0 + (|x - src| + |x - mic|) dx / c).
 */
export function pairMigration(
  paths: Path[],
  params: SimParams,
  f0 = F0,
): MigrationImages {
  const [rows, cols] = params.shape;
  const dt = new Simulation(params).dt;
  const cdx = params.dx / params.c;
  const coherent = new Float32Array(rows * cols);
  const left = new Float32Array(rows * cols);
  const right = new Float32Array(rows * cols);
  const inco = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      let acc = 0;
      let accL = 0;
      let acc2 = 0;
      for (const p of paths) {
        const k = Math.round(
          (p.t0 +
            Math.hypot(i - p.src[0], j - p.src[1]) * cdx +
            Math.hypot(i - p.mic[0], j - p.mic[1]) * cdx) /
            dt,
        );
        if (k < p.data.length) {
          const v = p.data[k];
          acc += v;
          if (p.group === 0) accL += v;
          acc2 += v * v;
        }
      }
      coherent[i * cols + j] = acc;
      left[i * cols + j] = accL;
      right[i * cols + j] = acc - accL;
      inco[i * cols + j] = acc2;
    }
  const r = Math.max(1, Math.round((0.25 / f0 / params.dx) * params.c));
  return {
    coherent,
    left,
    right,
    image: boxSmooth(coherent, rows, cols, r, true),
    incoherent: boxSmooth(inco, rows, cols, r, false),
  };
}

export interface TwoSpeakerResult {
  images: MigrationImages;
  /** Device elements (all placements): the near field and centre of the features. */
  elements: number[][];
  traces: Trace[];
  backprojection: Uint8Array;
  estimate: Uint8Array | null;
  probability: Float32Array | null;
  iou: { backprojection: number; learned: number | null } | null;
  /** Measurement time: seconds of sound in simulation steps (all shots). */
  steps: number;
}

/**
 * Sense a room with a two-speaker scheme. `empty` holds the empty-room
 * recordings of the same scheme (in the same room; computed if absent),
 * `noise` adds white noise to the room recordings (the reference stays clean,
 * as if averaged), `model` runs the U-Net on the images, `truth` scores.
 */
export function senseTwoSpeaker(
  room: Geometry,
  s: Scheme,
  opts: {
    empty?: SchemeRecordings;
    recorder?: Recorder;
    noise?: { sigma: number; seed: number };
    model?: LoopUNet | null;
    truth?: Uint8Array;
  } = {},
): TwoSpeakerResult {
  const g = inSchemeRoom(s, room);
  const empty =
    opts.empty ??
    recordScheme(
      {
        params: g.params,
        materials: new Uint8Array(g.materials.length),
        speed: null,
      },
      s,
      { recorder: opts.recorder },
    );
  const rec = recordScheme(g, s, {
    recorder: opts.recorder,
    noise: opts.noise,
  });
  const res = residualRecordings(rec, empty);
  const traces = s.placements.flatMap((p, a) => placementTraces(s, p, res[a]));
  const images = pairMigration(
    schemePaths(s, traces, g.params.shape),
    g.params,
  );
  const elements = schemeElements(s);
  const backprojection = backProjectionEstimate(
    images.image,
    elements,
    g.params.shape,
  );
  const learned = opts.model ? opts.model.estimate(images, elements) : null;
  const steps = s.placements.reduce(
    (t, p) => t + placementShots(s, p).reduce((a, sh) => a + sh.steps, 0),
    0,
  );
  return {
    images,
    elements,
    traces,
    backprojection,
    estimate: learned?.estimate ?? null,
    probability: learned?.probability ?? null,
    iou: opts.truth
      ? {
          backprojection: maskIou(backprojection, opts.truth),
          learned: learned ? maskIou(learned.estimate, opts.truth) : null,
        }
      : null,
    steps,
  };
}

/**
 * A two-speaker sensor for the browser: the exported U-Net of one scheme
 * (web/public/models/two_speaker_<scheme>.{json,bin}, loop-unet-v1 format
 * plus `scheme` and `device`) and its empty-room reference, cached per room
 * parameters. `sense` runs emission schedule -> recordings -> separation ->
 * images -> U-Net on a room of the loop's 100 x 100 grid.
 */
export class TwoSpeakerSensor {
  readonly scheme: Scheme;
  private empties = new Map<string, SchemeRecordings>();

  constructor(readonly model: LoopUNet) {
    const name = (model.manifest as unknown as { scheme?: string }).scheme;
    if (!name)
      throw new Error("not a two-speaker model (manifest has no scheme)");
    this.scheme = scheme(name);
  }

  /** Empty-room recordings of the scheme in the room of `g` (computed once per room parameters). */
  emptyFor(g: Geometry): SchemeRecordings {
    const gg = inSchemeRoom(this.scheme, g);
    const key = JSON.stringify(gg.params);
    let e = this.empties.get(key);
    if (!e) {
      e = recordScheme(
        {
          params: gg.params,
          materials: new Uint8Array(gg.materials.length),
          speed: null,
        },
        this.scheme,
      );
      this.empties.set(key, e);
    }
    return e;
  }

  sense(
    room: Geometry,
    opts: { noise?: { sigma: number; seed: number }; truth?: Uint8Array } = {},
  ): TwoSpeakerResult {
    const shape = this.model.shape;
    if (!room.params.shape.every((v, i) => v === shape[i]))
      throw new Error(`two-speaker model is trained for ${shape.join("x")}`);
    return senseTwoSpeaker(room, this.scheme, {
      empty: this.emptyFor(room),
      model: this.model,
      ...opts,
    });
  }
}
