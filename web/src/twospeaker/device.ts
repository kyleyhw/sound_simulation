/**
 * Two-speaker room sensing (study tests/reports/two_speaker_2026_09_26.md):
 * a laptop-like device on the bottom wall of the loop's 100 x 100 CPML room,
 * with two speakers and two microphones, and the emission schemes compared
 * against the loop's 8-element bar.
 *
 * Geometry (row 86, the bar's row; columns):
 *   narrow  speakers c - 6 and c + 6 (12 cells apart), mics c - 4 and c + 4
 *   wide    speakers 31 and 59 (the bar's ends, 28 apart), mics 33 and 57
 * with c = 45 (the bar's centre). "Moved" schemes place the narrow device at
 * K = 2 (c = 39, 51) or K = 4 (c = 27, 39, 51, 63) positions, sliding it by
 * its speaker spacing; each placement records only with its own two mics.
 *
 * Emission schemes (per placement):
 *   seq    speakers ping in turn (Ricker, f0 = 0.08); 2 shots, 4 traces
 *   sum    both speakers ping at once with the same pulse; 1 shot (only the
 *          sum is observable)
 *   band   both at once: speaker A emits the Ricker's low band (below 0.075),
 *          B its high band (above 0.085); 1 shot, separated by filtering
 *   code   both at once: two independent pseudo-noise codes with the Ricker's
 *          spectrum (or up/down chirps), 1244 steps; 1 shot, separated by
 *          joint least-squares deconvolution
 *   beams  4 same-pulse shots with relative phase / delay steering (in phase,
 *          anti-phase, B delayed, A delayed); separated per frequency
 *   bar8   the loop's bar (8 pings, 8 mics), for reference
 *
 * Room: the loop's CPML room (anechoic outside the 100 x 100 grid), or the
 * same grid with rigid outer walls ('*_rigid'; the walls sit half a cell
 * outside the edge cells, so the image of row r across the top wall is
 * -1 - r and across the bottom wall 2 n - 1 - r). The empty-room reference
 * is always simulated in the same room, so the residual holds only the
 * obstacles' scattered field (including its wall reflections).
 *
 * Times are in simulation steps (dt = 0.5 dx / c) unless marked as time units.
 */

import { fft, nextPow2 } from "../lib/dsp";
import { mulberry32 } from "../loop/scenarios";

export const DEVICE_ROW = 86;
export const F0 = 0.08;
export const DT = 0.5;
/** Listening window after a pulse: closedLoop.senseSteps for the 100 x 100 room. */
export const LISTEN = 622;
/** Ricker delay used by the loop, 1.5 / f0, in time units (37.5 steps). */
export const RICKER_DELAY = 1.5 / F0;

export interface Placement {
  speakers: number[][];
  mics: number[][];
}

export type EmissionKind = "seq" | "sum" | "band" | "code" | "beams";

export interface Scheme {
  name: string;
  kind: EmissionKind;
  placements: Placement[];
  /** Code length in steps ('code' only). */
  codeSteps?: number;
  /** Codes ('code' only): Ricker-band pseudo-noise (default) or up/down chirps. */
  code?: "noise" | "chirp";
  /** Separation of coded shots: joint least squares (default) or the naive per-speaker inverse filter. */
  separation?: "ls" | "mf";
  /** Outer walls: the loop's anechoic CPML room (default) or known rigid walls. */
  room?: "cpml" | "rigid";
  /** Migrate with first-order image sources across the known walls as well (rigid room only). */
  imageSources?: boolean;
}

export function laptop(
  centre: number,
  spacing = 12,
  micInset = 2,
  row = DEVICE_ROW,
): Placement {
  const a = centre - Math.floor(spacing / 2);
  const b = a + spacing;
  return {
    speakers: [
      [row, a],
      [row, b],
    ],
    mics: [
      [row, a + micInset],
      [row, b - micInset],
    ],
  };
}

/** Placement centres: K = 2 and 4 slide the device by its speaker spacing (so neighbouring placements share a speaker position). */
export const PLACEMENT_CENTRES: Record<number, number[]> = {
  1: [45],
  2: [39, 51],
  4: [27, 39, 51, 63],
};

/** Code length: 2 x the listening window, the shortest that separates well (joint LS needs T_code + L >= 2 L_u). */
export const DEFAULT_CODE_STEPS = 1244;

/** The loop's 8-element bar (scenarios.demoScenario, n = 100) as one placement. */
export function barPlacement(): Placement {
  const el = Array.from({ length: 8 }, (_, k) => [DEVICE_ROW, 31 + 4 * k]);
  return { speakers: el, mics: el.map((p) => [...p]) };
}

export function scheme(name: string, codeSteps = DEFAULT_CODE_STEPS): Scheme {
  const narrow = [laptop(45)];
  switch (name) {
    case "seq":
      return { name, kind: "seq", placements: narrow };
    case "wide_seq":
      return { name, kind: "seq", placements: [laptop(45, 28)] };
    case "sum":
      return { name, kind: "sum", placements: narrow };
    case "band":
      return { name, kind: "band", placements: narrow };
    case "code":
      return { name, kind: "code", placements: narrow, codeSteps };
    case "code622":
      return { name, kind: "code", placements: narrow, codeSteps: 622 };
    case "code_mf":
      return {
        name,
        kind: "code",
        placements: narrow,
        codeSteps,
        separation: "mf",
      };
    case "chirp":
      return {
        name,
        kind: "code",
        placements: narrow,
        codeSteps,
        code: "chirp",
      };
    case "chirp_mf":
      return {
        name,
        kind: "code",
        placements: narrow,
        codeSteps,
        code: "chirp",
        separation: "mf",
      };
    case "beams":
      return { name, kind: "beams", placements: narrow };
    case "seq_k2":
      return {
        name,
        kind: "seq",
        placements: PLACEMENT_CENTRES[2].map((c) => laptop(c)),
      };
    case "seq_k4":
      return {
        name,
        kind: "seq",
        placements: PLACEMENT_CENTRES[4].map((c) => laptop(c)),
      };
    case "code_k4":
      return {
        name,
        kind: "code",
        placements: PLACEMENT_CENTRES[4].map((c) => laptop(c)),
        codeSteps,
      };
    case "seq_rigid":
      return { name, kind: "seq", placements: narrow, room: "rigid" };
    case "seq_rigid_img":
      return {
        name,
        kind: "seq",
        placements: narrow,
        room: "rigid",
        imageSources: true,
      };
    case "bar8":
      return { name, kind: "seq", placements: [barPlacement()] };
    default:
      throw new Error(`unknown two-speaker scheme ${name}`);
  }
}

/** Every element position (speakers and mics of all placements, deduplicated): the near field and centre of the features. */
export function schemeElements(s: Scheme): number[][] {
  const seen = new Set<string>();
  const out: number[][] = [];
  for (const p of s.placements)
    for (const e of [...p.speakers, ...p.mics]) {
      const k = e.join(",");
      if (!seen.has(k)) {
        seen.add(k);
        out.push(e);
      }
    }
  return out;
}

// ---------------------------------------------------------------- emissions

/** Ricker samples at the simulation rate: w[k] = ricker(k dt), as the engine evaluates a 'ricker' driver. */
export function rickerSamples(
  len: number,
  f0 = F0,
  delay = RICKER_DELAY,
): Float64Array {
  const w = new Float64Array(len);
  for (let k = 0; k < len; k++) {
    const a = Math.PI * f0 * (k * DT - delay);
    w[k] = (1 - 2 * a * a) * Math.exp(-a * a);
  }
  return w;
}

/** Raised-cosine step: 0 below a, 1 above b. */
export function rcStep(f: number, a: number, b: number): number {
  if (f <= a) return 0;
  if (f >= b) return 1;
  return 0.5 - 0.5 * Math.cos((Math.PI * (f - a)) / (b - a));
}

/** Band edges (cycles per unit time): low band below LOW_EDGE, high band above HIGH_EDGE, a guard gap between. */
export const BAND = {
  lowEdge: [0.065, 0.075],
  highEdge: [0.085, 0.095],
  split: [0.077, 0.083],
};
/** The band pulses are centred later than the Ricker so their filter tails fit after t = 0. */
export const BAND_DELAY = 120;
export const BAND_LEN = 240;

/** Low- and high-band parts of a Ricker centred at BAND_DELAY steps (zero-phase split with a guard gap, Hann-tapered). */
export function bandPulses(): { low: Float64Array; high: Float64Array } {
  const N = 1024;
  const r = rickerSamples(N, F0, BAND_DELAY * DT);
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  re.set(r);
  const R = fftCopy(re, im);
  const lowS = { re: new Float64Array(N), im: new Float64Array(N) };
  const highS = { re: new Float64Array(N), im: new Float64Array(N) };
  for (let k = 0; k < N; k++) {
    const f = Math.min(k, N - k) / (N * DT);
    const lo = 1 - rcStep(f, BAND.lowEdge[0], BAND.lowEdge[1]);
    const hi = rcStep(f, BAND.highEdge[0], BAND.highEdge[1]);
    lowS.re[k] = R.re[k] * lo;
    lowS.im[k] = R.im[k] * lo;
    highS.re[k] = R.re[k] * hi;
    highS.im[k] = R.im[k] * hi;
  }
  fft(lowS.re, lowS.im, true);
  fft(highS.re, highS.im, true);
  const low = new Float64Array(BAND_LEN);
  const high = new Float64Array(BAND_LEN);
  for (let k = 0; k < BAND_LEN; k++) {
    // Hann taper over the pulse support (the tails are below 1e-3 of the peak there).
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (BAND_LEN - 1));
    low[k] = lowS.re[k] * w;
    high[k] = highS.re[k] * w;
  }
  return { low, high };
}

/** Code band (cycles per unit time): covers the Ricker's band to about -30 dB. */
export const CODE_BAND: [number, number] = [0.01, 0.22];

/** Linear chirp over CODE_BAND, `len` steps, 10 % raised-cosine tapers, unit amplitude; `down` sweeps high to low. */
export function chirpCode(len: number, down = false): Float64Array {
  const [fa, fb] = down ? [CODE_BAND[1], CODE_BAND[0]] : CODE_BAND;
  const T = (len - 1) * DT;
  const k = (fb - fa) / T;
  const edge = 0.1 * T;
  const out = new Float64Array(len);
  for (let i = 0; i < len; i++) {
    const t = i * DT;
    let env = 1;
    if (t < edge) env = 0.5 - 0.5 * Math.cos((Math.PI * t) / edge);
    else if (t > T - edge)
      env = 0.5 - 0.5 * Math.cos((Math.PI * (T - t)) / edge);
    out[i] = env * Math.sin(2 * Math.PI * (fa + 0.5 * k * t) * t);
  }
  return out;
}

/**
 * Pseudo-noise code with the Ricker's amplitude spectrum: seeded white
 * Gaussian noise, zero-phase filtered by |R(f)| (so the Ricker band is excited
 * as a Ricker ping excites it), 5 % raised-cosine tapers, peak amplitude 1.
 */
export function noiseCode(len: number, seed: number): Float64Array {
  const N = nextPow2(2 * len);
  const rnd = mulberry32(seed);
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let k = 0; k < len; k++)
    re[k] = Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
  fft(re, im);
  const R = fftCopy(rickerSamples(N), new Float64Array(N));
  for (let k = 0; k < N; k++) {
    const a = Math.hypot(R.re[k], R.im[k]);
    re[k] *= a;
    im[k] *= a;
  }
  fft(re, im, true);
  const out = new Float64Array(len);
  const edge = Math.max(1, Math.round(0.05 * len));
  let peak = 0;
  for (let k = 0; k < len; k++) {
    let env = 1;
    if (k < edge) env = 0.5 - 0.5 * Math.cos((Math.PI * k) / edge);
    else if (k > len - 1 - edge)
      env = 0.5 - 0.5 * Math.cos((Math.PI * (len - 1 - k)) / edge);
    out[k] = re[k] * env;
    peak = Math.max(peak, Math.abs(out[k]));
  }
  for (let k = 0; k < len; k++) out[k] /= peak;
  return out;
}

/** The two codes of a 'code' scheme: up/down chirps, or two independent Ricker-band noise codes. */
export function schemeCodes(s: Scheme): Float64Array[] {
  const L = s.codeSteps ?? DEFAULT_CODE_STEPS;
  return s.code === "chirp"
    ? [chirpCode(L), chirpCode(L, true)]
    : [noiseCode(L, 1001), noiseCode(L, 2002)];
}

/** Beam shots: per shot (gain, delay in steps) of speaker A and speaker B, all emitting the same Ricker. */
export const BEAM_DELAY = 12;
export const BEAM_SHOTS: { a: [number, number]; b: [number, number] }[] = [
  { a: [1, 0], b: [1, 0] }, // in phase (broadside)
  { a: [1, 0], b: [-1, 0] }, // anti-phase (difference beam)
  { a: [1, 0], b: [1, BEAM_DELAY] }, // steered one way
  { a: [1, BEAM_DELAY], b: [1, 0] }, // steered the other way
];

export interface ShotDriver {
  pos: number[];
  data: Float64Array; // injected sample per step (soft source, +=)
}

export interface Shot {
  drivers: ShotDriver[];
  steps: number;
  /** Key identifying the emission (same key and room = same recording). */
  key: string;
}

const delayed = (x: Float64Array, d: number, g: number, len: number) => {
  const out = new Float64Array(len);
  for (let k = 0; k + d < len && k < x.length; k++) out[k + d] = g * x[k];
  return out;
};

/** The emission schedule of one placement: the shots it plays, in order. */
export function placementShots(s: Scheme, p: Placement): Shot[] {
  const [A, B] = p.speakers;
  const pk = (pos: number[]) => pos.join(",");
  switch (s.kind) {
    case "seq": {
      const ric = rickerSamples(LISTEN);
      return p.speakers.map((sp) => ({
        drivers: [{ pos: sp, data: ric }],
        steps: LISTEN,
        key: `ricker@${pk(sp)}`,
      }));
    }
    case "sum": {
      const ric = rickerSamples(LISTEN);
      return [
        {
          drivers: [
            { pos: A, data: ric },
            { pos: B, data: ric },
          ],
          steps: LISTEN,
          key: `ricker@${pk(A)}+${pk(B)}`,
        },
      ];
    }
    case "band": {
      const { low, high } = bandPulses();
      const steps = LISTEN + Math.ceil(BAND_DELAY - RICKER_DELAY / DT);
      return [
        {
          drivers: [
            { pos: A, data: low },
            { pos: B, data: high },
          ],
          steps,
          key: `band@${pk(A)}+${pk(B)}`,
        },
      ];
    }
    case "code": {
      const L = s.codeSteps ?? DEFAULT_CODE_STEPS;
      const [ca, cb] = schemeCodes(s);
      return [
        {
          drivers: [
            { pos: A, data: ca },
            { pos: B, data: cb },
          ],
          steps: L + LISTEN,
          key: `code${s.code ?? "noise"}${L}@${pk(A)}+${pk(B)}`,
        },
      ];
    }
    case "beams": {
      const len = LISTEN + BEAM_DELAY;
      const ric = rickerSamples(len);
      return BEAM_SHOTS.map((b, j) => ({
        drivers: [
          { pos: A, data: delayed(ric, b.a[1], b.a[0], len) },
          { pos: B, data: delayed(ric, b.b[1], b.b[0], len) },
        ],
        steps: len,
        key: `beam${j}@${pk(A)}+${pk(B)}`,
      }));
    }
  }
}

/** Total measurement time of a scheme, in steps (all placements, all shots). */
export function schemeSteps(s: Scheme): number {
  return s.placements.reduce(
    (t, p) => t + placementShots(s, p).reduce((a, sh) => a + sh.steps, 0),
    0,
  );
}

// ---------------------------------------------------------------- FFT helper

function fftCopy(
  re: Float64Array,
  im: Float64Array,
): { re: Float64Array; im: Float64Array } {
  const r = Float64Array.from(re);
  const i = Float64Array.from(im);
  fft(r, i);
  return { re: r, im: i };
}
