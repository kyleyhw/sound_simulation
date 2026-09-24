/**
 * Crosstalk cancellation (CTC) for two loudspeakers: "virtual headphones"
 * (plan 7.3 and 9.10).
 *
 * Plant: H(f) is the 2x2 matrix from the speakers (columns) to the ears
 * (rows). The free-field point-source model used here is
 *   H_ej(f) = exp(-i k r_ej) / r_ej,  k = 2 pi f / c,
 * with the ears at +-a from the head centre (a = 8.75 cm), ignoring head
 * shadowing, which only strengthens natural separation. The Python
 * control module designs the same filters from FDTD-measured responses.
 *
 * Filter: a regularised inverse with a modelling delay (Kirkeby et al. 1998)
 *   C(f) = H^H (H H^H + beta I)^{-1} exp(-i 2 pi f tau).
 * With beta -> 0, H C = I e^{-i w tau}: each ear hears only its own
 * programme channel. beta limits the effort (the filter gain) at the
 * ill-conditioned frequencies, where H is nearly singular. For a laptop
 * these are the low frequencies and the "ringing" frequencies where the
 * path difference is a whole wavelength.
 */

import { fft } from '../lib/dsp';

export interface CtcGeometry {
  /** Speaker separation, metres (laptop: 0.25-0.35). */
  speakerSpan: number;
  /** Head centre relative to the midpoint between the speakers: x lateral, y forward (m). */
  head: { x: number; y: number };
  earRadius?: number;
  c?: number;
}

type C = [number, number]; // complex re, im

const cmul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cadd = (a: C, b: C): C => [a[0] + b[0], a[1] + b[1]];
const csub = (a: C, b: C): C => [a[0] - b[0], a[1] - b[1]];
const conj = (a: C): C => [a[0], -a[1]];
const cdiv = (a: C, b: C): C => {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
};

/** 2x2 plant at frequency f: [[H_L,spkL, H_L,spkR],[H_R,spkL, H_R,spkR]]. */
export function plant(g: CtcGeometry, f: number): C[][] {
  const c = g.c ?? 343;
  const a = g.earRadius ?? 0.0875;
  const k = (2 * Math.PI * f) / c;
  const spk = [-g.speakerSpan / 2, g.speakerSpan / 2];
  const ears = [g.head.x - a, g.head.x + a];
  return ears.map((ex) =>
    spk.map((sx) => {
      const r = Math.hypot(ex - sx, g.head.y);
      return [Math.cos(-k * r) / r, Math.sin(-k * r) / r] as C;
    }),
  );
}

/** Regularised inverse C = H^H (H H^H + beta I)^{-1} at one frequency. */
export function inverse2x2(H: C[][], beta: number): C[][] {
  const Hh = [
    [conj(H[0][0]), conj(H[1][0])],
    [conj(H[0][1]), conj(H[1][1])],
  ];
  const A = [
    [cadd(cadd(cmul(H[0][0], Hh[0][0]), cmul(H[0][1], Hh[1][0])), [beta, 0]), cadd(cmul(H[0][0], Hh[0][1]), cmul(H[0][1], Hh[1][1]))],
    [cadd(cmul(H[1][0], Hh[0][0]), cmul(H[1][1], Hh[1][0])), cadd(cadd(cmul(H[1][0], Hh[0][1]), cmul(H[1][1], Hh[1][1])), [beta, 0])],
  ];
  const det = csub(cmul(A[0][0], A[1][1]), cmul(A[0][1], A[1][0]));
  const inv = [
    [cdiv(A[1][1], det), cdiv([-A[0][1][0], -A[0][1][1]], det)],
    [cdiv([-A[1][0][0], -A[1][0][1]], det), cdiv(A[0][0], det)],
  ];
  return [
    [cadd(cmul(Hh[0][0], inv[0][0]), cmul(Hh[0][1], inv[1][0])), cadd(cmul(Hh[0][0], inv[0][1]), cmul(Hh[0][1], inv[1][1]))],
    [cadd(cmul(Hh[1][0], inv[0][0]), cmul(Hh[1][1], inv[1][0])), cadd(cmul(Hh[1][0], inv[0][1]), cmul(Hh[1][1], inv[1][1]))],
  ];
}

export interface CtcFilters {
  /** taps[speaker][programme] impulse responses. */
  taps: Float32Array[][];
  sampleRate: number;
  length: number;
}

/**
 * FIR crosstalk-cancellation filters. The modelling delay is half the
 * filter length. Regularisation beta is relative to the mean |H|^2.
 */
export function designCtc(g: CtcGeometry, sampleRate: number, length = 1024, beta = 0.005, band: [number, number] = [150, 7000]): CtcFilters {
  const n = length;
  const tau = n / 2 / sampleRate;
  const re = [0, 1].map(() => [0, 1].map(() => new Float64Array(n)));
  const im = [0, 1].map(() => [0, 1].map(() => new Float64Array(n)));
  const r0 = Math.hypot(g.speakerSpan / 2, g.head.y);
  const scale = r0 * r0; // normalise |H|^2 ~ 1
  for (let k = 0; k <= n / 2; k++) {
    const f = (k * sampleRate) / n;
    const H = plant(g, Math.max(f, 1)).map((row) => row.map((v) => [v[0] * r0, v[1] * r0] as C));
    let Cm = inverse2x2(H, beta);
    // Outside the band, fall back to plain stereo (identity) with the same delay.
    if (f < band[0] || f > band[1]) {
      Cm = [
        [
          [1, 0],
          [0, 0],
        ],
        [
          [0, 0],
          [1, 0],
        ],
      ];
    }
    const ph: C = [Math.cos(-2 * Math.PI * f * tau), Math.sin(-2 * Math.PI * f * tau)];
    for (let s = 0; s < 2; s++)
      for (let p = 0; p < 2; p++) {
        const v = cmul(Cm[s][p], ph);
        re[s][p][k] = v[0];
        im[s][p][k] = v[1];
        if (k > 0 && k < n / 2) {
          re[s][p][n - k] = v[0];
          im[s][p][n - k] = -v[1];
        }
      }
  }
  void scale;
  const taps = [0, 1].map((s) =>
    [0, 1].map((p) => {
      const r = re[s][p];
      const i = im[s][p];
      fft(r, i, true);
      // Hann window to taper the truncation.
      const out = new Float32Array(n);
      for (let t = 0; t < n; t++) out[t] = r[t] * (0.5 - 0.5 * Math.cos((2 * Math.PI * t) / (n - 1)));
      return out;
    }),
  );
  return { taps, sampleRate, length: n };
}

/**
 * Channel separation (dB) delivered at the ears by filters designed for
 * geometry `design` when the head is actually at `actual`: 20 log10 of
 * |intended ear| / |other ear| for programme L, per frequency.
 */
export function separationDb(filters: CtcFilters, design: CtcGeometry, actual: CtcGeometry, freqs: number[]): number[] {
  const n = filters.length;
  const spec = filters.taps.map((row) =>
    row.map((t) => {
      const r = Float64Array.from(t);
      const i = new Float64Array(n);
      fft(r, i);
      return { r, i };
    }),
  );
  void design;
  return freqs.map((f) => {
    const k = Math.round((f * n) / filters.sampleRate);
    const H = plant(actual, f);
    const Cf = [0, 1].map((s) => [0, 1].map((p) => [spec[s][p].r[k], spec[s][p].i[k]] as C));
    // ear e, programme L (p = 0): sum_s H[e][s] C[s][0]
    const eL = cadd(cmul(H[0][0], Cf[0][0]), cmul(H[0][1], Cf[1][0]));
    const eR = cadd(cmul(H[1][0], Cf[0][0]), cmul(H[1][1], Cf[1][0]));
    return 20 * Math.log10(Math.hypot(...eL) / Math.max(Math.hypot(...eR), 1e-12));
  });
}

/** Natural (no-CTC) separation of plain stereo at the ears. */
export function stereoSeparationDb(g: CtcGeometry, freqs: number[]): number[] {
  return freqs.map((f) => {
    const H = plant(g, f);
    return 20 * Math.log10(Math.hypot(...H[0][0]) / Math.hypot(...H[1][0]));
  });
}
