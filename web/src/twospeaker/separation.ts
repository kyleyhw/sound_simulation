/**
 * Separating simultaneous emissions of the two-speaker device (see device.ts).
 *
 * The room is linear and time-invariant in discrete time: a mic records
 *   y[k] = sum_s (c_s * g_s)[k],
 * the sum over speakers of the emitted samples c_s convolved with the
 * speaker-to-mic impulse response g_s. Each separator returns, per speaker,
 * the trace the mic would have recorded had that speaker pinged alone (the
 * "Ricker-equivalent" trace r_s = ricker * g_s), or its band-limited part:
 *
 *   bandSplit    disjoint emission bands: zero-phase low/high-pass split
 *   lsSeparate   broadband codes: joint least squares for g_A, g_B of length
 *                Lu (CGLS with FFT convolutions, Tikhonov lambda), exact when
 *                the recording is long enough (T_code + L >= 2 L in band)
 *   mfSeparate   broadband codes, naive: per-speaker inverse filter that
 *                treats the other speaker as noise (its crosstalk remains)
 *   beamSeparate same-pulse shots with known per-speaker gains and delays:
 *                per-frequency least squares over the shots
 */

import { fft, nextPow2 } from "../lib/dsp";
import { BAND, DT, LISTEN, rcStep, rickerSamples } from "./device";

interface Spec {
  re: Float64Array;
  im: Float64Array;
}

function spectrum(x: ArrayLike<number>, N: number): Spec {
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let k = 0; k < Math.min(x.length, N); k++) re[k] = x[k];
  fft(re, im);
  return { re, im };
}

function inverse(s: Spec, len: number): Float64Array {
  fft(s.re, s.im, true);
  return s.re.slice(0, len);
}

/** Frequency (cycles per unit time) of FFT bin k of N (two-sided, absolute). */
const binFreq = (k: number, N: number) => Math.min(k, N - k) / (N * DT);

/** Zero-phase split of a recording into its low (< 0.08) and high (> 0.08) bands (raised-cosine crossover inside the guard gap). */
export function bandSplit(y: ArrayLike<number>): {
  low: Float32Array;
  high: Float32Array;
} {
  const N = nextPow2(2 * y.length);
  const Y = spectrum(y, N);
  const L: Spec = { re: new Float64Array(N), im: new Float64Array(N) };
  const H: Spec = { re: new Float64Array(N), im: new Float64Array(N) };
  for (let k = 0; k < N; k++) {
    const h = rcStep(binFreq(k, N), BAND.split[0], BAND.split[1]);
    L.re[k] = Y.re[k] * (1 - h);
    L.im[k] = Y.im[k] * (1 - h);
    H.re[k] = Y.re[k] * h;
    H.im[k] = Y.im[k] * h;
  }
  return {
    low: Float32Array.from(inverse(L, y.length)),
    high: Float32Array.from(inverse(H, y.length)),
  };
}

/** Low- or high-band part of a trace (the ideal separated trace, for scoring). */
export function bandPart(y: ArrayLike<number>, high: boolean): Float32Array {
  const s = bandSplit(y);
  return high ? s.high : s.low;
}

/** Causal convolution of x with the loop's Ricker samples, truncated to `len` (a unit-sample response -> Ricker-equivalent trace). */
export function toRicker(x: Float64Array, len = LISTEN): Float32Array {
  const N = nextPow2(len + x.length);
  const R = spectrum(rickerSamples(len), N);
  const X = spectrum(x, N);
  const P: Spec = { re: new Float64Array(N), im: new Float64Array(N) };
  for (let k = 0; k < N; k++) {
    P.re[k] = R.re[k] * X.re[k] - R.im[k] * X.im[k];
    P.im[k] = R.re[k] * X.im[k] + R.im[k] * X.re[k];
  }
  return Float32Array.from(inverse(P, len));
}

/**
 * Joint least-squares deconvolution of one recording y = sum_s c_s * g_s:
 * min sum_k (y - sum_s c_s * u_s)^2 + lambda sum_s |u_s|^2 over u_s of
 * length Lu, by CGLS (conjugate gradients on the normal equations) with FFT
 * convolutions. lambda = lambdaRel x max_f sum_s |C_s(f)|^2.
 */
export function lsSeparate(
  y: ArrayLike<number>,
  codes: Float64Array[],
  Lu: number,
  opts: { iters?: number; lambdaRel?: number } = {},
): { u: Float64Array[]; relResidual: number } {
  const S = codes.length;
  const Tc = Math.max(...codes.map((c) => c.length));
  const N = nextPow2(Math.max(y.length, Tc + Lu));
  const C = codes.map((c) => spectrum(c, N));
  let cmax = 0;
  for (let k = 0; k < N; k++) {
    let a = 0;
    for (const c of C) a += c.re[k] ** 2 + c.im[k] ** 2;
    cmax = Math.max(cmax, a);
  }
  const lambda = (opts.lambdaRel ?? 1e-4) * cmax;
  const T = y.length;
  // A: (u_s) -> sum_s c_s * u_s, truncated to T.
  const A = (u: Float64Array[]): Float64Array => {
    const acc: Spec = { re: new Float64Array(N), im: new Float64Array(N) };
    for (let s = 0; s < S; s++) {
      const U = spectrum(u[s], N);
      const c = C[s];
      for (let k = 0; k < N; k++) {
        acc.re[k] += c.re[k] * U.re[k] - c.im[k] * U.im[k];
        acc.im[k] += c.re[k] * U.im[k] + c.im[k] * U.re[k];
      }
    }
    return inverse(acc, T);
  };
  // A^T: r -> (corr(c_s, r))[0..Lu).
  const At = (r: Float64Array): Float64Array[] => {
    const R = spectrum(r, N);
    return C.map((c) => {
      const P: Spec = { re: new Float64Array(N), im: new Float64Array(N) };
      for (let k = 0; k < N; k++) {
        P.re[k] = c.re[k] * R.re[k] + c.im[k] * R.im[k];
        P.im[k] = c.re[k] * R.im[k] - c.im[k] * R.re[k];
      }
      return inverse(P, Lu);
    });
  };
  const dot = (a: Float64Array[], b: Float64Array[]) =>
    a.reduce((t, x, s) => t + x.reduce((u, v, k) => u + v * b[s][k], 0), 0);
  const x = codes.map(() => new Float64Array(Lu));
  const r = Float64Array.from(y);
  const y2 = r.reduce((t, v) => t + v * v, 0);
  let sv = At(r);
  const p = sv.map((a) => Float64Array.from(a));
  let gamma = dot(sv, sv);
  const iters = opts.iters ?? 200;
  for (let it = 0; it < iters && gamma > 0; it++) {
    const q = A(p);
    const alpha =
      gamma / (q.reduce((t, v) => t + v * v, 0) + lambda * dot(p, p));
    for (let s = 0; s < S; s++)
      for (let k = 0; k < Lu; k++) x[s][k] += alpha * p[s][k];
    for (let k = 0; k < T; k++) r[k] -= alpha * q[k];
    sv = At(r);
    for (let s = 0; s < S; s++)
      for (let k = 0; k < Lu; k++) sv[s][k] -= lambda * x[s][k];
    const g2 = dot(sv, sv);
    const beta = g2 / gamma;
    gamma = g2;
    for (let s = 0; s < S; s++)
      for (let k = 0; k < Lu; k++) p[s][k] = sv[s][k] + beta * p[s][k];
  }
  const res = r.reduce((t, v) => t + v * v, 0);
  return { u: x, relResidual: Math.sqrt(res / Math.max(y2, 1e-300)) };
}

/**
 * Naive per-speaker separation of a coded recording: the inverse filter of
 * speaker s's code (regularised), mapped to the Ricker band. The other
 * speaker's contribution stays in as a dispersed crosstalk term.
 */
export function mfSeparate(
  y: ArrayLike<number>,
  codes: Float64Array[],
  len = LISTEN,
  epsRel = 1e-3,
): Float32Array[] {
  const N = nextPow2(y.length + len);
  const Y = spectrum(y, N);
  const R = spectrum(rickerSamples(len), N);
  return codes.map((c) => {
    const Cs = spectrum(c, N);
    let cmax = 0;
    for (let k = 0; k < N; k++)
      cmax = Math.max(cmax, Cs.re[k] ** 2 + Cs.im[k] ** 2);
    const eps = epsRel * cmax;
    const P: Spec = { re: new Float64Array(N), im: new Float64Array(N) };
    for (let k = 0; k < N; k++) {
      // conj(C) Y / (|C|^2 + eps), times the Ricker spectrum
      const d = Cs.re[k] ** 2 + Cs.im[k] ** 2 + eps;
      const xr = (Cs.re[k] * Y.re[k] + Cs.im[k] * Y.im[k]) / d;
      const xi = (Cs.re[k] * Y.im[k] - Cs.im[k] * Y.re[k]) / d;
      P.re[k] = R.re[k] * xr - R.im[k] * xi;
      P.im[k] = R.re[k] * xi + R.im[k] * xr;
    }
    return Float32Array.from(inverse(P, len));
  });
}

/**
 * Same-pulse shots j with speaker A at gain/delay a_j and B at b_j:
 * Y_j(f) = a_j e^{-i w dA_j} R_A(f) + b_j e^{-i w dB_j} R_B(f). Solves the
 * 2 x 2 normal equations per frequency (lambda-regularised) for R_A, R_B.
 */
export function beamSeparate(
  ys: ArrayLike<number>[],
  shots: { a: [number, number]; b: [number, number] }[],
  len = LISTEN,
  lambdaRel = 1e-9,
): Float32Array[] {
  const N = nextPow2(2 * Math.max(...ys.map((y) => y.length)));
  const Y = ys.map((y) => spectrum(y, N));
  const RA: Spec = { re: new Float64Array(N), im: new Float64Array(N) };
  const RB: Spec = { re: new Float64Array(N), im: new Float64Array(N) };
  const lam = lambdaRel * shots.length;
  for (let k = 0; k < N; k++) {
    const w = (-2 * Math.PI * k) / N;
    // Hermitian 2 x 2: [[m11, m12], [conj m12, m22]] and right-hand side (v1, v2).
    let m11 = lam;
    let m22 = lam;
    let m12r = 0;
    let m12i = 0;
    let v1r = 0;
    let v1i = 0;
    let v2r = 0;
    let v2i = 0;
    shots.forEach((sh, j) => {
      const ar = sh.a[0] * Math.cos(w * sh.a[1]);
      const ai = sh.a[0] * Math.sin(w * sh.a[1]);
      const br = sh.b[0] * Math.cos(w * sh.b[1]);
      const bi = sh.b[0] * Math.sin(w * sh.b[1]);
      m11 += ar * ar + ai * ai;
      m22 += br * br + bi * bi;
      // conj(a) b
      m12r += ar * br + ai * bi;
      m12i += ar * bi - ai * br;
      const yr = Y[j].re[k];
      const yi = Y[j].im[k];
      v1r += ar * yr + ai * yi;
      v1i += ar * yi - ai * yr;
      v2r += br * yr + bi * yi;
      v2i += br * yi - bi * yr;
    });
    const det = m11 * m22 - (m12r * m12r + m12i * m12i);
    // x1 = (m22 v1 - m12 v2) / det, x2 = (m11 v2 - conj(m12) v1) / det
    RA.re[k] = (m22 * v1r - (m12r * v2r - m12i * v2i)) / det;
    RA.im[k] = (m22 * v1i - (m12r * v2i + m12i * v2r)) / det;
    RB.re[k] = (m11 * v2r - (m12r * v1r + m12i * v1i)) / det;
    RB.im[k] = (m11 * v2i - (m12r * v1i - m12i * v1r)) / det;
  }
  return [
    Float32Array.from(inverse(RA, len)),
    Float32Array.from(inverse(RB, len)),
  ];
}

/** Signal-to-distortion ratio (dB) of an estimate against a reference trace (over the reference's length). */
export function sdrDb(est: ArrayLike<number>, ref: ArrayLike<number>): number {
  let e = 0;
  let s = 0;
  for (let k = 0; k < ref.length; k++) {
    s += ref[k] ** 2;
    e += ((est[k] ?? 0) - ref[k]) ** 2;
  }
  return 10 * Math.log10(Math.max(s, 1e-300) / Math.max(e, 1e-300));
}
