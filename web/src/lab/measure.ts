/**
 * Room acoustic measurement DSP (plan Phase 9): exponential sine sweep,
 * impulse-response recovery, echo picking, reverberation metrics.
 *
 * Exponential sine sweep (ESS; Farina 2000):
 *   x(t) = sin( 2 pi f1 T / ln(f2/f1) * (exp(t ln(f2/f1) / T) - 1) ),  0 <= t < T.
 * Its inverse filter is the time-reversed sweep with a -6 dB/octave
 * amplitude envelope, exp(-t ln(f2/f1)/T) applied after reversal, so that
 * x * inv ~ band-limited impulse. Convolving the recording with the inverse
 * gives the linear impulse response at its correct delay; harmonic
 * distortion products land at negative times and can be discarded.
 */

import { fft, nextPow2 } from '../lib/dsp';

export interface SweepSpec {
  f1: number; // Hz
  f2: number; // Hz
  seconds: number;
  sampleRate: number;
  fadeMs?: number;
}

export function essSweep(s: SweepSpec): Float32Array {
  const n = Math.round(s.seconds * s.sampleRate);
  const x = new Float32Array(n);
  const L = Math.log(s.f2 / s.f1);
  const K = (2 * Math.PI * s.f1 * s.seconds) / L;
  const fade = Math.round(((s.fadeMs ?? 10) / 1000) * s.sampleRate);
  for (let i = 0; i < n; i++) {
    const t = i / s.sampleRate;
    let env = 1;
    if (i < fade) env = 0.5 - 0.5 * Math.cos((Math.PI * i) / fade);
    else if (i > n - fade) env = 0.5 - 0.5 * Math.cos((Math.PI * (n - i)) / fade);
    x[i] = env * Math.sin(K * (Math.exp((t * L) / s.seconds) - 1));
  }
  return x;
}

export function essInverse(s: SweepSpec, sweep: Float32Array): Float32Array {
  const n = sweep.length;
  const inv = new Float32Array(n);
  const L = Math.log(s.f2 / s.f1);
  for (let i = 0; i < n; i++) {
    const t = i / s.sampleRate;
    inv[i] = sweep[n - 1 - i] * Math.exp((-t * L) / s.seconds);
  }
  // Normalise so that sweep * inv has unit peak.
  const peak = convolvePeak(sweep, inv);
  for (let i = 0; i < n; i++) inv[i] /= peak;
  return inv;
}

function convolvePeak(a: Float32Array, b: Float32Array): number {
  const c = convolve(a, b);
  let m = 0;
  for (let i = 0; i < c.length; i++) m = Math.max(m, Math.abs(c[i]));
  return m || 1;
}

/** Linear convolution via FFT. */
export function convolve(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const n = nextPow2(a.length + b.length - 1);
  const ar = new Float64Array(n);
  const ai = new Float64Array(n);
  const br = new Float64Array(n);
  const bi = new Float64Array(n);
  for (let i = 0; i < a.length; i++) ar[i] = a[i];
  for (let i = 0; i < b.length; i++) br[i] = b[i];
  fft(ar, ai);
  fft(br, bi);
  for (let k = 0; k < n; k++) {
    const r = ar[k] * br[k] - ai[k] * bi[k];
    const im = ar[k] * bi[k] + ai[k] * br[k];
    ar[k] = r;
    ai[k] = im;
  }
  fft(ar, ai, true);
  return ar.subarray(0, a.length + b.length - 1);
}

/**
 * Impulse response from a recording of the sweep: recording * inverse,
 * returned from the sweep's own zero-lag onwards (index n-1 of the full
 * convolution), `length` samples long.
 */
export function deconvolve(recording: Float32Array, inverse: Float32Array, length: number): Float32Array {
  const full = convolve(recording, inverse);
  const start = inverse.length - 1;
  const out = new Float32Array(Math.min(length, full.length - start));
  for (let i = 0; i < out.length; i++) out[i] = full[start + i];
  return out;
}

export interface Echo {
  /** Sample index in the impulse response. */
  index: number;
  delayMs: number;
  /** Extra path relative to the direct sound, metres. */
  extraPath: number;
  /** Distance to a reflector for a co-located source and microphone: extraPath / 2. */
  distance: number;
  levelDb: number;
}

/**
 * Direct sound + echoes by peak picking on the envelope of the impulse
 * response. Delays are relative to the direct peak, so constant system
 * latency cancels.
 */
export function findEchoes(
  h: Float32Array,
  sampleRate: number,
  opts: { c?: number; thresholdDb?: number; minSeparationMs?: number; maxEchoes?: number; directIndex?: number } = {},
): { direct: number; echoes: Echo[] } {
  const c = opts.c ?? 343;
  const thr = opts.thresholdDb ?? -30;
  const sep = Math.max(1, Math.round(((opts.minSeparationMs ?? 0.6) / 1000) * sampleRate));
  const env = Float32Array.from(h, Math.abs);
  let direct = opts.directIndex ?? 0;
  if (opts.directIndex === undefined) for (let i = 0; i < env.length; i++) if (env[i] > env[direct]) direct = i;
  const ref = env[direct] || 1;
  const cand: number[] = [];
  for (let i = direct + sep; i < env.length - 1; i++) {
    if (env[i] >= env[i - 1] && env[i] > env[i + 1] && 20 * Math.log10(env[i] / ref) > thr) cand.push(i);
  }
  cand.sort((a, b) => env[b] - env[a]);
  const picked: number[] = [];
  for (const i of cand) {
    if (picked.every((j) => Math.abs(i - j) >= sep)) picked.push(i);
    if (picked.length >= (opts.maxEchoes ?? 8)) break;
  }
  picked.sort((a, b) => a - b);
  return {
    direct,
    echoes: picked.map((i) => {
      const dt = (i - direct) / sampleRate;
      return { index: i, delayMs: dt * 1000, extraPath: dt * c, distance: (dt * c) / 2, levelDb: 20 * Math.log10(env[i] / ref) };
    }),
  };
}

export interface DecayMetrics {
  edc: Float32Array; // energy decay curve in dB
  t20: number | null; // T60 extrapolated from -5..-25 dB
  t30: number | null; // T60 extrapolated from -5..-35 dB
  edt: number | null; // T60 extrapolated from 0..-10 dB
}

/** Schroeder backward integration and T20/T30/EDT line fits. */
export function decayMetrics(h: Float32Array, sampleRate: number, from = 0): DecayMetrics {
  const x = h.subarray(from);
  const e = new Float64Array(x.length);
  let acc = 0;
  for (let i = x.length - 1; i >= 0; i--) {
    acc += x[i] * x[i];
    e[i] = acc;
  }
  const edc = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) edc[i] = 10 * Math.log10(e[i] / (e[0] || 1) + 1e-30);
  const fit = (hi: number, lo: number): number | null => {
    let i0 = -1;
    let i1 = -1;
    for (let i = 0; i < edc.length; i++) {
      if (i0 < 0 && edc[i] <= hi) i0 = i;
      if (edc[i] <= lo) {
        i1 = i;
        break;
      }
    }
    if (i0 < 0 || i1 < 0 || i1 - i0 < 4) return null;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    const n = i1 - i0 + 1;
    for (let i = i0; i <= i1; i++) {
      const t = i / sampleRate;
      sx += t;
      sy += edc[i];
      sxx += t * t;
      sxy += t * edc[i];
    }
    const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    return slope < 0 ? -60 / slope : null;
  };
  return { edc, t20: fit(-5, -25), t30: fit(-5, -35), edt: fit(0, -10) };
}

/** Band-pass an impulse response by FFT masking (octave band around fc). */
export function octaveBand(h: Float32Array, sampleRate: number, fc: number): Float32Array {
  const n = nextPow2(h.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re.set(h);
  fft(re, im);
  const lo = fc / Math.SQRT2;
  const hi = fc * Math.SQRT2;
  for (let k = 0; k < n; k++) {
    const f = (Math.min(k, n - k) * sampleRate) / n;
    // raised-cosine skirts over a quarter octave
    let g = 0;
    if (f >= lo && f <= hi) g = 1;
    else if (f < lo && f > lo / 1.19) g = 0.5 - 0.5 * Math.cos((Math.PI * (f - lo / 1.19)) / (lo - lo / 1.19));
    else if (f > hi && f < hi * 1.19) g = 0.5 + 0.5 * Math.cos((Math.PI * (f - hi)) / (hi * 1.19 - hi));
    re[k] *= g;
    im[k] *= g;
  }
  fft(re, im, true);
  return Float32Array.from(re.subarray(0, h.length));
}

// ---------------------------------------------------------------------- //
// Room twin: diffuse-field estimates for a shoebox room (3D)
// ---------------------------------------------------------------------- //

export interface Shoebox {
  lx: number;
  ly: number;
  lz: number;
  /** Mean random-incidence absorption coefficient of the surfaces. */
  alpha: number;
}

export function sabine(r: Shoebox, c = 343): number {
  const V = r.lx * r.ly * r.lz;
  const S = 2 * (r.lx * r.ly + r.lx * r.lz + r.ly * r.lz);
  return (24 * Math.log(10) * V) / (c * S * r.alpha);
}

export function eyring(r: Shoebox, c = 343): number {
  const V = r.lx * r.ly * r.lz;
  const S = 2 * (r.lx * r.ly + r.lx * r.lz + r.ly * r.lz);
  return (24 * Math.log(10) * V) / (c * S * -Math.log(1 - r.alpha));
}

/** Absorption implied by a measured T60 under Eyring (inverse problem). */
export function alphaFromT60(r: Omit<Shoebox, 'alpha'>, t60: number, c = 343): number {
  const V = r.lx * r.ly * r.lz;
  const S = 2 * (r.lx * r.ly + r.lx * r.lz + r.ly * r.lz);
  return 1 - Math.exp(-(24 * Math.log(10) * V) / (c * S * t60));
}

/** Echo arrival times predicted by first-order image sources for a source/mic
 * pair at `pos` (co-located) in a shoebox: the six wall distances. */
export function shoeboxFirstEchoes(r: Omit<Shoebox, 'alpha'>, pos: [number, number, number]): number[] {
  if (shoeboxGeometryError(r, pos)) return [];
  const [x, y, z] = pos;
  return [x, r.lx - x, y, r.ly - y, z, r.lz - z].sort((a, b) => a - b);
}

/** Why a room/position pair is impossible (the laptop must sit strictly
 * inside the box), or null when it is valid. */
export function shoeboxGeometryError(r: Omit<Shoebox, 'alpha'>, pos: [number, number, number]): string | null {
  const dims = [r.lx, r.ly, r.lz];
  const names = ['length', 'width', 'height'];
  for (let a = 0; a < 3; a++) {
    if (!(Number.isFinite(dims[a]) && dims[a] > 0)) return `The room ${names[a]} must be a positive number.`;
    const p = pos[a];
    if (!(Number.isFinite(p) && p > 0 && p < dims[a]))
      return `Laptop ${'xyz'[a]} = ${Number.isFinite(p) ? p.toFixed(2) : p} m is outside the room (0 to ${dims[a].toFixed(2)} m along its ${names[a]}).`;
  }
  return null;
}

/**
 * Device calibration (plan 9.6). The direct path from the laptop's own
 * speaker to its own microphone, a few centimetres away, arrives long
 * before any room reflection, so a short window around it is the device
 * response (speaker x mic x converters) with the room excluded.
 *
 * `directPathWindow` cuts that window (Hann-tapered, `ms` long, starting
 * `preMs` before the peak). `equalize` then removes the device response
 * from any later impulse response of the same device with a Wiener-style
 * regularised inverse,
 *   G(f) = H(f) conj(D(f)) / (|D(f)|^2 + eps max|D|^2),
 * which flattens the colouration and sharpens the echo peaks without
 * dividing by the device's spectral nulls. The window's pre-peak offset is
 * added back so that the direct sound stays where it was.
 *
 * Round-trip latency (output buffer + input buffer + converters) is the
 * delay of the direct peak after the moment playback was scheduled, minus
 * the few-centimetre acoustic flight time.
 */
export interface DeviceCalibration {
  sampleRate: number;
  response: Float32Array;
  preSamples: number;
  latencyMs: number;
}

export function directPathWindow(h: Float32Array, direct: number, sampleRate: number, ms = 2, preMs = 0.3): { response: Float32Array; preSamples: number } {
  const pre = Math.round((preMs / 1000) * sampleRate);
  const n = Math.max(8, Math.round((ms / 1000) * sampleRate));
  const out = new Float32Array(n);
  const start = direct - pre;
  const fallFrom = Math.max(pre + 1, Math.round(0.7 * n));
  for (let i = 0; i < n; i++) {
    const j = start + i;
    // Half-Hann rise over the pre-peak samples, flat through the peak and
    // its ringing, half-Hann fall over the last 30 %.
    let w = 1;
    if (i < pre) w = 0.5 - 0.5 * Math.cos((Math.PI * (i + 0.5)) / pre);
    else if (i >= fallFrom) w = 0.5 + 0.5 * Math.cos((Math.PI * (i - fallFrom + 0.5)) / (n - fallFrom));
    out[i] = j >= 0 && j < h.length ? h[j] * w : 0;
  }
  return { response: out, preSamples: pre };
}

export function equalize(h: Float32Array, cal: { response: Float32Array; preSamples: number }, eps = 0.01): Float32Array {
  const n = nextPow2(h.length + cal.response.length);
  const hr = new Float64Array(n);
  const hi = new Float64Array(n);
  const dr = new Float64Array(n);
  const di = new Float64Array(n);
  hr.set(h);
  dr.set(cal.response);
  fft(hr, hi);
  fft(dr, di);
  let maxP = 0;
  for (let k = 0; k < n; k++) maxP = Math.max(maxP, dr[k] * dr[k] + di[k] * di[k]);
  const reg = eps * maxP;
  // Normalise so that the device response itself maps to a unit impulse.
  for (let k = 0; k < n; k++) {
    const p = dr[k] * dr[k] + di[k] * di[k] + reg;
    const re = (hr[k] * dr[k] + hi[k] * di[k]) / p;
    const im = (hi[k] * dr[k] - hr[k] * di[k]) / p;
    // Delay by preSamples to keep the direct sound in place.
    const ph = (-2 * Math.PI * k * cal.preSamples) / n;
    hr[k] = re * Math.cos(ph) - im * Math.sin(ph);
    hi[k] = re * Math.sin(ph) + im * Math.cos(ph);
  }
  fft(hr, hi, true);
  return Float32Array.from(hr.subarray(0, h.length));
}

export function roundTripLatencyMs(directIndex: number, preRollSamples: number, sampleRate: number, speakerMicM = 0.05, c = 343): number {
  return ((directIndex - preRollSamples) / sampleRate - speakerMicM / c) * 1000;
}
