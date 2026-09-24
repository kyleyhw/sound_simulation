/** Minimal DSP: radix-2 FFT, magnitude spectrum, spectrogram, resampling. */

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/** In-place iterative radix-2 complex FFT (re, im have power-of-two length). */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k];
        const ai = im[i + k];
        const br = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const bi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ar + br;
        im[i + k] = ai + bi;
        re[i + k + len / 2] = ar - br;
        im[i + k + len / 2] = ai - bi;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/** One-sided magnitude spectrum of a real signal (Hann window, zero-padded). */
export function magnitudeSpectrum(x: ArrayLike<number>, nfft?: number): Float64Array {
  const n = nfft ?? nextPow2(x.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const m = Math.min(x.length, n);
  for (let i = 0; i < m; i++) re[i] = x[i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / Math.max(1, m - 1)));
  fft(re, im);
  const out = new Float64Array(n / 2 + 1);
  for (let k = 0; k <= n / 2; k++) out[k] = Math.hypot(re[k], im[k]);
  return out;
}

/** Spectrogram in dB: frames x bins, Hann window, hop = nfft/4. */
export function spectrogram(x: ArrayLike<number>, nfft = 256, hop = 64): { frames: Float32Array[]; bins: number } {
  const bins = nfft / 2 + 1;
  const frames: Float32Array[] = [];
  const win = new Float64Array(nfft);
  for (let i = 0; i < nfft; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (nfft - 1));
  for (let start = 0; start + nfft <= x.length; start += hop) {
    const re = new Float64Array(nfft);
    const im = new Float64Array(nfft);
    for (let i = 0; i < nfft; i++) re[i] = x[start + i] * win[i];
    fft(re, im);
    const f = new Float32Array(bins);
    for (let k = 0; k < bins; k++) f[k] = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-12);
    frames.push(f);
  }
  return { frames, bins };
}

/** Linear-interpolation resample to `outLen` samples. */
export function resampleLinear(x: ArrayLike<number>, outLen: number): Float32Array {
  const out = new Float32Array(outLen);
  if (x.length === 0) return out;
  const scale = (x.length - 1) / Math.max(1, outLen - 1);
  for (let i = 0; i < outLen; i++) {
    const t = i * scale;
    const k = Math.floor(t);
    const a = t - k;
    out[i] = k + 1 < x.length ? (1 - a) * x[k] + a * x[k + 1] : x[x.length - 1];
  }
  return out;
}

/** Full cross-correlation via FFT: r[lag] for lag in [0, n) (circular-safe padding). */
export function xcorr(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const n = nextPow2(a.length + b.length);
  const ar = new Float64Array(n);
  const ai = new Float64Array(n);
  const br = new Float64Array(n);
  const bi = new Float64Array(n);
  for (let i = 0; i < a.length; i++) ar[i] = a[i];
  for (let i = 0; i < b.length; i++) br[i] = b[i];
  fft(ar, ai);
  fft(br, bi);
  // a * conj(b)
  for (let k = 0; k < n; k++) {
    const r = ar[k] * br[k] + ai[k] * bi[k];
    const im = ai[k] * br[k] - ar[k] * bi[k];
    ar[k] = r;
    ai[k] = im;
  }
  fft(ar, ai, true);
  return ar;
}

/** Peak absolute value. */
export function peakAbs(x: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < x.length; i++) {
    const v = Math.abs(x[i]);
    if (v > m) m = v;
  }
  return m;
}
