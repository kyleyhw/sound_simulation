/**
 * Minimal CPU tensor ops for in-browser inference of the sensing CNN
 * (plan 10.3). Batch size 1, layout (C, H, W), float32. Semantics follow
 * PyTorch exactly: Conv2d (zero padding), ConvTranspose2d, MaxPool2d
 * (floor), adaptive_avg_pool2d (window [floor(i H/S), ceil((i+1) H/S))).
 */

export interface Tensor {
  c: number;
  h: number;
  w: number;
  d: Float32Array;
}

export const tensor = (c: number, h: number, w: number, d?: Float32Array): Tensor => ({ c, h, w, d: d ?? new Float32Array(c * h * w) });

export function relu(x: Tensor): Tensor {
  for (let i = 0; i < x.d.length; i++) if (x.d[i] < 0) x.d[i] = 0;
  return x;
}

/** Conv2d, stride 1, square kernel k, zero padding `pad`. W: (out, in, k, k). */
export function conv2d(x: Tensor, W: Float32Array, b: Float32Array, cout: number, k: number, pad: number): Tensor {
  const { c: cin, h, w } = x;
  const ho = h + 2 * pad - k + 1;
  const wo = w + 2 * pad - k + 1;
  const y = tensor(cout, ho, wo);
  const plane = ho * wo;
  for (let o = 0; o < cout; o++) {
    const out = y.d.subarray(o * plane, (o + 1) * plane);
    out.fill(b[o]);
    for (let i = 0; i < cin; i++) {
      const xin = x.d.subarray(i * h * w, (i + 1) * h * w);
      for (let ky = 0; ky < k; ky++)
        for (let kx = 0; kx < k; kx++) {
          const wv = W[((o * cin + i) * k + ky) * k + kx];
          if (wv === 0) continue;
          const y0 = Math.max(0, pad - ky);
          const y1 = Math.min(ho, h + pad - ky);
          const x0 = Math.max(0, pad - kx);
          const x1 = Math.min(wo, w + pad - kx);
          for (let yy = y0; yy < y1; yy++) {
            const src = (yy + ky - pad) * w - pad + kx;
            const dst = yy * wo;
            for (let xx = x0; xx < x1; xx++) out[dst + xx] += wv * xin[src + xx];
          }
        }
    }
  }
  return y;
}

/** ConvTranspose2d, kernel 4, stride 2, padding 1 (doubles H and W). W: (in, out, 4, 4). */
export function convTranspose4s2(x: Tensor, W: Float32Array, b: Float32Array, cout: number): Tensor {
  const { c: cin, h, w } = x;
  const ho = 2 * h;
  const wo = 2 * w;
  const y = tensor(cout, ho, wo);
  const plane = ho * wo;
  for (let o = 0; o < cout; o++) y.d.subarray(o * plane, (o + 1) * plane).fill(b[o]);
  for (let i = 0; i < cin; i++) {
    const xin = x.d.subarray(i * h * w, (i + 1) * h * w);
    for (let o = 0; o < cout; o++) {
      const out = y.d.subarray(o * plane, (o + 1) * plane);
      for (let ky = 0; ky < 4; ky++)
        for (let kx = 0; kx < 4; kx++) {
          const wv = W[((i * cout + o) * 4 + ky) * 4 + kx];
          if (wv === 0) continue;
          for (let iy = 0; iy < h; iy++) {
            const yy = 2 * iy - 1 + ky;
            if (yy < 0 || yy >= ho) continue;
            for (let ix = 0; ix < w; ix++) {
              const xx = 2 * ix - 1 + kx;
              if (xx < 0 || xx >= wo) continue;
              out[yy * wo + xx] += wv * xin[iy * w + ix];
            }
          }
        }
    }
  }
  return y;
}

export function maxPool2(x: Tensor): Tensor {
  const ho = Math.floor(x.h / 2);
  const wo = Math.floor(x.w / 2);
  const y = tensor(x.c, ho, wo);
  for (let c = 0; c < x.c; c++)
    for (let i = 0; i < ho; i++)
      for (let j = 0; j < wo; j++) {
        const b = c * x.h * x.w;
        const a = b + 2 * i * x.w + 2 * j;
        y.d[(c * ho + i) * wo + j] = Math.max(x.d[a], x.d[a + 1], x.d[a + x.w], x.d[a + x.w + 1]);
      }
  return y;
}

export function adaptiveAvgPool(x: Tensor, S: number): Tensor {
  const y = tensor(x.c, S, S);
  for (let c = 0; c < x.c; c++)
    for (let i = 0; i < S; i++) {
      const r0 = Math.floor((i * x.h) / S);
      const r1 = Math.ceil(((i + 1) * x.h) / S);
      for (let j = 0; j < S; j++) {
        const c0 = Math.floor((j * x.w) / S);
        const c1 = Math.ceil(((j + 1) * x.w) / S);
        let acc = 0;
        for (let r = r0; r < r1; r++) for (let q = c0; q < c1; q++) acc += x.d[(c * x.h + r) * x.w + q];
        y.d[(c * S + i) * S + j] = acc / ((r1 - r0) * (c1 - c0));
      }
    }
  return y;
}

export function concat(a: Tensor, b: Tensor): Tensor {
  if (a.h !== b.h || a.w !== b.w) throw new Error('concat: spatial mismatch');
  const y = tensor(a.c + b.c, a.h, a.w);
  y.d.set(a.d, 0);
  y.d.set(b.d, a.d.length);
  return y;
}

/**
 * torchaudio.transforms.Spectrogram(n_fft, hop, power=None) defaults:
 * win_length = n_fft, the given (periodic Hann) window, center = True with
 * reflect padding, one-sided. Returns re/im planes of shape (F, frames).
 */
export function stft(x: Float32Array, nfft: number, hop: number, window: Float32Array): { re: Float32Array; im: Float32Array; F: number; T: number } {
  const pad = nfft >> 1;
  const n = x.length;
  const at = (j: number) => {
    let q = j - pad;
    if (q < 0) q = -q;
    if (q >= n) q = 2 * (n - 1) - q;
    return x[q];
  };
  const frames = 1 + Math.floor(n / hop);
  const F = (nfft >> 1) + 1;
  const re = new Float32Array(F * frames);
  const im = new Float32Array(F * frames);
  const cosT = new Float64Array(nfft);
  const sinT = new Float64Array(nfft);
  for (let k = 0; k < nfft; k++) {
    cosT[k] = Math.cos((2 * Math.PI * k) / nfft);
    sinT[k] = Math.sin((2 * Math.PI * k) / nfft);
  }
  const seg = new Float64Array(nfft);
  for (let t = 0; t < frames; t++) {
    for (let k = 0; k < nfft; k++) seg[k] = at(t * hop + k) * window[k];
    for (let f = 0; f < F; f++) {
      let sr = 0;
      let si = 0;
      for (let k = 0, idx = 0; k < nfft; k++, idx = (idx + f) % nfft) {
        sr += seg[k] * cosT[idx];
        si -= seg[k] * sinT[idx];
      }
      re[f * frames + t] = sr;
      im[f * frames + t] = si;
    }
  }
  return { re, im, F, T: frames };
}
