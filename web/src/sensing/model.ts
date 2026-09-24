/**
 * SkipSensingCNN inference in the browser (plan 10.3): a line-by-line port
 * of learning/model.py for weights exported by scripts/export_web_model.py.
 */
import { adaptiveAvgPool, concat, conv2d, convTranspose4s2, maxPool2, relu, stft, type Tensor, tensor } from './nn';

export interface ModelManifest {
  format: string;
  model_type: 'skip';
  epoch?: number;
  checkpoint?: string;
  tensors: { name: string; shape: number[]; offset: number }[];
  protocol: {
    grid: number;
    duration: number;
    courant: number;
    mic_spacing: number;
    f_start: number;
    f_end: number;
    sample_rate: number;
    amplitude: number;
    target_size: number;
    protocol: string;
    sensor_n_fft: number;
    sensor_hop: number;
    source_n_fft: number;
    source_hop: number;
  };
  calibration: { temperature: number; bias: number; prior: number; threshold: number };
  prior_map: number[];
  /** No-audio baseline threshold on the prior map, chosen on training rooms. */
  prior_threshold: number;
}

export class SkipModel {
  private w = new Map<string, Float32Array>();
  readonly manifest: ModelManifest;
  private sourceCache: { key: Float32Array; u: Tensor } | null = null;

  constructor(manifest: ModelManifest, weights: ArrayBuffer) {
    this.manifest = manifest;
    const all = new Float32Array(weights);
    for (const t of manifest.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      this.w.set(t.name, all.subarray(t.offset, t.offset + n));
    }
  }

  static async load(base: string): Promise<SkipModel> {
    const [m, b] = await Promise.all([fetch(`${base}.json`).then((r) => r.json()), fetch(`${base}.bin`).then((r) => r.arrayBuffer())]);
    return new SkipModel(m as ModelManifest, b);
  }

  private p(name: string): Float32Array {
    const v = this.w.get(name);
    if (!v) throw new Error(`missing weight ${name}`);
    return v;
  }

  private conv(x: Tensor, name: string, k = 3, pad = 1): Tensor {
    const W = this.p(`${name}.weight`);
    const b = this.p(`${name}.bias`);
    return conv2d(x, W, b, b.length, k, pad);
  }

  private convT(x: Tensor, name: string): Tensor {
    const b = this.p(`${name}.bias`);
    return convTranspose4s2(x, this.p(`${name}.weight`), b, b.length);
  }

  /** StereoPhaseFrontEnd: [log1p|X1|^2, log1p|X2|^2, cos phi, sin phi], phi = arg(X1 X2*). */
  private front(m1: Float32Array, m2: Float32Array): Tensor {
    const { sensor_n_fft: n, sensor_hop: hop } = this.manifest.protocol;
    const win = this.p('front.spec.window');
    const a = stft(m1, n, hop, win);
    const b = stft(m2, n, hop, win);
    const plane = a.F * a.T;
    const x = tensor(4, a.F, a.T);
    for (let q = 0; q < plane; q++) {
      const ar = a.re[q];
      const ai = a.im[q];
      const br = b.re[q];
      const bi = b.im[q];
      x.d[q] = Math.log1p(ar * ar + ai * ai);
      x.d[plane + q] = Math.log1p(br * br + bi * bi);
      // X1 X2* = (ar + i ai)(br - i bi)
      const cr = ar * br + ai * bi;
      const ci = ai * br - ar * bi;
      const phi = Math.atan2(ci, cr);
      x.d[2 * plane + q] = Math.cos(phi);
      x.d[3 * plane + q] = Math.sin(phi);
    }
    return x;
  }

  /** Source branch: log1p(power spectrogram) -> SpectrogramEncoder -> 8x8 (cached per source). */
  private source(src: Float32Array): Tensor {
    if (this.sourceCache && this.sourceCache.key === src) return this.sourceCache.u;
    const { source_n_fft: n, source_hop: hop } = this.manifest.protocol;
    const s = stft(src, n, hop, this.p('source_spec.window'));
    let x = tensor(1, s.F, s.T);
    for (let q = 0; q < s.F * s.T; q++) x.d[q] = Math.log1p(s.re[q] * s.re[q] + s.im[q] * s.im[q]);
    x = relu(this.conv(x, 'encoder_u.net.0'));
    x = relu(this.conv(x, 'encoder_u.net.2'));
    x = maxPool2(x);
    x = relu(this.conv(x, 'encoder_u.net.5'));
    x = maxPool2(x);
    x = relu(this.conv(x, 'encoder_u.net.8'));
    const u = adaptiveAvgPool(x, 8);
    this.sourceCache = { key: src, u };
    return u;
  }

  /** Logits (64 x 64) for one pose: two mic recordings and the source signal. */
  forward(mic1: Float32Array, mic2: Float32Array, src: Float32Array): Float32Array {
    return this.forwardFeatures(this.front(mic1, mic2), src);
  }

  /** The 4-channel front-end features of one pose (exposed for tests and display). */
  features(mic1: Float32Array, mic2: Float32Array): Tensor {
    return this.front(mic1, mic2);
  }

  /** Logits from precomputed front-end features. */
  forwardFeatures(x: Tensor, src: Float32Array): Float32Array {
    const f1 = relu(this.conv(relu(this.conv(x, 'encoder_s.stage1.0')), 'encoder_s.stage1.2'));
    const f2 = relu(this.conv(relu(this.conv(maxPool2(f1), 'encoder_s.stage2.1.0')), 'encoder_s.stage2.1.2'));
    const f3 = relu(this.conv(relu(this.conv(maxPool2(f2), 'encoder_s.stage3.1.0')), 'encoder_s.stage3.1.2'));
    const s3 = adaptiveAvgPool(f3, 8);
    const s2 = adaptiveAvgPool(f2, 16);
    const s1 = adaptiveAvgPool(f1, 32);
    const u = this.source(src);
    let d = relu(this.convT(relu(this.conv(concat(s3, u), 'dec3.0')), 'dec3.2'));
    d = relu(this.convT(relu(this.conv(concat(d, s2), 'dec2.0')), 'dec2.2'));
    d = relu(this.convT(relu(this.conv(concat(d, s1), 'dec1.0')), 'dec1.2'));
    return this.conv(d, 'dec1.4', 1, 0).d;
  }
}

/** Calibrated Bayes fusion (learning/calibration.calibrated_bayes_fuse). */
export function fuse(logits: Float32Array[], cal: { temperature: number; bias: number; prior: number }): Float32Array {
  const k = logits.length;
  const pr = Math.min(1 - 1e-4, Math.max(1e-4, cal.prior));
  const pl = Math.log(pr / (1 - pr));
  const out = new Float32Array(logits[0].length);
  for (let i = 0; i < out.length; i++) {
    let s = 0;
    for (const l of logits) s += l[i] / cal.temperature + cal.bias;
    s -= (k - 1) * pl;
    out[i] = 1 / (1 + Math.exp(-s));
  }
  return out;
}
