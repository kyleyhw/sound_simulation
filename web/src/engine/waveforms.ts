/**
 * Source waveforms. Formulas mirror src/acoustic_system/simulation/waveforms.py
 * exactly (same parameter names and defaults), so a scene built here drives
 * the Python engine identically and vice versa.
 */

export type WaveformSpec =
  | { type: 'ricker'; amplitude: number; frequency: number; delay: number }
  | { type: 'gaussian'; amplitude: number; center_time: number; width: number }
  | { type: 'cosine'; amplitude: number; frequency: number }
  | { type: 'chirp'; amplitude: number; f0: number; f1: number; duration: number; delay: number }
  | { type: 'burst'; amplitude: number; frequency: number; cycles: number; delay: number }
  | { type: 'noise'; amplitude: number; seed: number; duration: number }
  | { type: 'samples'; amplitude: number; rate: number; delay: number; data: number[] };

export type WaveformType = WaveformSpec['type'];

export const WAVEFORM_LABELS: Record<WaveformType, string> = {
  ricker: 'Ricker pulse',
  gaussian: 'Gaussian pulse',
  cosine: 'Continuous tone',
  chirp: 'Linear chirp',
  burst: 'Tone burst',
  noise: 'White noise',
  samples: 'Recorded signal',
};

/** Sensible defaults in grid units (c = dx = 1, dt = 0.5). */
export function defaultWaveform(type: WaveformType): WaveformSpec {
  switch (type) {
    case 'ricker':
      return { type, amplitude: 5, frequency: 0.1, delay: 20 };
    case 'gaussian':
      return { type, amplitude: 5, center_time: 15, width: 3 };
    case 'cosine':
      return { type, amplitude: 1, frequency: 0.05 };
    case 'chirp':
      return { type, amplitude: 5, f0: 0.02, f1: 0.2, duration: 100, delay: 0 };
    case 'burst':
      return { type, amplitude: 3, frequency: 0.08, cycles: 4, delay: 5 };
    case 'noise':
      return { type, amplitude: 1, seed: 1, duration: 200 };
    case 'samples':
      return { type, amplitude: 1, rate: 2, delay: 0, data: [] };
  }
}

/** Deterministic hash noise in [-1, 1) for sample index n (seeded). */
function hashNoise(n: number, seed: number): number {
  let x = (n * 374761393 + seed * 668265263) | 0;
  x = Math.imul(x ^ (x >>> 13), 1274126177);
  x ^= x >>> 16;
  return ((x >>> 0) / 4294967296) * 2 - 1;
}

/** Evaluate a waveform at simulation time t. */
export function evalWaveform(w: WaveformSpec, t: number): number {
  switch (w.type) {
    case 'ricker': {
      const arg = Math.PI * w.frequency * (t - w.delay);
      const arg2 = arg * arg;
      return w.amplitude * (1 - 2 * arg2) * Math.exp(-arg2);
    }
    case 'gaussian':
      return w.amplitude * Math.exp(-((t - w.center_time) ** 2) / (2 * w.width * w.width));
    case 'cosine':
      return w.amplitude * Math.cos(2 * Math.PI * w.frequency * t);
    case 'chirp': {
      // Linear sweep u(t) = A sin(2 pi (f0 + k t / 2) t), k = (f1 - f0) / T,
      // the same form as dataset.synthetic_chirp, with a raised-cosine
      // taper over the first/last 10 % to avoid switch-on clicks.
      const tt = t - w.delay;
      if (tt < 0 || tt > w.duration) return 0;
      const k = (w.f1 - w.f0) / w.duration;
      const edge = 0.1 * w.duration;
      let env = 1;
      if (tt < edge) env = 0.5 - 0.5 * Math.cos((Math.PI * tt) / edge);
      else if (tt > w.duration - edge) env = 0.5 - 0.5 * Math.cos((Math.PI * (w.duration - tt)) / edge);
      return w.amplitude * env * Math.sin(2 * Math.PI * (w.f0 + 0.5 * k * tt) * tt);
    }
    case 'burst': {
      const tt = t - w.delay;
      const T = w.cycles / w.frequency;
      if (tt < 0 || tt > T) return 0;
      const env = 0.5 - 0.5 * Math.cos((2 * Math.PI * tt) / T); // Hann window
      return w.amplitude * env * Math.sin(2 * Math.PI * w.frequency * tt);
    }
    case 'noise': {
      if (t < 0 || t > w.duration) return 0;
      return w.amplitude * hashNoise(Math.floor(t * 8), w.seed);
    }
    case 'samples': {
      // Linear interpolation, zero outside support (AudioFileWaveform).
      const x = (t - w.delay) * w.rate;
      if (x < 0) return 0;
      const k = Math.floor(x);
      if (k + 1 >= w.data.length) return 0;
      const a = x - k;
      return w.amplitude * ((1 - a) * w.data[k] + a * w.data[k + 1]);
    }
  }
}

/** Dominant frequency of a waveform (for Nyquist / resolution warnings). */
export function nominalFrequency(w: WaveformSpec): number {
  switch (w.type) {
    case 'ricker':
    case 'cosine':
    case 'burst':
      return w.frequency;
    case 'gaussian':
      return 1 / (2 * Math.PI * w.width);
    case 'chirp':
      return Math.max(w.f0, w.f1);
    case 'noise':
      return 4;
    case 'samples':
      return w.rate / 2;
  }
}
