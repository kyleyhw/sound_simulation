/**
 * Auralisation: play a probe recording through Web Audio.
 *
 * In SI units the recording already has a physical sample rate (1/dt), so it
 * is resampled to the audio context rate and played at true pitch. In grid
 * units there is no physical time scale; the recording is pitch-mapped so
 * that its dominant frequency lands at `targetHz` (default 440 Hz), i.e. a
 * sample rate of fs = f_audio / f_sim per simulation time unit.
 */

import { magnitudeSpectrum, peakAbs } from './dsp';

let ctx: AudioContext | null = null;
let current: AudioBufferSourceNode | null = null;

export function audioContext(): AudioContext {
  if (!ctx) ctx = new AudioContext();
  return ctx;
}

/** Dominant frequency (cycles per sample) of a signal. */
export function dominantCyclesPerSample(x: Float32Array): number {
  if (x.length < 16) return 0.05;
  const spec = magnitudeSpectrum(x);
  let best = 1;
  for (let k = 2; k < spec.length; k++) if (spec[k] > spec[best]) best = k;
  const n = (spec.length - 1) * 2;
  return best / n;
}

export interface PlayOptions {
  /** Simulation timestep (seconds when units are SI). */
  dt: number;
  units: 'grid' | 'si';
  targetHz?: number;
  loop?: boolean;
}

export async function playSignal(x: Float32Array, opt: PlayOptions): Promise<{ rate: number; seconds: number }> {
  stopPlayback();
  const ac = audioContext();
  if (ac.state === 'suspended') await ac.resume();
  let sampleRate: number;
  if (opt.units === 'si') sampleRate = 1 / opt.dt;
  else {
    const cps = Math.max(dominantCyclesPerSample(x), 1e-4);
    sampleRate = (opt.targetHz ?? 440) / cps;
  }
  // Web Audio accepts buffer rates 3 kHz-768 kHz; stretch if needed.
  const rate = Math.min(384000, Math.max(3000, sampleRate));
  const peak = peakAbs(x) || 1;
  // Fade in/out 5 ms to avoid clicks; normalise to -3 dBFS.
  const buf = ac.createBuffer(1, x.length, rate);
  const ch = buf.getChannelData(0);
  const fade = Math.max(1, Math.floor(rate * 0.005));
  for (let i = 0; i < x.length; i++) {
    const env = Math.min(1, i / fade, (x.length - 1 - i) / fade);
    ch[i] = (0.7 * x[i] * env) / peak;
  }
  const src = ac.createBufferSource();
  src.buffer = buf;
  src.loop = !!opt.loop;
  src.connect(ac.destination);
  src.start();
  current = src;
  return { rate, seconds: x.length / rate };
}

export function stopPlayback(): void {
  try {
    current?.stop();
  } catch {
    /* already stopped */
  }
  current = null;
}
