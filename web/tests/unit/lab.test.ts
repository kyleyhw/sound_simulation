/** Measurement DSP on synthetic rooms with known answers. */
import { describe, expect, it } from 'vitest';
import { alphaFromT60, convolve, decayMetrics, deconvolve, essInverse, essSweep, eyring, findEchoes, sabine } from '../../src/lab/measure';

const fs = 48000;
const spec = { f1: 100, f2: 16000, seconds: 1.0, sampleRate: fs };

/** Synthetic impulse response: direct + echoes + exponential diffuse tail. */
function syntheticRir(t60: number, echoes: [number, number][], len = 0.8): Float32Array {
  const n = Math.round(len * fs);
  const h = new Float32Array(n);
  const d0 = 200;
  h[d0] = 1;
  for (const [ms, g] of echoes) h[d0 + Math.round((ms / 1000) * fs)] += g;
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  const k = (3 * Math.log(10)) / t60; // amplitude decay rate: 60 dB in t60 for energy
  for (let i = d0 + 600; i < n; i++) h[i] += 0.05 * rnd() * Math.exp((-k * (i - d0)) / fs);
  return h;
}

describe('ESS deconvolution', () => {
  it('recovers a known impulse response', () => {
    const sweep = essSweep(spec);
    const inv = essInverse(spec, sweep);
    const h = syntheticRir(0.5, [[5.83, 0.6], [11.66, 0.4]]);
    const rec = Float32Array.from(convolve(sweep, h));
    const est = deconvolve(rec, inv, h.length);
    let peak = 0;
    let at = 0;
    for (let i = 0; i < est.length; i++) if (Math.abs(est[i]) > peak) (peak = Math.abs(est[i])), (at = i);
    expect(at).toBe(200); // direct sound at the right sample
    const { echoes } = findEchoes(est, fs, { thresholdDb: -12 });
    expect(echoes.length).toBeGreaterThanOrEqual(2);
    // 5.83 ms round trip = 1.0 m to the reflector.
    expect(echoes[0].distance).toBeCloseTo(1.0, 1);
    expect(echoes[1].distance).toBeCloseTo(2.0, 1);
  });

  it('estimates T60 from the decay', () => {
    for (const t60 of [0.3, 0.6]) {
      const h = syntheticRir(t60, [], 1.5);
      const m = decayMetrics(h, fs, 800);
      expect(m.t30!).toBeGreaterThan(t60 * 0.9);
      expect(m.t30!).toBeLessThan(t60 * 1.1);
    }
  });
});

describe('room twin formulas', () => {
  it('Sabine and Eyring behave and invert', () => {
    const room = { lx: 5, ly: 4, lz: 2.7, alpha: 0.2 };
    expect(sabine(room)).toBeGreaterThan(eyring(room));
    const t = eyring(room);
    expect(alphaFromT60(room, t)).toBeCloseTo(0.2, 6);
  });
});
