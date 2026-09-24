import { describe, expect, it } from 'vitest';
import { PRESETS } from '../../src/engine/presets';
import { buildSimulation, decodeRle, decodeSceneUrl, emptyScene, encodeRle, encodeSceneUrl, validateScene } from '../../src/engine/scene';
import { absorptionFromBeta, Simulation } from '../../src/engine/simulation';
import { evalWaveform } from '../../src/engine/waveforms';
import { fft, magnitudeSpectrum } from '../../src/lib/dsp';
import { discCells, ellipseCells, lineCells, rectCells } from '../../src/lib/geometry';

describe('scene serialisation', () => {
  it('RLE round-trips', () => {
    const a = new Uint8Array(1000);
    a.fill(2, 100, 350);
    a.fill(5, 900);
    expect(Array.from(decodeRle(encodeRle(a), 1000))).toEqual(Array.from(a));
  });
  it('rejects inconsistent RLE', () => {
    expect(() => decodeRle('0:10', 11)).toThrow();
  });
  it('share URL round-trips every preset', async () => {
    for (const p of PRESETS) {
      const s = p.build();
      const back = await decodeSceneUrl(await encodeSceneUrl(s));
      expect(back.materials).toBe(s.materials);
      expect(back.drivers.length).toBe(s.drivers.length);
    }
  });
  it('validates scenes', () => {
    expect(() => validateScene({})).toThrow();
    const s = emptyScene({ shape: [32, 32] });
    expect(validateScene(JSON.parse(JSON.stringify(s))).params.shape).toEqual([32, 32]);
    expect(() => validateScene({ ...s, params: { ...s.params, shape: [5, 5] } })).toThrow();
  });
  it('every preset builds and runs finite', () => {
    for (const p of PRESETS) {
      const sim = buildSimulation(p.build());
      for (let i = 0; i < 50; i++) sim.step();
      let finite = true;
      for (let i = 0; i < sim.n; i++) if (!Number.isFinite(sim.p[i])) finite = false;
      expect(finite, p.id).toBe(true);
    }
  });
});

describe('physics invariants', () => {
  it('conserves discrete energy in a closed lossless box once the source is off', () => {
    const sim = new Simulation({ shape: [80, 80] });
    sim.setDrivers([{ id: 'd', pos: [40, 40], waveform: { type: 'ricker', amplitude: 1, frequency: 0.1, delay: 15 }, enabled: true }]);
    for (let i = 0; i < 120; i++) sim.step(); // pulse finished by t ~ 40
    const e0 = sim.energy();
    for (let i = 0; i < 400; i++) sim.step();
    expect(Math.abs(sim.energy() - e0) / e0).toBeLessThan(1e-3);
  });
  it('rigid walls also conserve energy (general kernel)', () => {
    const sim = new Simulation({ shape: [64, 64], outer: 'rigid' });
    sim.setDrivers([{ id: 'd', pos: [30, 30], waveform: { type: 'ricker', amplitude: 1, frequency: 0.1, delay: 15 }, enabled: true }]);
    for (let i = 0; i < 120; i++) sim.step();
    const e0 = sim.energy();
    for (let i = 0; i < 400; i++) sim.step();
    expect(Math.abs(sim.energy() - e0) / e0).toBeLessThan(2e-2);
  });
  it('absorbing walls lose energy monotonically', () => {
    const sim = new Simulation({ shape: [64, 64], outer: 'absorb', outerBeta: 1 });
    sim.setDrivers([{ id: 'd', pos: [30, 30], waveform: { type: 'ricker', amplitude: 1, frequency: 0.1, delay: 15 }, enabled: true }]);
    for (let i = 0; i < 80; i++) sim.step();
    const e0 = sim.energy();
    for (let i = 0; i < 600; i++) sim.step();
    expect(sim.energy()).toBeLessThan(0.05 * e0);
  });
  it('absorption coefficient from admittance', () => {
    expect(absorptionFromBeta(1)).toBeCloseTo(1);
    expect(absorptionFromBeta(0)).toBeCloseTo(0);
  });
  it('Ricker formula matches the Python definition', () => {
    const w = { type: 'ricker', amplitude: 2, frequency: 0.1, delay: 20 } as const;
    expect(evalWaveform(w, 20)).toBeCloseTo(2);
    const arg = Math.PI * 0.1 * 3;
    expect(evalWaveform(w, 23)).toBeCloseTo(2 * (1 - 2 * arg * arg) * Math.exp(-arg * arg));
  });
});

describe('dsp + geometry', () => {
  it('FFT finds a pure tone', () => {
    const n = 256;
    const x = Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * 16 * i) / n));
    const s = magnitudeSpectrum(x, n);
    let k = 0;
    for (let i = 1; i < s.length; i++) if (s[i] > s[k]) k = i;
    expect(k).toBe(16);
  });
  it('inverse FFT restores the input', () => {
    const re = Float64Array.from([1, 2, 3, 4, 0, -1, 5, 2]);
    const im = new Float64Array(8);
    const orig = re.slice();
    fft(re, im);
    fft(re, im, true);
    for (let i = 0; i < 8; i++) expect(re[i]).toBeCloseTo(orig[i]);
  });
  it('raster shapes stay in bounds and have expected sizes', () => {
    expect(discCells(5, 5, 1, 10, 10)).toEqual([[5, 5]]);
    expect(rectCells([0, 0], [3, 3], 1, true, 10, 10).length).toBe(16);
    expect(rectCells([0, 0], [3, 3], 1, false, 10, 10).length).toBe(12);
    for (const [r, c] of ellipseCells([-5, -5], [20, 20], 2, false, 10, 10)) {
      expect(r >= 0 && r < 10 && c >= 0 && c < 10).toBe(true);
    }
    expect(lineCells([0, 0], [0, 9], 1, 10, 10).length).toBe(10);
  });
});
