/** Browser sound-field control (plan 7.7): linear algebra, designs, and a
 * time-domain check that the applied weights deliver the predicted contrast. */
import { describe, expect, it } from 'vitest';
import { cabs2, cmul, cx, matvec, principalGeneralized, solve } from '../../src/control/complex';
import { contrastDb, design, type Geometry, lineArray, measureTransfer, type Rect, verifyContrast, weightedDrivers } from '../../src/control/soundfield';
import { DEFAULT_PARAMS } from '../../src/engine/simulation';

describe('complex linear algebra', () => {
  it('solves a complex system', () => {
    const A = [
      [cx(2, 1), cx(0, -1), cx(1)],
      [cx(1), cx(3, 0.5), cx(0, 2)],
      [cx(0, 1), cx(1, 1), cx(4)],
    ];
    const x = [cx(1, -1), cx(0.5, 2), cx(-1, 0.25)];
    const b = matvec(A, x);
    const y = solve(A, b);
    y.forEach((v, i) => expect(cabs2([v[0] - x[i][0], v[1] - x[i][1]])).toBeLessThan(1e-20));
  });

  it('finds the principal generalised eigenvector of a diagonal pencil', () => {
    const Rb = [
      [cx(1), cx(0)],
      [cx(0), cx(4)],
    ];
    const Rd = [
      [cx(1), cx(0)],
      [cx(0), cx(1)],
    ];
    const w = principalGeneralized(Rb, Rd, 100);
    expect(cabs2(w[1])).toBeCloseTo(1, 6);
    expect(cabs2(cmul(w[0], [1, 0]))).toBeLessThan(1e-8);
  });
});

describe('array designs in an anechoic room', () => {
  const params = { ...DEFAULT_PARAMS, shape: [90, 90], outer: 'cpml' as const, cpmlCells: 12 };
  const g: Geometry = { params, materials: new Uint8Array(90 * 90), speed: null };
  const speakers = lineArray([74, 45], 8, 4, 1, params.shape);
  const bright: Rect = [22, 22, 32, 32];
  const dark: Rect = [22, 56, 32, 66];
  const f = 0.05; // 20 cells per wavelength

  it('ACC reaches >= 10 dB contrast, beats delay-and-sum, and the time-domain run agrees', async () => {
    const T = await measureTransfer(g, speakers, bright, dark, f, { yieldEvery: 1e9 });
    const res = Object.fromEntries((['das', 'focus', 'pm', 'acc'] as const).map((k) => [k, contrastDb(T, design(k, T, speakers, params, bright))]));
    console.log('predicted contrast (dB)', JSON.stringify(Object.fromEntries(Object.entries(res).map(([k, v]) => [k, +v.toFixed(1)]))));
    expect(res.acc).toBeGreaterThanOrEqual(10);
    expect(res.acc).toBeGreaterThanOrEqual(res.das - 1e-6);
    expect(res.acc).toBeGreaterThanOrEqual(res.pm - 1e-6);
    const w = design('acc', T, speakers, params, bright);
    const measured = verifyContrast(g, weightedDrivers(speakers, w, f), bright, dark, f);
    console.log(`ACC predicted ${res.acc.toFixed(1)} dB, measured ${measured.toFixed(1)} dB`);
    expect(measured).toBeGreaterThanOrEqual(10);
    expect(Math.abs(measured - res.acc)).toBeLessThan(3);
  }, 120_000);
});
