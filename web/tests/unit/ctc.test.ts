import { describe, expect, it } from 'vitest';
import { designCtc, inverse2x2, plant, separationDb, stereoSeparationDb } from '../../src/control/ctc';

const geo = { speakerSpan: 0.3, head: { x: 0, y: 0.5 } };

describe('crosstalk cancellation', () => {
  it('regularised inverse approaches H^-1', () => {
    const H = plant(geo, 2000);
    const C = inverse2x2(H, 1e-9);
    // H C ~ I
    const m = (a: number[], b: number[]) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
    const e00 = [m(H[0][0], C[0][0])[0] + m(H[0][1], C[1][0])[0], m(H[0][0], C[0][0])[1] + m(H[0][1], C[1][0])[1]];
    const e10 = [m(H[1][0], C[0][0])[0] + m(H[1][1], C[1][0])[0], m(H[1][0], C[0][0])[1] + m(H[1][1], C[1][0])[1]];
    expect(Math.hypot(e00[0], e00[1])).toBeCloseTo(1, 4);
    expect(Math.hypot(e10[0], e10[1])).toBeLessThan(1e-4);
  });

  it('FIR filters give >= 15 dB separation in band at the design position', () => {
    const f = designCtc(geo, 48000, 2048, 0.002);
    const freqs = [500, 1000, 2000, 3000, 4000];
    const sep = separationDb(f, geo, geo, freqs);
    const nat = stereoSeparationDb(geo, freqs);
    for (let q = 0; q < freqs.length; q++) {
      expect(sep[q]).toBeGreaterThan(15);
      expect(sep[q]).toBeGreaterThan(nat[q] + 10);
    }
  });

  it('separation degrades when the head moves away from the design point', () => {
    const f = designCtc(geo, 48000, 2048, 0.002);
    const at = separationDb(f, geo, geo, [2000])[0];
    const off = separationDb(f, geo, { ...geo, head: { x: 0.05, y: 0.5 } }, [2000])[0];
    expect(off).toBeLessThan(at - 5);
  });
});
