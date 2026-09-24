/** Room twin: absorption mapping and the 3D FDTD shoebox against Eyring. */
import { describe, expect, it } from 'vitest';
import { eyring, sabine } from '../../src/lab/measure';
import { alphaRandom3d, betaForAlpha, simulateShoebox } from '../../src/lab/twin';

describe('room twin', () => {
  it('maps random-incidence alpha to admittance beta and back', () => {
    for (const a of [0.05, 0.2, 0.5, 0.8]) expect(alphaRandom3d(betaForAlpha(a))).toBeCloseTo(a, 3);
    // Paris average peaks near 0.95 (grazing incidence always reflects).
    expect(alphaRandom3d(0.6)).toBeGreaterThan(0.9);
    expect(alphaRandom3d(0.6)).toBeLessThan(0.97);
  });

  it('simulated T30 of a small shoebox is in the diffuse-theory ballpark', async () => {
    const room = { lx: 3, ly: 2.5, lz: 2.2, alpha: 0.2 };
    const r = await simulateShoebox(room, [0.9, 1.1, 0.7], { maxCells: 32, seconds: 0.7 });
    expect(r.cells.length).toBe(3);
    expect(r.ir.length).toBeGreaterThan(1000);
    const ey = eyring(room);
    const sb = sabine(room);
    console.log(`twin T30 ${r.t30?.toFixed(3)} s, T20 ${r.t20?.toFixed(3)} s, Eyring ${ey.toFixed(3)}, Sabine ${sb.toFixed(3)}`);
    const t = r.t30 ?? r.t20;
    expect(t).not.toBeNull();
    expect(t!).toBeGreaterThan(0.6 * ey);
    expect(t!).toBeLessThan(1.5 * sb);
  }, 60_000);
});
