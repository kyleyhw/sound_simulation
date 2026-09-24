/** Gallery presets (plan 10.5): each one must actually show its physics. */
import { describe, expect, it } from 'vitest';
import { presetById } from '../../src/engine/presets';
import { buildSimulation } from '../../src/engine/scene';

function run(id: string, steps: number, from = 0) {
  const sim = buildSimulation(presetById(id)!.build());
  for (let k = 0; k < steps; k++) sim.step();
  const series = (pid: string) => sim.probeSeries(pid).slice(from);
  const rms = (x: Float32Array) => Math.sqrt(x.reduce((s, v) => s + v * v, 0) / Math.max(1, x.length));
  const peak = (x: Float32Array) => x.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  return { sim, series, rms, peak };
}

describe('gallery presets', () => {
  it('time reversal refocuses on the original source', () => {
    const { series, peak } = run('time-reversal', 820);
    const at = peak(series('p1'));
    const beside = peak(series('p2'));
    console.log(`time reversal: focus ${at.toFixed(3)}, 30 cells away ${beside.toFixed(3)}`);
    expect(at).toBeGreaterThan(2 * beside);
  }, 60_000);

  it('beam steering puts the beam where it was aimed', () => {
    const { series, rms } = run('beam-steering', 900, 500);
    const on = rms(series('p1'));
    const off = rms(series('p2'));
    const db = 20 * Math.log10(on / off);
    console.log(`beam steering: on/off beam ${db.toFixed(1)} dB`);
    expect(db).toBeGreaterThan(6);
  }, 60_000);

  it('the quiet-zone design keeps the quiet probe quiet', () => {
    const { series, rms } = run('quiet-zone', 1400, 900);
    const db = 20 * Math.log10(rms(series('p1')) / rms(series('p2')));
    console.log(`quiet zone: loud/quiet probes ${db.toFixed(1)} dB`);
    expect(db).toBeGreaterThan(20);
  }, 60_000);

  it('the whispering gallery carries sound along the wall', () => {
    const { series, peak } = run('whispering-gallery', 900);
    const far = peak(series('p1'));
    const inside = peak(series('p2'));
    console.log(`whispering gallery: far wall ${far.toFixed(3)}, inside (nearer) ${inside.toFixed(3)}`);
    expect(far).toBeGreaterThan(inside);
  }, 60_000);
});
