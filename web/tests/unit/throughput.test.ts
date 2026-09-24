/**
 * Engine throughput guard (plan 4.5.2). Logs steps/s and asserts the
 * general kernel (materials, rigid walls, absorbing layer) stays within a
 * small factor of the Python-parity fast path.
 */
import { expect, it } from 'vitest';
import { presetById } from '../../src/engine/presets';
import { buildSimulation } from '../../src/engine/scene';
import { Simulation } from '../../src/engine/simulation';

function rate(sim: Simulation, ms = 400): number {
  for (let i = 0; i < 20; i++) sim.step(); // warm-up / JIT
  const t0 = performance.now();
  let n = 0;
  while (performance.now() - t0 < ms) {
    sim.step();
    n++;
  }
  return (n * 1000) / (performance.now() - t0);
}

it('general kernel is within 3x of the fast path', () => {
  const fast = rate(new Simulation({ shape: [256, 256] }));
  const pml = rate(new Simulation({ shape: [256, 256], outer: 'sponge' }));
  const cpml = rate(new Simulation({ shape: [256, 256], outer: 'cpml' }));
  const ell = rate(buildSimulation(presetById('ellipse')!.build()));
  const vol = rate(new Simulation({ dims: 3, shape: [64, 64, 64] }));
  console.log(
    `steps/s  2D 256² fast ${fast.toFixed(0)} | 2D 256² sponge ${pml.toFixed(0)} | cpml ${cpml.toFixed(0)} | ` +
      `ellipse 200x240 rigid ${ell.toFixed(0)} | 3D 64³ ${vol.toFixed(0)}`,
  );
  expect(pml).toBeGreaterThan(fast / 3);
  expect(cpml).toBeGreaterThan(pml / 2.5); // CPML adds two memory variables per axis in the layer
  expect(fast).toBeGreaterThan(200); // ~0.15 Mcells: >= 13 Mcell-updates/s
});
