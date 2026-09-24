/**
 * Engine throughput guard (plan 4.5.2). Logs steps/s and asserts the
 * general kernel (materials, rigid walls, absorbing layers) stays within a
 * small factor of the Python-parity fast path.
 */
import { expect, it } from 'vitest';
import { presetById } from '../../src/engine/presets';
import { buildSimulation } from '../../src/engine/scene';
import { Simulation } from '../../src/engine/simulation';

/** Best of three timing windows: shared CI runners are noisy, and the best
 * window is the least disturbed estimate of the kernel's own speed. */
function rate(sim: Simulation, ms = 200): number {
  for (let i = 0; i < 20; i++) sim.step(); // warm-up / JIT
  let best = 0;
  for (let w = 0; w < 3; w++) {
    const t0 = performance.now();
    let n = 0;
    while (performance.now() - t0 < ms) {
      sim.step();
      n++;
    }
    best = Math.max(best, (n * 1000) / (performance.now() - t0));
  }
  return best;
}

// Regression guard, not a benchmark. The general/fast ratio depends on the
// machine (2.8x on the 4-core dev container, 5-6.4x on GitHub runners), so
// the bounds only catch order-of-magnitude slowdowns, plus absolute floors.
it('general kernels stay within an order of magnitude of the fast path', () => {
  const fast = rate(new Simulation({ shape: [256, 256] }));
  const pml = rate(new Simulation({ shape: [256, 256], outer: 'sponge' }));
  const cpml = rate(new Simulation({ shape: [256, 256], outer: 'cpml' }));
  const ell = rate(buildSimulation(presetById('ellipse')!.build()));
  const vol = rate(new Simulation({ dims: 3, shape: [64, 64, 64] }));
  console.log(
    `steps/s  2D 256² fast ${fast.toFixed(0)} | 2D 256² sponge ${pml.toFixed(0)} | cpml ${cpml.toFixed(0)} | ` +
      `ellipse 200x240 rigid ${ell.toFixed(0)} | 3D 64³ ${vol.toFixed(0)}`,
  );
  expect(pml).toBeGreaterThan(fast / 10);
  expect(cpml).toBeGreaterThan(fast / 12); // CPML adds two memory variables per axis in the layer
  expect(fast).toBeGreaterThan(200); // 65k cells: >= 13 Mcell-updates/s
  expect(pml).toBeGreaterThan(60); // >= 4 Mcell-updates/s on the general path
  expect(cpml).toBeGreaterThan(40);
});
