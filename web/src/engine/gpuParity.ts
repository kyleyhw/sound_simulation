/**
 * GPU-vs-CPU parity check (used by tests/e2e/gpu.spec.ts and the About
 * panel): run the same scene on both backends and compare the fields.
 */
import { GpuStepper } from './gpu';
import { buildSimulation, type Scene } from './scene';

export interface ParityResult {
  maxAbs: number;
  rel: number;
  peak: number;
  probeMaxAbs: number;
  cpuMs: number;
  gpuMs: number;
}

export async function gpuParity(scene: Scene, steps: number, batches = 3): Promise<ParityResult> {
  const cpu = buildSimulation(scene);
  const gpuSim = buildSimulation(scene);
  const t0 = performance.now();
  for (let k = 0; k < steps; k++) cpu.step();
  const t1 = performance.now();
  const g = await GpuStepper.create(gpuSim);
  const per = Math.ceil(steps / batches);
  let done = 0;
  const t2 = performance.now();
  while (done < steps) {
    const k = Math.min(per, steps - done);
    await g.run(k, done + k >= steps);
    done += k;
  }
  const t3 = performance.now();
  g.destroy();
  let maxAbs = 0;
  let num = 0;
  let den = 0;
  let peak = 0;
  for (let i = 0; i < cpu.n; i++) {
    const d = cpu.p[i] - gpuSim.p[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
    num += d * d;
    den += cpu.p[i] * cpu.p[i];
    peak = Math.max(peak, Math.abs(cpu.p[i]));
  }
  let probeMaxAbs = 0;
  for (const pr of cpu.probes) {
    const a = cpu.probeSeries(pr.id);
    const b = gpuSim.probeSeries(pr.id);
    for (let k = 0; k < a.length; k++) probeMaxAbs = Math.max(probeMaxAbs, Math.abs(a[k] - (b[k] ?? NaN)));
  }
  return { maxAbs, rel: Math.sqrt(num / (den || 1)), peak, probeMaxAbs, cpuMs: t1 - t0, gpuMs: t3 - t2 };
}
