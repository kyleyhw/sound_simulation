import { expect, test } from '@playwright/test';
import { open } from './helpers';

// Plan 10.1 / 10.2: the WebGPU backend reproduces the CPU engine on every
// physics path (fast path, rigid, impedance, sponge, CPML, Mur, c(x), 3D).
// Headless Chromium provides a SwiftShader (CPU) WebGPU adapter, which
// checks correctness, not speed.
const CASES: { id: string; steps: number; mutate?: string }[] = [
  { id: 'pulse-room', steps: 240 },
  { id: 'ellipse', steps: 240 },
  { id: 'helmholtz', steps: 240 },
  { id: 'lens', steps: 200 },
  { id: 'echolocation', steps: 240 },
  { id: 'quiet-zone', steps: 200 },
  { id: 'pulse-room', steps: 200, mutate: "s.params.faces = ['mur', 'absorb', 'sponge', 'rigid']; s.params.outerBeta = 0.4; s.params.spongeCells = 12;" },
  { id: 'pulse-room', steps: 120, mutate: "s.params.dims = 3; s.params.shape = [40, 36, 32]; s.params.outer = 'cpml'; s.params.cpmlCells = 8; s.materials = '0:46080'; s.speed = undefined; s.drivers = [{ id: 'd', pos: [20, 18, 16], waveform: { type: 'ricker', amplitude: 5, frequency: 0.12, delay: 12 }, enabled: true }]; s.probes = [{ id: 'p', pos: [10, 10, 10] }];" },
];

test('WebGPU backend matches the CPU engine', async ({ page }) => {
  test.setTimeout(300_000);
  const errors = await open(page, '#/gallery');
  const hasGpu = await page.evaluate(async () => !!(navigator.gpu && (await navigator.gpu.requestAdapter())));
  test.skip(!hasGpu, 'no WebGPU adapter in this browser');
  for (const c of CASES) {
    const r = await page.evaluate(
      `(async () => { const s = window.__presets.find((p) => p.id === '${c.id}').build(); ${c.mutate ?? ''} return window.__gpuParity(s, ${c.steps}, 3); })()`,
    );
    const res = r as { maxAbs: number; rel: number; peak: number; probeMaxAbs: number; cpuMs: number; gpuMs: number };
    console.log(`${c.id}${c.mutate ? ' (mutated)' : ''}: rel ${res.rel.toExponential(2)} maxAbs ${res.maxAbs.toExponential(2)} peak ${res.peak.toFixed(3)} probes ${res.probeMaxAbs.toExponential(2)}`);
    expect(res.peak).toBeGreaterThan(0);
    expect(res.rel).toBeLessThan(1e-4);
    expect(res.maxAbs).toBeLessThan(1e-4 * Math.max(1, res.peak));
    expect(res.probeMaxAbs).toBeLessThan(1e-4 * Math.max(1, res.peak));
  }
  expect(errors).toEqual([]);
});

test('the sandbox runs on the GPU and switches back without losing state', async ({ page }) => {
  const errors = await open(page);
  const hasGpu = await page.evaluate(async () => !!(navigator.gpu && (await navigator.gpu.requestAdapter())));
  test.skip(!hasGpu, 'no WebGPU adapter in this browser');
  const app = (fn: string) => page.evaluate(`(() => { const s = window.__app.getState(); return (${fn})(s); })()`);
  await expect(page.getByLabel('Compute backend').locator('option[value="gpu"]')).toBeEnabled();
  await page.getByLabel('Compute backend').selectOption('gpu');
  await expect.poll(() => app('(s) => s.runtime.backend')).toBe('gpu');
  await page.getByTestId('run').click();
  await expect.poll(() => app('(s) => s.runtime.sim.step_count'), { timeout: 30_000 }).toBeGreaterThan(60);
  await page.getByTestId('run').click();
  // Probe samples arrive from the device.
  expect((await app("(s) => s.runtime.sim.probeSeries(s.scene.probes[0].id).length")) as number).toBeGreaterThan(60);
  const before = (await app('(s) => s.runtime.sim.step_count')) as number;
  await page.getByLabel('Compute backend').selectOption('cpu');
  await expect.poll(() => app('(s) => s.runtime.backend')).toBe('cpu');
  expect(await app('(s) => s.runtime.sim.step_count')).toBe(before);
  await page.getByTestId('step').click();
  expect(await app('(s) => s.runtime.sim.step_count')).toBe(before + 1);
  expect(await app('(s) => Number.isFinite(s.runtime.peak()) && s.runtime.peak() > 0')).toBe(true);
  expect(errors).toEqual([]);
});
