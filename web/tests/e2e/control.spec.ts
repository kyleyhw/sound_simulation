import { expect, test } from '@playwright/test';
import { app, drag, open } from './helpers';

// Plan 7.7: zones, speaker array, measured design, live contrast.
test('control panel designs an ACC array that makes a quiet zone', async ({ page }) => {
  const errors = await open(page);
  await page.evaluate(`(() => { const s = window.__app.getState(); s.setParams({ shape: [90, 90], outer: 'cpml', cpmlCells: 12 }); s.clearObstacles(); s.clearDrivers(); })()`);
  await page.getByTestId('tab-control').click();
  await expect(page.getByTestId('control-panel')).toBeVisible();

  await page.getByTestId('zone-tool-bright').click();
  await drag(page, [0.25, 0.25], [0.36, 0.36]);
  await page.getByTestId('zone-tool-dark').click();
  await drag(page, [0.62, 0.25], [0.73, 0.36]);
  await expect(page.getByTestId('zone-bright')).toBeVisible();
  await expect(page.getByTestId('zone-dark')).toBeVisible();

  await page.getByTestId('place-array').click();
  expect(await app<number>(page, "(s) => s.scene.drivers.filter((d) => d.id.startsWith('arr-')).length")).toBe(8);

  // B07: the measurement runs in a worker: the main thread stays free.
  await page.evaluate(() => {
    const w = window as unknown as { __long: number[] };
    w.__long = [];
    new PerformanceObserver((l) => l.getEntries().forEach((e) => w.__long.push(e.duration))).observe({ type: 'longtask', buffered: false });
  });
  await page.getByTestId('design').click();
  await expect(page.getByTestId('cancel-design')).toBeVisible();
  await expect(page.getByTestId('predicted')).toBeVisible({ timeout: 60_000 });
  const long = await page.evaluate(() => (window as unknown as { __long: number[] }).__long);
  expect(Math.max(0, ...long)).toBeLessThan(400);
  const acc = await page.getByTestId('predicted').locator('tr', { hasText: 'ACC' }).locator('td.mono').innerText();
  expect(parseFloat(acc)).toBeGreaterThan(10);
  // B19: the measured design survives a switch to another inspector tab and back.
  await page.getByTestId('tab-scene').click();
  await expect(page.getByTestId('control-panel')).toBeHidden();
  await page.getByTestId('tab-control').click();
  await expect(page.getByTestId('predicted')).toBeVisible();
  await expect(page.getByTestId('apply-design')).toBeVisible();
  // Weights were applied as gains/delays.
  expect(await app<boolean>(page, "(s) => s.scene.drivers.filter((d) => d.id.startsWith('arr-')).some((d) => (d.delay ?? 0) > 0)")).toBe(true);

  await page.getByTestId('run').click();
  await expect.poll(() => app<number>(page, '(s) => s.runtime.sim.step_count'), { timeout: 30_000 }).toBeGreaterThan(900);
  await page.getByRole('button', { name: 'Reset averaging' }).click();
  await expect.poll(() => app<number>(page, '(s) => s.runtime.sim.step_count'), { timeout: 30_000 }).toBeGreaterThan(1300);
  await expect(page.getByTestId('live-contrast')).toContainText('dB');
  const live = parseFloat(await page.getByTestId('live-contrast').locator('.v').innerText());
  expect(live).toBeGreaterThan(10);

  // B16/B17: clearing the zones blanks the readout and disables Apply, with a reason.
  await page.getByTestId('clear-zones').click();
  await expect(page.getByTestId('live-contrast').locator('.v')).toHaveText('—');
  await expect(page.getByTestId('apply-design')).toBeDisabled();
  await expect(page.getByTestId('apply-blocked')).toContainText('Draw a loud and a quiet zone');
  expect(errors).toEqual([]);
});

test('B07: a running measurement can be cancelled', async ({ page }) => {
  await open(page);
  await page.getByTestId('tab-control').click();
  await page.getByTestId('zone-tool-bright').click();
  await drag(page, [0.25, 0.25], [0.36, 0.36]);
  await page.getByTestId('zone-tool-dark').click();
  await drag(page, [0.62, 0.25], [0.73, 0.36]);
  await page.getByTestId('place-array').click();
  await page.getByTestId('design').click();
  await expect(page.getByTestId('design')).toContainText('Measuring');
  // The UI still responds while it measures.
  const before = await app<number>(page, '(s) => s.runtime.sim.step_count');
  await page.getByTestId('step').click();
  expect(await app<number>(page, '(s) => s.runtime.sim.step_count')).toBe(before + 1);
  await expect(page.getByTestId('design')).toContainText('Measuring');
  await page.getByTestId('cancel-design').click();
  await expect(page.getByTestId('design')).toHaveText('Measure room & design');
  await expect(page.getByTestId('predicted')).toHaveCount(0);
});
