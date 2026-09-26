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

  await page.getByTestId('design').click();
  await expect(page.getByTestId('predicted')).toBeVisible({ timeout: 60_000 });
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
  expect(errors).toEqual([]);
});
