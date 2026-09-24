import { expect, test } from '@playwright/test';
import { open } from './helpers';

// Plan 4.3.5 / 6.7.1: in-browser sensing with uncertainty and a next-pose hint.
test('sensing panel runs the model in the browser and scores it against the baseline', async ({ page }) => {
  test.setTimeout(120_000);
  const errors = await open(page);
  await page.getByTestId('tab-sensing').click();
  await expect(page.getByTestId('sensing-panel')).toBeVisible({ timeout: 30_000 });
  await page.getByTestId('sense-random').click();
  await page.getByLabel('Poses to add').selectOption('2');
  await page.getByTestId('sense-run').click();
  await expect(page.getByTestId('sense-run')).toBeEnabled({ timeout: 60_000 });
  await expect(page.getByTestId('sense-scores')).toContainText('IoU');
  const v = await page.getByTestId('sense-scores').locator('.v').first().innerText();
  expect(Number.isFinite(parseFloat(v))).toBe(true);
  // The hint adds a pose where the map is least certain.
  await page.getByTestId('sense-next').click();
  await expect(page.getByTestId('sense-run')).toBeEnabled({ timeout: 60_000 });
  await expect(page.getByLabel('Fused obstacle probability (K = 3)')).toBeVisible();
  // The current sandbox scene can be sensed too.
  await page.getByTestId('sense-scene').click();
  await page.getByTestId('sense-run').click();
  await expect(page.getByLabel(/Fused obstacle probability \(K = 2\)/)).toBeVisible({ timeout: 60_000 });
  expect(errors).toEqual([]);
});
