import { expect, test } from '@playwright/test';
import { open } from './helpers';

// Plan 8.6: the closed-loop dashboard runs the loop in a worker and reports every epoch.
test('closed-loop dashboard runs four epochs', async ({ page }) => {
  test.setTimeout(240_000);
  const errors = await open(page, '#/loop');
  await expect(page.getByTestId('loop')).toBeVisible();
  await page.getByLabel('Grid size').selectOption('60');
  await page.getByTestId('run-loop').click();
  await expect(page.getByTestId('loop-results')).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId('loop-results').locator('tbody tr')).toHaveCount(4, { timeout: 200_000 });
  await expect(page.getByTestId('run-loop')).toBeEnabled();
  const cells = await page.getByTestId('loop-results').locator('tbody tr td:nth-child(2) b').allInnerTexts();
  expect(cells.map(parseFloat).every((v) => Number.isFinite(v))).toBe(true);
  expect(errors).toEqual([]);
});
