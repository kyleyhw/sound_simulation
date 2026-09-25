import { expect, test } from '@playwright/test';
import { open } from './helpers';

// Plan 10.11/10.12: write-ups hosted on the site.
test('research page lists write-ups and renders them', async ({ page }) => {
  const errors = await open(page, '#/research');
  await expect(page.getByTestId('writeup-card').first()).toBeVisible();
  expect(await page.getByTestId('writeup-card').count()).toBeGreaterThanOrEqual(3);
  await page.getByTestId('writeup-card').filter({ hasText: 'What can a laptop hear?' }).click();
  await expect(page.getByTestId('writeup')).toContainText('no-audio baseline');
  await expect(page.getByTestId('writeup').locator('table').first()).toBeVisible();
  await expect(page.getByTestId('writeup').locator('.katex').first()).toBeVisible();
  // Report links stay in the app.
  await page.getByRole('link', { name: 'physics imaging report' }).click();
  await expect(page).toHaveURL(/#\/docs\/tests\/reports\/imaging_2026_09_24/);
  expect(errors).toEqual([]);
});
