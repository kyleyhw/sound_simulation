import { expect, test } from '@playwright/test';
import { open } from './helpers';

// Plan 10.9: the docs site renders every repository document, with maths and images.
test('docs site renders documents with maths, tables, images and in-app links', async ({ page }) => {
  const errors = await open(page, '#/docs/docs/physics');
  await expect(page.getByTestId('doc-body')).toContainText('Physics model and verification');
  await expect(page.getByTestId('doc-body').locator('.katex').first()).toBeVisible();
  await expect(page.getByTestId('doc-body').locator('table').first()).toBeVisible();
  // Navigation between documents stays in the app.
  await page.getByRole('link', { name: 'debug audit 2026 09 24' }).click();
  await expect(page).toHaveURL(/#\/docs\/tests\/reports\/debug_audit_2026_09_24/);
  await expect(page.getByTestId('doc-body')).toContainText('Debug audit');
  // Images (the README hero GIFs) resolve to bundled assets.
  await page.goto('./#/docs/README');
  const img = page.getByTestId('doc-body').locator('img').first();
  await expect(img).toBeVisible();
  expect(await img.evaluate((el: HTMLImageElement) => el.naturalWidth)).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});
