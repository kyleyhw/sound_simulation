import { expect, test } from '@playwright/test';
import { open } from './helpers';

// B04: in-page anchor links in rendered Markdown scroll to their target
// instead of being routed (the hash router used to open the Sandbox).
test('B04: in-page anchor links scroll within the document', async ({ page }) => {
  const errors = await open(page, '#/docs/docs/gpu');
  const link = page.locator('a[data-anchor="ref-cuda-guide"]').first();
  await expect(link).toBeVisible();
  await link.click();
  await expect(page).toHaveURL(/#\/docs\/docs\/gpu/);
  await expect(page.getByTestId('doc-body')).toBeVisible();
  await expect(page.locator('#ref-cuda-guide')).toBeInViewport();
  expect(errors).toEqual([]);
});

test('B04: heading anchors work, and ?h= deep links scroll to the target', async ({ page }) => {
  await open(page, '#/docs/tests/reports/imaging_2026_09_24');
  const link = page.locator('a[data-anchor="limitations"]').first();
  await expect(link).toBeVisible();
  await link.click();
  await expect(page).toHaveURL(/imaging_2026_09_24/);
  await expect(page.locator('h2#limitations, h3#limitations, h1#limitations').first()).toBeInViewport();

  await page.goto('./#/docs/docs/gpu?h=ref-cuda-guide');
  await expect(page.locator('#ref-cuda-guide')).toBeInViewport();
});

test('B33: maths renders without KaTeX errors or raw dollar signs', async ({ page }) => {
  await open(page, '#/docs/docs/imaging');
  await expect(page.locator('.katex').first()).toBeVisible();
  await expect(page.locator('.katex-error')).toHaveCount(0);
  await page.goto('./#/docs/docs/simulate');
  await expect(page.locator('.katex').first()).toBeVisible();
  await expect(page.getByTestId('doc-body')).not.toContainText('$64^3');
});
