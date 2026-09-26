import { expect, test } from '@playwright/test';
import { open } from './helpers';

// Plan 8.6: the closed-loop dashboard runs the loop in a worker and reports every epoch.
test('closed-loop dashboard runs four epochs', async ({ page }) => {
  test.setTimeout(240_000);
  const errors = await open(page, '#/loop');
  await expect(page.getByTestId('loop')).toBeVisible();
  await page.getByLabel('Grid size').selectOption('60');
  await page.getByTestId('run-loop').click();
  // The learned model is trained for 100²; other grids fall back to back-projection, and the page says so.
  await expect(page.getByTestId('loop-estimator')).toHaveAttribute('data-estimator', 'backprojection', { timeout: 30_000 });
  await expect(page.getByTestId('loop-results')).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId('loop-results').locator('tbody tr')).toHaveCount(4, { timeout: 200_000 });
  await expect(page.getByTestId('run-loop')).toBeEnabled();
  const cells = await page.getByTestId('loop-results').locator('tbody tr td:nth-child(2) b').allInnerTexts();
  expect(cells.map(parseFloat).every((v) => Number.isFinite(v))).toBe(true);
  expect(errors).toEqual([]);
});

// Loop sensing study (2026-09-25): at 100² the twin comes from the learned U-Net.
test('closed loop uses the learned room estimate at 100²', async ({ page }) => {
  test.setTimeout(240_000);
  const errors = await open(page, '#/loop');
  await page.getByLabel('Grid size').selectOption('100');
  await page.getByLabel('Room estimate').selectOption('learned');
  await page.getByTestId('run-loop').click();
  await expect(page.getByTestId('loop-estimator')).toHaveAttribute('data-estimator', 'learned', { timeout: 60_000 });
  const first = page.getByTestId('loop-results').locator('tbody tr').first();
  await expect(first).toBeVisible({ timeout: 200_000 });
  await expect(first.locator('td:nth-child(7)')).toContainText('learned U-Net');
  expect(Number.isFinite(parseFloat(await first.locator('td:nth-child(2) b').innerText()))).toBe(true);
  expect(errors).toEqual([]);
});

// B05 / declutter: on a phone the controls wrap, the run button is on screen,
// and the results table scrolls inside its own box instead of the page.
test.describe('phone', () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  test('loop fits at 390 px', async ({ page }) => {
    test.setTimeout(240_000);
    const errors = await open(page, '#/loop');
    const run = page.getByTestId('run-loop');
    await expect(run).toBeVisible();
    const box = (await run.boundingBox())!;
    expect(box.x + box.width).toBeLessThanOrEqual(390);
    for (const label of ['Grid size', 'Room estimate']) {
      const b = (await page.getByLabel(label).boundingBox())!;
      expect(b.x + b.width).toBeLessThanOrEqual(390);
    }
    // The long explanation is folded away.
    await expect(page.getByText('How it works')).toBeVisible();
    await expect(page.getByText('Five monitor mics per zone')).toBeHidden();
    await page.getByLabel('Grid size').selectOption('60');
    await run.click();
    await expect(page.getByTestId('loop-results').locator('tbody tr').first()).toBeVisible({ timeout: 200_000 });
    const overflow = await page.locator('.page').evaluate((el) => el.scrollWidth - el.clientWidth);
    expect(overflow).toBeLessThanOrEqual(1);
    expect(await page.locator('.table-scroll').evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);
    expect(errors).toEqual([]);
  });
});

// B26: the fallback note describes the run, not the select changed afterwards.
test('estimator note follows the last run, not the select', async ({ page }) => {
  test.setTimeout(240_000);
  await open(page, '#/loop');
  await page.getByLabel('Grid size').selectOption('60');
  await page.getByLabel('Room estimate').selectOption('backprojection');
  await page.getByTestId('run-loop').click();
  await expect(page.getByTestId('loop-estimator')).toHaveAttribute('data-estimator', 'backprojection', { timeout: 60_000 });
  await page.getByTestId('loop-results').locator('tbody tr').first().waitFor({ timeout: 200_000 });
  await expect(page.getByTestId('run-loop')).toBeEnabled({ timeout: 200_000 });
  await page.getByLabel('Room estimate').selectOption('learned');
  await expect(page.getByTestId('loop-estimator')).not.toContainText('trained for the 100² grid');
});
