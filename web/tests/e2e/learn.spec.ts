import { expect, test } from '@playwright/test';
import { open } from './helpers';

// Plan 10.6: explainers with live embedded simulations.
test('explainers render maths and run live simulations', async ({ page }) => {
  const errors = await open(page, '#/learn');
  await expect(page.getByTestId('learn')).toBeVisible();
  const cards = page.locator('[data-testid^="article-"]');
  await expect(cards).toHaveCount(6);
  await page.getByTestId('article-walls').click();
  await expect(page.getByTestId('learn-article')).toContainText('Walls: soft, hard and absorbing');
  await expect(page.locator('.katex').first()).toBeVisible();
  const sim = page.getByTestId('livesim').first();
  await sim.getByTestId('livesim-play').click();
  await expect(sim.locator('.mono')).not.toHaveText('step 0', { timeout: 10_000 });
  await sim.getByRole('button', { name: 'Rigid' }).click();
  await expect(sim.getByRole('button', { name: 'Rigid' })).toHaveAttribute('aria-pressed', 'true');
  // Every article opens without errors.
  for (const id of ['fdtd', 'rooms', 'echolocation', 'control', 'diffraction']) {
    await page.goto(`./#/learn/${id}`);
    await expect(page.getByTestId('learn-article')).toBeVisible();
    await expect(page.getByTestId('livesim').first()).toBeVisible();
  }
  // Hand-off to the sandbox.
  await page.getByRole('button', { name: 'Open in sandbox' }).first().click();
  await expect(page).toHaveURL(/#\/sandbox\?preset=double-slit/);
  expect(errors).toEqual([]);
});

test('B03: auto-replay restarts the selected variant, not the first', async ({ page }) => {
  const errors = await open(page, '#/learn/walls');
  const sim = page.getByTestId('livesim').first();
  await sim.scrollIntoViewIfNeeded();
  await sim.getByTestId('livesim-play').click();
  await sim.getByRole('button', { name: 'No wall (PML)' }).click();
  await expect(sim).toHaveAttribute('data-walls', '0');
  await expect.poll(() => sim.getAttribute('data-replays'), { timeout: 45_000 }).not.toBe('0');
  await expect(sim).toHaveAttribute('data-walls', '0');
  await expect(sim).toHaveAttribute('data-variant', '3');
  expect(errors).toEqual([]);
});

test('B32: Next opens the article at the top', async ({ page }) => {
  await open(page, '#/learn/fdtd');
  await expect(page.getByTestId('livesim').first()).toBeVisible();
  const scroller = page.locator('.page');
  await expect.poll(() => scroller.evaluate((el) => ((el.scrollTop = el.scrollHeight), el.scrollTop))).toBeGreaterThan(100);
  await page.getByRole('link', { name: /Walls: soft, hard and absorbing/ }).click();
  await expect(page.getByTestId('learn-article')).toContainText('Walls: soft, hard and absorbing');
  await expect(page.getByRole('heading', { level: 1 })).toBeInViewport();
  expect(await scroller.evaluate((el) => el.scrollTop)).toBe(0);
});
