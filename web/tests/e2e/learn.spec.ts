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
