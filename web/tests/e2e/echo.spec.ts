import { expect, type Page, test } from '@playwright/test';
import { open } from './helpers';

async function listen(page: Page) {
  await page.getByTestId('echo-listen').click();
  await expect(page.getByTestId('echo')).toHaveAttribute('data-phase', /listening|analysing/);
  await expect(page.getByTestId('echo-result')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('echo')).toHaveAttribute('data-phase', 'done');
}

const ious = async (page: Page) => {
  const r = page.getByTestId('echo-result');
  return { learned: Number(await r.getAttribute('data-iou-learned')), bp: Number(await r.getAttribute('data-iou-bp')) };
};

// Echo vision: the loop's U-Net reconstructs a room from the bar's echoes, next to back-projection.
test('echo vision: the network beats the raw echo image on a training-family room', async ({ page }) => {
  test.setTimeout(120_000);
  const errors = await open(page, '#/echo');
  await expect(page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Echo vision' })).toHaveAttribute('aria-current', 'page');
  await page.getByTestId('echo-example-door').click();
  await expect(page.getByTestId('echo-example-door')).toHaveAttribute('aria-pressed', 'true');
  await page.getByTestId('echo-listen').click();
  // The pings stream the live field and a progress line while the worker runs.
  await expect(page.getByTestId('echo-progress')).toContainText(/Ping \d of 8|Getting ready|Turning the echoes/, { timeout: 30_000 });
  await expect(page.getByTestId('echo-result')).toBeVisible({ timeout: 60_000 });
  const { learned, bp } = await ious(page);
  expect(learned).toBeGreaterThan(0.6);
  expect(learned).toBeGreaterThan(bp);
  await expect(page.getByTestId('echo-iou-learned')).toHaveText(`IoU ${learned.toFixed(2)}`);
  await expect(page.getByTestId('echo-iou-bp')).toHaveText(`IoU ${bp.toFixed(2)}`);
  await expect(page.getByTestId('echo-verdict')).toContainText(new RegExp(`${Math.round(100 * learned)}\\s%`));
  await expect(page.getByTestId('echo-result')).toHaveAttribute('data-in-family', 'true');
  await expect(page.getByTestId('echo-family-note')).toHaveCount(0);
  for (const id of ['echo-truth', 'echo-image', 'echo-network']) await expect(page.getByTestId(id)).toBeVisible();
  // Editing the room clears the stale answer.
  await page.getByTestId('echo-random').click();
  await expect(page.getByTestId('echo-result')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('echo vision: drawing, hiding, and the out-of-family note', async ({ page }) => {
  test.setTimeout(120_000);
  const errors = await open(page, '#/echo');
  // Draw a block by dragging on the room.
  await page.getByTestId('echo-tool-block').click();
  const box = (await page.getByTestId('echo-room').boundingBox())!;
  const at = (r: number, c: number) => [box.x + ((c + 0.5) / 100) * box.width, box.y + ((r + 0.5) / 100) * box.height] as const;
  await page.getByTestId('echo-example-pillar').click();
  await page.mouse.move(...at(20, 20));
  await page.mouse.down();
  await page.mouse.move(...at(26, 26), { steps: 4 });
  await page.mouse.move(...at(28, 32), { steps: 4 });
  await page.mouse.up();
  await expect(page.getByTestId('echo-example-pillar')).toHaveAttribute('aria-pressed', 'false');
  // Hidden rooms keep the truth back until revealed.
  await page.getByTestId('echo-hide').click();
  await listen(page);
  await expect(page.getByTestId('echo-truth')).toHaveCount(0);
  await expect(page.getByTestId('echo-result')).toHaveAttribute('data-in-family', 'false');
  await expect(page.getByTestId('echo-family-note')).toContainText('it draws boxes');
  await page.getByTestId('echo-reveal').click();
  await expect(page.getByTestId('echo-truth')).toBeVisible();
  expect(errors).toEqual([]);
});

test('echo vision fits a phone screen', async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 390, height: 844 });
  const errors = await open(page, '#/echo');
  const overflow = () => page.evaluate(() => [document.documentElement.scrollWidth, document.querySelector('.page')!.scrollWidth]);
  for (const w of await overflow()) expect(w).toBeLessThanOrEqual(390);
  await page.getByTestId('echo-example-box').click();
  await listen(page);
  for (const w of await overflow()) expect(w).toBeLessThanOrEqual(390);
  // Stacked: each result panel spans (nearly) the full width.
  const b = (await page.getByTestId('echo-network').boundingBox())!;
  expect(b.width).toBeGreaterThan(300);
  expect(errors).toEqual([]);
});
