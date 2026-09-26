import { expect, type Page, test } from '@playwright/test';
import { open } from './helpers';

/** Record main-thread long tasks from page start (read with `longTasks`). */
async function watchLongTasks(page: Page) {
  await page.addInitScript(() => {
    const w = window as unknown as { __longTasks: [number, number][] };
    w.__longTasks = [];
    try {
      new PerformanceObserver((l) => {
        for (const e of l.getEntries()) w.__longTasks.push([e.startTime, e.duration]);
      }).observe({ type: 'longtask', buffered: true });
    } catch {
      /* not supported */
    }
  });
}
const now = (page: Page) => page.evaluate(() => performance.now());
const longTasks = (page: Page, since: number) =>
  page.evaluate((t) => (window as unknown as { __longTasks: [number, number][] }).__longTasks.filter(([s]) => s >= t).map(([, d]) => Math.round(d)), since);

/** Listen and skip straight to the answer. */
async function listen(page: Page) {
  await page.getByTestId('echo-listen').click();
  await expect(page.getByTestId('echo')).toHaveAttribute('data-phase', /listening|analysing/);
  await page.getByTestId('echo-skip').click();
  await expect(page.getByTestId('echo-result')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('echo')).toHaveAttribute('data-phase', 'done');
}

const ious = async (page: Page) => {
  const r = page.getByTestId('echo-result');
  return { learned: Number(await r.getAttribute('data-iou-learned')), bp: Number(await r.getAttribute('data-iou-bp')) };
};

// Echo vision: the loop's U-Net reconstructs a room from the bar's echoes, next to back-projection.
test('echo vision: the story plays through its three stages, and the network beats the raw echo image', async ({ page }) => {
  test.setTimeout(120_000);
  await watchLongTasks(page);
  const errors = await open(page, '#/echo');
  await expect(page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Echo vision' })).toHaveAttribute('aria-current', 'page');
  await page.getByTestId('echo-example-door').click();
  await expect(page.getByTestId('echo-example-door')).toHaveAttribute('aria-pressed', 'true');
  const t0 = await now(page);
  await page.getByTestId('echo-listen').click();
  const player = page.getByTestId('echo-player');
  // 1 Clicks and echoes: the first ping plays with its echo timeline under the room.
  await expect(player).toHaveAttribute('data-stage', '1');
  await expect(page.getByTestId('echo-progress')).toContainText(/Ping 1 of 8|Getting ready|Still listening/, { timeout: 30_000 });
  await expect(page.getByTestId('echo-progress')).toContainText('Ping 1 of 8', { timeout: 30_000 });
  for (const id of ['echo-field', 'echo-timeline', 'echo-picture', 'echo-scrub']) await expect(page.getByTestId(id)).toBeVisible();
  await expect(page.getByTestId('echo-stages').locator('li[data-state="active"]')).toContainText('Clicks and echoes');
  // 2 Tracing echoes back: the picture builds up ping by ping.
  await expect(player).toHaveAttribute('data-stage', '2', { timeout: 30_000 });
  await expect(page.getByTestId('echo-picture-panel')).toContainText(/traced back: \d of 8 pings/);
  await expect(player).toHaveAttribute('data-ping', '2', { timeout: 30_000 });
  // The answer stays back until the story gets there.
  await expect(page.getByTestId('echo-result')).toHaveCount(0);
  // 3 The network's guess, then the results.
  await expect(player).toHaveAttribute('data-stage', '3', { timeout: 60_000 });
  await expect(page.getByTestId('echo-result')).toBeVisible();
  await expect(page.getByTestId('echo')).toHaveAttribute('data-phase', 'done');
  await expect(page.getByTestId('echo-player-verdict')).toContainText('The network recovered');
  await expect(player).toHaveAttribute('data-state', 'ended', { timeout: 10_000 });
  // No main-thread task during the playback took more than 200 ms.
  const lt = await longTasks(page, t0);
  expect(lt.filter((d) => d > 200), `long tasks ${lt}`).toEqual([]);
  const { learned, bp } = await ious(page);
  expect(learned).toBeGreaterThan(0.6);
  expect(learned).toBeGreaterThan(bp);
  await expect(page.getByTestId('echo-iou-learned')).toHaveText(`IoU ${learned.toFixed(2)}`);
  await expect(page.getByTestId('echo-player-iou')).toHaveText(`IoU ${learned.toFixed(2)}`);
  await expect(page.getByTestId('echo-iou-bp')).toHaveText(`IoU ${bp.toFixed(2)}`);
  await expect(page.getByTestId('echo-verdict')).toContainText(new RegExp(`${Math.round(100 * learned)}\\s%`));
  await expect(page.getByTestId('echo-result')).toHaveAttribute('data-in-family', 'true');
  await expect(page.getByTestId('echo-family-note')).toHaveCount(0);
  for (const id of ['echo-truth', 'echo-image', 'echo-network']) await expect(page.getByTestId(id)).toBeVisible();
  // Editing the room clears the stale answer and brings the room back.
  await page.getByTestId('echo-random').click();
  await expect(page.getByTestId('echo-result')).toHaveCount(0);
  await expect(page.getByTestId('echo-player')).toHaveCount(0);
  await expect(page.getByTestId('echo-room')).toBeVisible();
  expect(errors).toEqual([]);
});

test('echo vision: play, pause, scrub, speed and skip', async ({ page }) => {
  test.setTimeout(120_000);
  const errors = await open(page, '#/echo');
  await page.getByTestId('echo-example-box').click();
  await page.getByTestId('echo-listen').click();
  const player = page.getByTestId('echo-player');
  const scrub = page.getByTestId('echo-scrub');
  await expect(player).toHaveAttribute('data-state', 'playing', { timeout: 30_000 });
  // Pause holds the playhead.
  await page.getByTestId('echo-play').click();
  await expect(player).toHaveAttribute('data-state', 'paused');
  const v = await scrub.inputValue();
  await page.waitForTimeout(600);
  expect(await scrub.inputValue()).toBe(v);
  // Speed is a choice of three.
  await page.getByTestId('echo-speed-2').click();
  await expect(page.getByTestId('echo-speed-2')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('echo-speed-1')).toHaveAttribute('aria-pressed', 'false');
  // Play resumes and moves the playhead.
  await page.getByTestId('echo-play').click();
  await expect(player).toHaveAttribute('data-state', /playing|waiting/);
  await expect.poll(async () => Number(await scrub.inputValue()), { timeout: 20_000 }).toBeGreaterThan(Number(v));
  // Skip goes to the answer as soon as it is computed.
  await page.getByTestId('echo-skip').click();
  await expect(player).toHaveAttribute('data-stage', '3', { timeout: 60_000 });
  await expect(player).toHaveAttribute('data-state', 'ended');
  await expect(page.getByTestId('echo-result')).toBeVisible();
  // Scrubbing back to the start shows the first ping again; the answer stays.
  await scrub.fill('0');
  await expect(player).toHaveAttribute('data-stage', '1');
  await expect(player).toHaveAttribute('data-ping', '1');
  await expect(page.getByTestId('echo-progress')).toContainText('Ping 1 of 8');
  await expect(page.getByTestId('echo-result')).toBeVisible();
  // Scrub past the middle: a later ping, played or traced.
  await scrub.fill('550');
  await expect(player).toHaveAttribute('data-stage', /1|2/);
  expect(Number(await player.getAttribute('data-ping'))).toBeGreaterThan(2);
  // Replay from the end.
  await page.getByTestId('echo-skip').click();
  await expect(player).toHaveAttribute('data-state', 'ended');
  await expect(page.getByTestId('echo-play')).toHaveAttribute('aria-label', 'Replay');
  await page.getByTestId('echo-play').click();
  await expect(player).toHaveAttribute('data-stage', '1');
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
  // Picking a drawing tool goes back to the room editor.
  await page.getByTestId('echo-tool-wall').click();
  await expect(page.getByTestId('echo-room')).toBeVisible();
  await expect(page.getByTestId('echo-result')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('echo vision: reduced motion shows a static summary', async ({ page }) => {
  test.setTimeout(120_000);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const errors = await open(page, '#/echo');
  await page.getByTestId('echo-example-box').click();
  await page.getByTestId('echo-listen').click();
  const player = page.getByTestId('echo-player');
  // Nothing plays by itself: once computed, the page shows the end state.
  await expect(player).toHaveAttribute('data-state', 'paused');
  await expect(page.getByTestId('echo-result')).toBeVisible({ timeout: 60_000 });
  await expect(player).toHaveAttribute('data-stage', '3');
  await expect(player).toHaveAttribute('data-state', 'ended');
  await expect(page.getByTestId('echo-player-verdict')).toBeVisible();
  // Playing is still on offer.
  await page.getByTestId('echo-play').click();
  await expect(player).toHaveAttribute('data-stage', '1');
  expect(errors).toEqual([]);
});

test('echo vision fits a phone screen', async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 390, height: 844 });
  await watchLongTasks(page);
  const errors = await open(page, '#/echo');
  const overflow = () => page.evaluate(() => [document.documentElement.scrollWidth, document.querySelector('.page')!.scrollWidth]);
  for (const w of await overflow()) expect(w).toBeLessThanOrEqual(390);
  await page.getByTestId('echo-example-box').click();
  const t0 = await now(page);
  await page.getByTestId('echo-listen').click();
  const player = page.getByTestId('echo-player');
  await expect(player).toHaveAttribute('data-state', 'playing', { timeout: 30_000 });
  for (const w of await overflow()) expect(w).toBeLessThanOrEqual(390);
  // The room, its echo timeline and the controls are on one screen, on one row of controls.
  const field = (await page.getByTestId('echo-field').boundingBox())!;
  expect(field.width).toBeGreaterThan(330);
  const play = (await page.getByTestId('echo-play').boundingBox())!;
  const skip = (await page.getByTestId('echo-skip').boundingBox())!;
  expect(Math.abs(play.y - skip.y)).toBeLessThan(4);
  expect(skip.x + skip.width).toBeLessThanOrEqual(390);
  expect(play.y + play.height).toBeLessThanOrEqual(844);
  await expect(player).toHaveAttribute('data-stage', '3', { timeout: 60_000 });
  await expect(page.getByTestId('echo')).toHaveAttribute('data-phase', 'done');
  for (const w of await overflow()) expect(w).toBeLessThanOrEqual(390);
  const lt = await longTasks(page, t0);
  expect(lt.filter((d) => d > 200), `long tasks ${lt}`).toEqual([]);
  // Stacked: each result panel spans (nearly) the full width.
  const b = (await page.getByTestId('echo-network').boundingBox())!;
  expect(b.width).toBeGreaterThan(300);
  expect(errors).toEqual([]);
});
