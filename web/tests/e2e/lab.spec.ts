import { expect, test } from '@playwright/test';
import { open } from './helpers';

// Chromium runs with a fake microphone and camera (see playwright.config.ts),
// so the full measure pipeline runs end to end without real hardware.
test.describe('lab', () => {
  test('measure, save a capture, run the room twin and show CTC separation', async ({ page }) => {
    const errors = await open(page, '#/lab');
    await expect(page.getByTestId('lab')).toBeVisible();
    await page.getByLabel('Sweep (s)').fill('0.5');
    await page.getByLabel('Sweep (s)').press('Enter');
    await page.getByTestId('measure').click();
    await expect(page.getByTestId('measure-results')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId('measure-results')).toContainText('dBFS');

    await expect(page.getByTestId('measure-results')).toContainText('round-trip latency');

    // Device calibration, then an equalised re-measurement.
    await page.getByTestId('calibrate').click();
    await expect(page.getByLabel('Equalise with device calibration')).toBeChecked();
    await page.getByTestId('measure').click();
    await expect(page.getByTestId('measure-results')).toContainText('equalised', { timeout: 30_000 });

    await page.getByLabel('Capture label').fill('e2e fake device');
    await page.getByTestId('save-capture').click();
    await expect(page.locator('#captures')).toContainText('e2e fake device');

    await page.getByTestId('run-twin').click();
    await expect(page.getByTestId('run-twin')).toBeEnabled({ timeout: 45_000 });
    await expect(page.locator('#twin')).toContainText(/cells\)/);

    // Separation plot title reports the plain-stereo baseline.
    await expect(page.locator('#headphones')).toContainText('plain stereo');
    expect(errors).toEqual([]);
  });

  test('B01/S1: virtual headphones play at the device sample rate and stop on leaving the page', async ({ page }) => {
    const errors = await open(page, '#/lab');
    await page.getByTestId('ctc-play').click();
    await expect(page.getByTestId('ctc-play')).toContainText('Stop');
    await page.getByLabel('Crosstalk cancellation').uncheck();
    await page.getByLabel('Head x').fill('0.1');
    await page.getByLabel('Crosstalk cancellation').check();
    await page.getByTestId('ctc-play').click();
    await expect(page.getByTestId('ctc-play')).toContainText('Play test');
    await page.getByTestId('ctc-play').click();
    await expect(page.getByTestId('ctc-play')).toContainText('Stop');
    await page.evaluate(() => (window.location.hash = '#/learn'));
    await expect(page.getByTestId('learn')).toBeVisible();
    expect(errors).toEqual([]);
  });

  test('B02/B14/B15: decimal ground truth, and importing the same capture file twice', async ({ page }) => {
    const errors = await open(page, '#/lab');
    await page.getByLabel('Sweep (s)').fill('0.3');
    await page.getByLabel('Sweep (s)').press('Enter');
    await page.getByTestId('measure').click();
    await expect(page.getByTestId('measure-results')).toBeVisible({ timeout: 30_000 });
    const dist = page.getByLabel('Measured distance');
    await dist.pressSequentially('1.25');
    await expect(dist).toHaveValue('1.25');
    await dist.fill('abc');
    await expect(page.getByTestId('save-capture')).toBeDisabled();
    await dist.fill('1.25');
    await page.getByLabel('Capture label').fill('decimal');
    await page.getByTestId('save-capture').click();
    const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('lab-captures-v1') ?? '[]') as { measuredDistance?: number }[]);
    expect(saved.at(-1)?.measuredDistance).toBe(1.25);
    await expect(page.locator('#captures tbody tr')).toHaveCount(1);

    const dl = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export JSON' }).click();
    const file = await (await dl).path();
    const input = page.getByTestId('import-captures');
    await input.setInputFiles(file);
    await expect(page.locator('#captures tbody tr')).toHaveCount(2);
    await input.setInputFiles(file); // the same file again still imports
    await expect(page.locator('#captures tbody tr')).toHaveCount(3);
    await page.getByRole('button', { name: 'Delete capture' }).first().click();
    await expect(page.locator('#captures tbody tr')).toHaveCount(2);
    expect(errors).toEqual([]);
  });

  test('B34/B35: impossible geometry is rejected, and the twin runs off the main thread', async ({ page }) => {
    const errors = await open(page, '#/lab');
    const x = page.getByLabel('Laptop x (m)');
    await x.fill('10');
    await x.press('Enter');
    await expect(page.locator('#twin')).toContainText('Between 0.05 and 4.95');
    await expect(page.locator('#twin')).not.toContainText('-5.00');
    // Shrinking the room below the laptop position is caught too.
    const len = page.getByLabel('Length (m)');
    await len.fill('1');
    await len.press('Enter');
    await expect(page.getByTestId('twin-geometry-error')).toContainText('outside the room');
    await expect(page.getByTestId('run-twin')).toBeDisabled();
    await len.fill('5');
    await len.press('Enter');
    await expect(page.getByTestId('twin-geometry-error')).toHaveCount(0);

    await page.evaluate(() => {
      const w = window as unknown as { __long: number[] };
      w.__long = [];
      new PerformanceObserver((l) => l.getEntries().forEach((e) => w.__long.push(e.duration))).observe({ type: 'longtask', buffered: false });
    });
    await page.getByTestId('run-twin').click();
    await expect(page.getByTestId('run-twin')).toBeEnabled({ timeout: 60_000 });
    await expect(page.locator('#twin')).toContainText(/cells\)/);
    const long = await page.evaluate(() => (window as unknown as { __long: number[] }).__long);
    // On the main thread this was ~10 s of 400-550 ms tasks.
    expect(long.reduce((a, b) => a + b, 0)).toBeLessThan(1500);
    expect(errors).toEqual([]);
  });
});

test.describe('lab on a phone', () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });
  test('B27: the measurement row wraps instead of clipping its controls', async ({ page }) => {
    await open(page, '#/lab');
    const card = await page.locator('#measure').boundingBox();
    for (const loc of [page.getByTestId('measure'), page.getByLabel('End (Hz)'), page.getByLabel('Output speaker'), page.getByTestId('track')]) {
      const b = await loc.boundingBox();
      expect(b).not.toBeNull();
      expect(b!.x + b!.width).toBeLessThanOrEqual(card!.x + card!.width + 1);
      // Not squeezed: the content fits its box.
      expect(await loc.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
    }
    expect((await page.getByLabel('Output speaker').boundingBox())!.width).toBeGreaterThan(80);
    expect(await page.getByLabel('End (Hz)').inputValue()).toBe('16000');
    expect((await page.getByLabel('End (Hz)').boundingBox())!.width).toBeGreaterThan(60);
  });
});
