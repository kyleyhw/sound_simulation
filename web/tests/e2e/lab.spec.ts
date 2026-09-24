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
});
