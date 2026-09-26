import { expect, type Page, test } from '@playwright/test';
import { app, clickField, drag, open } from './helpers';

// Regression tests for the 2026-09-26 bug hunt (bug ids in the test names).

const pastLen = (page: Page) => app<number>(page, '(s) => s.past.length');

test.describe('sandbox shortcuts and focus', () => {
  test('B09: Shift+R resets the field', async ({ page }) => {
    await open(page);
    await page.getByTestId('step').click();
    await page.getByTestId('step').click();
    expect(await app<number>(page, '(s) => s.runtime.sim.step_count')).toBe(2);
    await page.locator('body').click({ position: { x: 5, y: 5 } });
    await page.keyboard.press('Shift+R');
    expect(await app<number>(page, '(s) => s.runtime.sim.step_count')).toBe(0);
    // Plain R still selects the rectangle tool.
    await page.keyboard.press('r');
    expect(await app<string>(page, '(s) => s.tool')).toBe('rect');
  });

  test('B10: Space on a focused button activates it instead of running the simulation', async ({ page }) => {
    await open(page);
    await page.getByTestId('tab-view').focus();
    await page.keyboard.press('Space');
    expect(await app<string>(page, '(s) => s.inspectorTab')).toBe('view');
    expect(await app<boolean>(page, '(s) => s.runtime.running')).toBe(false);
  });

  test('B37/B38: the help dialog takes and traps focus, and lists every tool key', async ({ page }) => {
    await open(page);
    await page.keyboard.press('?');
    const dialog = page.getByTestId('help-dialog');
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText('Sound-speed brush');
    await expect(dialog).toContainText('control zone');
    expect(await page.evaluate(() => !!document.activeElement?.closest('[role="dialog"]'))).toBe(true);
    for (let i = 0; i < 4; i++) {
      await page.keyboard.press('Tab');
      expect(await page.evaluate(() => !!document.activeElement?.closest('[role="dialog"]'))).toBe(true);
    }
    // Keys behind the modal do nothing.
    await page.keyboard.press('b');
    expect(await app<string>(page, '(s) => s.tool')).toBe('brush');
    await page.keyboard.press('v');
    expect(await app<string>(page, '(s) => s.tool')).toBe('brush');
    await page.keyboard.press('Escape');
    await expect(dialog).toHaveCount(0);
  });

  test('B38: a corrupt share link explains itself', async ({ page }) => {
    await open(page, '#/sandbox?s=garbage');
    await expect(page.getByTestId('toast')).toContainText(/shared scene: the link is incomplete or corrupted/);
  });
});

test.describe('undo history', () => {
  test('B11: switching units is one undo step', async ({ page }) => {
    await open(page);
    const n0 = await pastLen(page);
    await page.getByRole('button', { name: 'SI (m, s, Hz)' }).click();
    expect(await pastLen(page)).toBe(n0 + 1);
    await page.keyboard.press('Control+z');
    expect(await app<string>(page, '(s) => s.scene.units')).toBe('grid');
    expect(await app<number>(page, '(s) => s.runtime.sim.params.c')).toBe(1);
  });

  test('B12: renaming a microphone is one undo step', async ({ page }) => {
    await open(page);
    await page.getByTestId('tab-probes').click();
    const name = page.getByLabel('Microphone name').first();
    const before = await name.inputValue();
    const n0 = await pastLen(page);
    await name.fill('');
    await name.pressSequentially('Kitchen');
    expect(await pastLen(page)).toBe(n0 + 1);
    await page.getByRole('button', { name: 'Undo' }).click();
    await expect(name).toHaveValue(before);
  });

  test('B13: clicking a marker just to select it adds no undo step', async ({ page }) => {
    await open(page);
    await page.locator('[data-tool="select"]').click();
    const pos = await app<number[]>(page, '(s) => s.scene.drivers[0].pos');
    const shape = await app<number[]>(page, '(s) => s.scene.params.shape');
    const n0 = await pastLen(page);
    await clickField(page, (pos[1] + 0.5) / shape[1], (pos[0] + 0.5) / shape[0]);
    expect(await app<string | null>(page, '(s) => s.selected?.kind ?? null')).toBe('driver');
    expect(await pastLen(page)).toBe(n0);
  });

  test('B38: clearing walls keeps the painted sound speed', async ({ page }) => {
    await open(page);
    await page.locator('[data-tool="speed"]').click();
    await drag(page, [0.4, 0.4], [0.6, 0.6]);
    expect(await app<boolean>(page, '(s) => s.scene.speed !== null')).toBe(true);
    await page.getByRole('button', { name: 'Clear all walls' }).click();
    expect(await app<boolean>(page, '(s) => s.scene.speed !== null')).toBe(true);
  });
});

test.describe('scene loading', () => {
  test('B24: browser Back to a ?preset= URL keeps the edits', async ({ page }) => {
    await open(page, '#/gallery');
    await page.goto('./#/sandbox?preset=double-slit');
    await expect.poll(() => app<string>(page, '(s) => s.scene.name')).toBe('Double slit');
    const n0 = await app<number>(page, '(s) => s.scene.drivers.length');
    await page.locator('[data-tool="driver"]').click();
    await clickField(page, 0.8, 0.8);
    expect(await app<number>(page, '(s) => s.scene.drivers.length')).toBe(n0 + 1);
    await page.evaluate(() => (window.location.hash = '#/learn'));
    await expect(page.getByTestId('learn')).toBeVisible();
    await page.goBack();
    await expect(page.getByTestId('field')).toBeVisible();
    await page.waitForTimeout(300);
    expect(await app<number>(page, '(s) => s.scene.drivers.length')).toBe(n0 + 1);
  });

  test('B18: control zones are dropped when another room is loaded', async ({ page }) => {
    await open(page);
    await page.locator('[data-tool="zone"]').click();
    await drag(page, [0.6, 0.6], [0.95, 0.95]);
    expect(await app<unknown>(page, '(s) => s.zones.bright')).not.toBeNull();
    await page.getByTestId('tab-scene').click();
    await page.getByLabel('Load preset').selectOption('corridor');
    expect(await app<unknown>(page, '(s) => s.zones.bright')).toBeNull();
    await expect(page.getByTestId('zone-bright')).toHaveCount(0);
    // A grid resize drops them too.
    await page.locator('[data-tool="zone"]').click();
    await drag(page, [0.2, 0.2], [0.4, 0.4]);
    expect(await app<unknown>(page, '(s) => s.zones.bright')).not.toBeNull();
    await page.getByTestId('tab-scene').click();
    await page.getByTestId('grid-0').fill('64');
    await page.getByTestId('grid-0').press('Enter');
    expect(await app<unknown>(page, '(s) => s.zones.bright')).toBeNull();
  });
});

test.describe('view and dock', () => {
  test('B22: the colour legend follows the fixed scale', async ({ page }) => {
    await open(page);
    // A silent field has no scale yet.
    await expect(page.getByTestId('legend').locator('.ticks')).toContainText('—');
    await page.getByTestId('tab-view').click();
    await page.getByLabel('Auto-scale to the peak').uncheck();
    await page.getByLabel('Full-scale value').fill('0.01');
    await page.getByLabel('Full-scale value').press('Enter');
    await expect(page.getByTestId('legend').locator('.ticks')).toContainText('0.010');
  });

  test('B21: the spectrum spans 0 to Nyquist', async ({ page }) => {
    await open(page);
    await page.getByTestId('run').click();
    await expect.poll(() => app<number>(page, '(s) => s.runtime.sim.step_count')).toBeGreaterThan(200);
    await page.getByTestId('run').click();
    await page.getByRole('button', { name: 'Spectrum' }).click();
    const dt = await app<number>(page, '(s) => s.runtime.sim.dt');
    const label = await page.getByTestId('spectrum-max').innerText();
    expect(label).toContain('Nyquist');
    expect(parseFloat(label)).toBeCloseTo(1 / (2 * dt), 1);
  });
});

test.describe('record menu', () => {
  test('B29: the Record menu closes on Escape and on an outside click', async ({ page }) => {
    await open(page);
    await page.getByTestId('record-menu').click();
    await expect(page.getByTestId('record-popover')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('record-popover')).toHaveCount(0);
    await page.getByTestId('record-menu').click();
    await expect(page.getByTestId('record-popover')).toBeVisible();
    await clickField(page, 0.5, 0.5);
    await expect(page.getByTestId('record-popover')).toHaveCount(0);
  });
});

test.describe('phone', () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });

  test('B28: the Record menu opens on-screen', async ({ page }) => {
    await open(page);
    await page.getByTestId('record-menu').click();
    const box = await page.getByTestId('record-popover').boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(390);
    await expect(page.getByRole('menuitem', { name: /Animated GIF/ })).toBeInViewport();
  });

  test('B30: the 3D Volume sliders are reachable', async ({ page }) => {
    await open(page);
    await page.getByRole('button', { name: '3D' }).click();
    await page.getByTestId('tab-view').click();
    await page.getByRole('button', { name: 'Volume' }).click();
    await expect(page.getByTestId('volume-controls')).toBeVisible();
    const opacity = page.getByRole('slider', { name: 'Opacity', exact: true });
    await opacity.scrollIntoViewIfNeeded();
    await expect(opacity).toBeInViewport();
    const panel = (await page.getByTestId('volume-controls').boundingBox())!;
    expect(panel.x).toBeGreaterThanOrEqual(0);
    expect(panel.x + panel.width).toBeLessThanOrEqual(390);
    await page.screenshot({ path: test.info().outputPath('volume-phone.png') });
    await expect(page.getByRole('slider', { name: 'Wall opacity' })).toBeInViewport();
  });
});
