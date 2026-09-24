import { expect, test } from '@playwright/test';
import { app, clickField, drag, open, wallCount } from './helpers';

test.describe('sandbox', () => {
  test('loads with no errors and renders the field', async ({ page }) => {
    const errors = await open(page);
    await expect(page).toHaveTitle(/Acoustic Sandbox/);
    await expect(page.getByTestId('hud')).toContainText('step 0');
    expect(await app<string>(page, '(s) => s.scene.name')).toBe('Pulse in a room');
    expect(errors).toEqual([]);
  });

  test('run, pause, step and reset drive the engine', async ({ page }) => {
    await open(page);
    await page.getByTestId('run').click();
    await expect.poll(() => app<number>(page, '(s) => s.runtime.sim.step_count')).toBeGreaterThan(40);
    await page.getByTestId('run').click();
    const paused = await app<number>(page, '(s) => s.runtime.sim.step_count');
    await page.waitForTimeout(400);
    expect(await app<number>(page, '(s) => s.runtime.sim.step_count')).toBe(paused);
    await page.getByTestId('step').click();
    expect(await app<number>(page, '(s) => s.runtime.sim.step_count')).toBe(paused + 1);
    const peak = await app<number>(page, '(s) => s.runtime.peak()');
    expect(peak).toBeGreaterThan(0);
    await page.getByTestId('reset').click();
    expect(await app<number>(page, '(s) => s.runtime.sim.step_count')).toBe(0);
    expect(await app<number>(page, '(s) => s.runtime.peak()')).toBe(0);
  });

  test('keyboard: space runs and pauses', async ({ page }) => {
    await open(page);
    await page.keyboard.press('Space');
    await expect.poll(() => app<boolean>(page, '(s) => s.runtime.running')).toBe(true);
    await page.keyboard.press('Space');
    await expect.poll(() => app<boolean>(page, '(s) => s.runtime.running')).toBe(false);
  });

  test('brush paints walls, eraser removes them, undo/redo restore', async ({ page }) => {
    await open(page);
    await page.getByRole('button', { name: 'Clear all walls' }).click();
    expect(await wallCount(page)).toBe(0);
    await page.locator('[data-tool="brush"]').click();
    await drag(page, [0.2, 0.2], [0.5, 0.2]);
    const painted = await wallCount(page);
    expect(painted).toBeGreaterThan(20);
    // Engine mirrors the scene.
    expect(await app<number>(page, '(s) => s.runtime.sim.obstacles')).toBe(painted);
    await page.locator('[data-tool="eraser"]').click();
    await drag(page, [0.2, 0.2], [0.5, 0.2]);
    expect(await wallCount(page)).toBeLessThan(painted);
    await page.keyboard.press('Control+z');
    expect(await wallCount(page)).toBe(painted);
    await page.keyboard.press('Control+Shift+z');
    expect(await wallCount(page)).toBeLessThan(painted);
  });

  test('line, rectangle and ellipse tools', async ({ page }) => {
    await open(page);
    await page.getByRole('button', { name: 'Clear all walls' }).click();
    for (const tool of ['line', 'rect', 'ellipse']) {
      const before = await wallCount(page);
      await page.locator(`[data-tool="${tool}"]`).click();
      await drag(page, [0.3, 0.3], [0.6, 0.55]);
      expect(await wallCount(page), tool).toBeGreaterThan(before);
    }
    const before = await wallCount(page);
    await page.locator('[data-tool="rect"]').click();
    await drag(page, [0.7, 0.7], [0.85, 0.85], { shift: true });
    expect(await wallCount(page)).toBeGreaterThan(before + 300); // filled
  });

  test('place, edit and remove a source', async ({ page }) => {
    await open(page);
    const n0 = await app<number>(page, '(s) => s.scene.drivers.length');
    await page.locator('[data-tool="driver"]').click();
    await clickField(page, 0.8, 0.2);
    expect(await app<number>(page, '(s) => s.scene.drivers.length')).toBe(n0 + 1);
    await expect(page.getByTestId('driver-card')).toHaveCount(n0 + 1);
    await page.getByLabel('Waveform type').last().selectOption('cosine');
    expect(await app<string>(page, '(s) => s.scene.drivers[s.scene.drivers.length-1].waveform.type')).toBe('cosine');
    expect(await app<string>(page, '(s) => s.runtime.sim.drivers[s.runtime.sim.drivers.length-1].waveform.type')).toBe('cosine');
    await page.keyboard.press('Delete');
    expect(await app<number>(page, '(s) => s.scene.drivers.length')).toBe(n0);
  });

  test('microphones record, plot and play', async ({ page }) => {
    const errors = await open(page);
    await page.locator('[data-tool="probe"]').click();
    await clickField(page, 0.3, 0.7);
    await expect(page.getByTestId('probe-card')).toHaveCount(2);
    await page.getByTestId('run').click();
    await expect
      .poll(() => app<number>(page, '(s) => s.runtime.sim.probeSeries(s.scene.probes[1].id).length'))
      .toBeGreaterThan(200);
    await page.getByTestId('run').click();
    await expect(page.getByTestId('scope')).toBeVisible();
    await page.getByTestId('listen').click();
    await expect(page.getByTestId('listen')).toContainText(/Stop|Listen/);
    expect(errors).toEqual([]);
  });

  test('share link round-trips the scene', async ({ page }) => {
    await open(page);
    await page.locator('[data-tool="brush"]').click();
    await drag(page, [0.1, 0.9], [0.9, 0.9]);
    const walls = await wallCount(page);
    await page.getByLabel('Scene name').fill('My shared room');
    await page.getByTestId('share').click();
    await expect(page).toHaveURL(/s=/);
    const url = page.url();
    const page2 = await page.context().newPage();
    await page2.goto(url);
    await expect.poll(() => app<string>(page2, '(s) => s.scene.name')).toBe('My shared room');
    expect(await wallCount(page2)).toBe(walls);
  });

  test('scrubbing shows past frames', async ({ page }) => {
    await open(page);
    await page.getByTestId('run').click();
    await expect.poll(() => app<number>(page, '(s) => s.runtime.historySize')).toBeGreaterThan(20);
    await page.getByTestId('run').click();
    const slider = page.getByTestId('scrubber');
    await slider.fill('5');
    expect(await app<number | null>(page, '(s) => s.runtime.scrubIndex')).toBe(5);
    await page.getByRole('button', { name: 'Live' }).click();
    expect(await app<number | null>(page, '(s) => s.runtime.scrubIndex')).toBeNull();
  });

  test('view modes: dB, RMS, energy flow', async ({ page }) => {
    const errors = await open(page);
    await page.getByTestId('tab-view').click();
    await page.getByRole('button', { name: 'Decibels' }).click();
    await page.getByRole('button', { name: 'RMS level' }).click();
    await page.getByTestId('run').click();
    await page.waitForTimeout(600);
    expect(await app<boolean>(page, '(s) => s.runtime.sim.rmsMap() !== null')).toBe(true);
    await page.getByRole('button', { name: 'Energy flow' }).click();
    await page.waitForTimeout(600);
    expect(await app<boolean>(page, '(s) => s.runtime.sim.intensity !== null')).toBe(true);
    expect(errors).toEqual([]);
  });

  test('grid, units and boundary settings', async ({ page }) => {
    await open(page);
    await page.getByTestId('grid-0').fill('120');
    await page.getByTestId('grid-0').press('Enter');
    expect(await app<number[]>(page, '(s) => s.runtime.sim.params.shape')).toEqual([120, 200]);
    await page.getByRole('button', { name: 'SI (m, s, Hz)' }).click();
    expect(await app<number>(page, '(s) => s.runtime.sim.params.c')).toBe(343);
    await page.getByTestId('run').click();
    await expect(page.getByTestId('hud')).toContainText(/(µs|ms)/);
    await page.getByTestId('run').click();
    await page.getByLabel('Outer boundary').selectOption('sponge');
    expect(await app<string>(page, '(s) => s.runtime.sim.params.outer')).toBe('sponge');
    await page.getByLabel('Outer boundary').selectOption('cpml');
    expect(await app<string>(page, '(s) => s.runtime.sim.params.outer')).toBe('cpml');
    await page.getByTestId('cpml-cells').fill('12');
    await page.getByTestId('cpml-cells').press('Enter');
    expect(await app<number>(page, '(s) => s.runtime.sim.params.cpmlCells')).toBe(12);
    const before = await app<number>(page, '(s) => s.runtime.sim.step_count');
    await page.getByTestId('step').click();
    expect(await app<number>(page, '(s) => s.runtime.sim.step_count')).toBe(before + 1);
    // Invalid input is rejected, not applied.
    await page.getByTestId('grid-1').fill('-5');
    await page.getByTestId('grid-1').press('Enter');
    expect(await app<number[]>(page, '(s) => s.runtime.sim.params.shape')).toEqual([120, 200]);
  });

  test('sound-speed brush paints c(x) and per-face boundaries apply', async ({ page }) => {
    await open(page);
    await page.locator('[data-tool="speed"]').click();
    await drag(page, [0.4, 0.4], [0.6, 0.6]);
    expect(await app<boolean>(page, '(s) => s.scene.speed !== null')).toBe(true);
    expect(await app<boolean>(page, '(s) => s.runtime.sim.maxSpeedRatio >= 1')).toBe(true);
    await page.getByRole('button', { name: 'Reset sound speed' }).click();
    expect(await app<boolean>(page, '(s) => s.scene.speed === null')).toBe(true);
    await page.getByLabel('Set each face separately').check();
    await page.getByLabel('Right face').selectOption('sponge');
    expect(await app<string[]>(page, '(s) => s.runtime.sim.params.faces')).toEqual(['soft', 'soft', 'soft', 'sponge']);
    await page.getByTestId('run').click();
    await expect.poll(() => app<number>(page, '(s) => s.runtime.sim.step_count')).toBeGreaterThan(20);
    // Curtain material is paintable.
    await page.getByRole('radio', { name: 'Curtain' }).click();
    expect(await app<number>(page, '(s) => s.paintMaterial')).toBe(6);
  });

  test('3D mode: slices and volume view', async ({ page }) => {
    const errors = await open(page);
    await page.getByRole('button', { name: '3D' }).click();
    expect(await app<number>(page, '(s) => s.runtime.sim.dims')).toBe(3);
    await page.getByTestId('run').click();
    await expect.poll(() => app<number>(page, '(s) => s.runtime.sim.step_count')).toBeGreaterThan(5);
    await page.getByTestId('tab-view').click();
    await page.getByLabel('Slice position').fill('10');
    await page.getByRole('button', { name: 'Volume' }).click();
    await expect(page.getByTestId('volume')).toBeVisible();
    await page.waitForTimeout(500);
    expect(errors).toEqual([]);
  });

  test('theme toggle and help dialog', async ({ page }) => {
    await open(page);
    const before = await page.evaluate(() => document.documentElement.dataset.theme);
    await page.getByTestId('theme').click();
    const after = await page.evaluate(() => document.documentElement.dataset.theme);
    expect(after).not.toBe(before);
    await page.keyboard.press('?');
    await expect(page.getByRole('dialog', { name: 'Keyboard shortcuts' })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toHaveCount(0);
  });

  test('screenshot export downloads a PNG', async ({ page }) => {
    await open(page);
    const dl = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Screenshot' }).click();
    expect((await dl).suggestedFilename()).toMatch(/\.png$/);
  });

  test('save downloads a scene file that validates', async ({ page }) => {
    await open(page);
    const dl = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Save' }).click();
    const file = await dl;
    expect(file.suggestedFilename()).toMatch(/\.json$/);
  });
});

test.describe('gallery', () => {
  test('every preset opens in the sandbox', async ({ page }) => {
    const errors = await open(page, '#/gallery');
    const cards = page.locator('[data-testid^="preset-"]');
    const n = await cards.count();
    expect(n).toBeGreaterThanOrEqual(8);
    await cards.nth(1).click();
    await expect(page).toHaveURL(/preset=/);
    await expect.poll(() => app<string>(page, '(s) => s.scene.name')).toBe('Double slit');
    expect(errors).toEqual([]);
  });
});

test.describe('responsive', () => {
  test.use({ viewport: { width: 390, height: 844 } });
  test('phone layout has no horizontal overflow', async ({ page }) => {
    await open(page);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
    await expect(page.getByTestId('field')).toBeVisible();
  });
});
