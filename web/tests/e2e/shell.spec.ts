import { expect, type Page, test } from '@playwright/test';
import { app, drag, open, openAdvanced } from './helpers';

// Declutter (2026-09-26): home page, navigation, routing, gallery thumbnails,
// and the sandbox and page fixes that came with it.

const SECTIONS = [
  { name: 'Echo vision', url: /#\/echo$/ },
  { name: 'Sandbox', url: /#\/sandbox$/ },
  { name: 'Gallery', url: /#\/gallery$/ },
  { name: 'Learn', url: /#\/learn$/ },
];
const MORE = [
  { name: 'Research', url: /#\/research$/ },
  { name: 'Loop', url: /#\/loop$/ },
  { name: 'Lab', url: /#\/lab$/ },
  { name: 'Docs', url: /#\/docs$/ },
];

async function inViewport(page: Page, loc: ReturnType<Page['locator']>) {
  await expect(loc).toBeVisible();
  const box = await loc.boundingBox();
  const vw = page.viewportSize()!.width;
  expect(box, 'has a box').not.toBeNull();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(vw + 0.5);
}

test.describe('home page', () => {
  test('one line, a live picture and three entry cards', async ({ page }) => {
    const errors = await open(page, '#/');
    await expect(page.getByTestId('home')).toBeVisible();
    await expect(page.locator('.home h1')).toHaveCount(1);
    await expect(page.getByTestId('hero-sim')).toBeVisible();
    for (const id of ['echo', 'sandbox', 'learn']) await expect(page.getByTestId(`entry-${id}`)).toBeVisible();
    await expect(page.getByTestId('entry-echo')).toContainText('rebuild a room from its echoes');
    // The secondary row reaches the rest of the site.
    for (const name of ['Gallery', 'Research', 'Loop', 'Lab', 'Docs']) await expect(page.locator('.home-more').getByRole('link', { name: new RegExp(name) })).toBeVisible();
    await page.getByTestId('entry-sandbox').click();
    await expect(page).toHaveURL(/#\/sandbox$/);
    await expect(page.getByTestId('field')).toBeVisible();
    // The logo leads back home.
    await page.getByRole('link', { name: 'Acoustic Sandbox: home' }).click();
    await expect(page.getByTestId('home')).toBeVisible();
    expect(errors).toEqual([]);
  });
});

for (const width of [390, 768]) {
  test.describe(`navigation at ${width} px`, () => {
    test.use({ viewport: { width, height: 844 } });
    test('every section is visible or one tap away in "More"', async ({ page }) => {
      const errors = await open(page, '#/');
      const nav = page.getByRole('navigation', { name: 'Main' });
      for (const s of SECTIONS) await inViewport(page, nav.getByRole('link', { name: s.name, exact: true }));
      await inViewport(page, page.getByTestId('nav-more'));
      await inViewport(page, page.getByRole('link', { name: 'Acoustic Sandbox: home' }));
      // No sideways-scrolling nav.
      expect(await nav.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      for (const s of SECTIONS) {
        await nav.getByRole('link', { name: s.name, exact: true }).click();
        await expect(page).toHaveURL(s.url);
        await expect(nav.getByRole('link', { name: s.name, exact: true })).toHaveAttribute('aria-current', 'page');
      }
      for (const s of MORE) {
        await page.getByTestId('nav-more').click();
        const item = page.getByRole('menuitem', { name: s.name, exact: true });
        await inViewport(page, item);
        await item.click();
        await expect(page).toHaveURL(s.url);
        await expect(page.getByRole('menu')).toHaveCount(0);
        await expect(page.getByTestId('nav-more')).toHaveAttribute('data-current', 'true');
      }
      if (width === 390) {
        // The theme toggle and the source link move into the menu on phones.
        await page.getByTestId('nav-more').click();
        const before = await page.evaluate(() => document.documentElement.dataset.theme);
        await page.getByRole('menuitem', { name: /theme/ }).click();
        expect(await page.evaluate(() => document.documentElement.dataset.theme)).not.toBe(before);
        await page.getByTestId('nav-more').click();
        await inViewport(page, page.getByRole('menuitem', { name: 'Source on GitHub' }));
      }
      expect(errors).toEqual([]);
    });
  });
}

test('"More" menu works from the keyboard and closes on Escape or an outside click', async ({ page }) => {
  await open(page, '#/');
  const more = page.getByTestId('nav-more');
  await more.focus();
  await page.keyboard.press('ArrowDown');
  await expect(page.getByRole('menu')).toBeVisible();
  await expect(page.getByRole('menuitem', { name: 'Research' })).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(page.getByRole('menuitem', { name: 'Loop' })).toBeFocused();
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('ArrowUp');
  await expect(page.getByRole('menuitem', { name: 'Docs' })).toBeFocused(); // wraps
  await page.keyboard.press('Escape');
  await expect(page.getByRole('menu')).toHaveCount(0);
  await expect(more).toBeFocused();
  await expect(more).toHaveAttribute('aria-expanded', 'false');
  // Enter opens it too; an outside click closes it.
  await page.keyboard.press('Enter');
  await expect(page.getByRole('menu')).toBeVisible();
  await page.mouse.click(5, 400);
  await expect(page.getByRole('menu')).toHaveCount(0);
  // Following a menu link navigates.
  await more.click();
  await page.getByRole('menuitem', { name: 'Loop' }).press('Enter');
  await expect(page).toHaveURL(/#\/loop$/);
});

test('unknown routes show "Page not found" with a way home (B25)', async ({ page }) => {
  for (const hash of ['#/nope', '#/sandbox/extra', '#ref-cuda-guide']) {
    await open(page, hash);
    await expect(page.getByTestId('not-found')).toContainText('Page not found');
    await expect(page.getByTestId('field')).toHaveCount(0);
  }
  await page.getByTestId('not-found').getByRole('link', { name: /home page/ }).click();
  await expect(page.getByTestId('home')).toBeVisible();
});

test('old sandbox links still open the sandbox with their scene', async ({ page }) => {
  // Old preset link (#/?preset=…) and the current one (#/sandbox?preset=…).
  await open(page, '#/?preset=double-slit');
  await expect(page.getByTestId('field')).toBeVisible();
  await expect.poll(() => app<string>(page, '(s) => s.scene.name')).toBe('Double slit');
  await expect(page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Sandbox' })).toHaveAttribute('aria-current', 'page');
  await page.goto('./#/sandbox?preset=ellipse');
  await expect.poll(() => app<string>(page, '(s) => s.scene.name')).toBe('Whispering ellipse');
  // Share links: the app now writes #/sandbox?s=…, and the old #/?s=… form still loads.
  await page.getByLabel('Scene name').fill('Old link room');
  await page.getByTestId('share').click();
  await expect(page).toHaveURL(/#\/sandbox\?s=/);
  const token = new URL(page.url()).hash.split('s=')[1];
  const page2 = await page.context().newPage();
  await page2.goto(`./#/?s=${token}`);
  await expect.poll(() => app<string>(page2, '(s) => s.scene.name')).toBe('Old link room');
  await expect(page2.getByTestId('field')).toBeVisible();
});

test('the Sensing tab is gone; Control results survive tab switches (B19)', async ({ page }) => {
  await open(page);
  await expect(page.getByTestId('tab-sensing')).toHaveCount(0);
  await expect(page.getByRole('tab')).toHaveCount(5);
  await page.getByTestId('tab-control').click();
  await page.getByTestId('array-count').fill('5');
  await page.getByTestId('array-count').press('Enter');
  await expect(page.getByTestId('place-array')).toContainText('5 speakers');
  await page.getByTestId('tab-view').click();
  await page.getByTestId('tab-control').click();
  await expect(page.getByTestId('place-array')).toContainText('5 speakers');
  // Arrow keys move between tabs.
  await page.getByTestId('tab-control').focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByTestId('tab-scene')).toHaveAttribute('aria-selected', 'true');
});

test('scene tab keeps grid and boundaries under "Advanced"', async ({ page }) => {
  await open(page);
  await expect(page.getByLabel('Load preset')).toBeVisible();
  await expect(page.getByRole('radiogroup', { name: 'Wall material' })).toBeVisible();
  await expect(page.getByTestId('grid-0')).toBeHidden();
  await expect(page.getByLabel('Outer boundary')).toBeHidden();
  await openAdvanced(page);
  await expect(page.getByTestId('grid-0')).toBeVisible();
  await expect(page.getByLabel('Outer boundary')).toBeVisible();
});

test('raising the Courant number over a painted c(x) is clamped with a warning (B23)', async ({ page }) => {
  await open(page);
  // Paint at the fastest "stable" speed ratio.
  await page.getByLabel('Speed ratio').focus();
  await page.keyboard.press('End');
  const ratio = await app<number>(page, '(s) => s.paintSpeed');
  expect(ratio).toBeGreaterThan(1.3);
  await page.locator('[data-tool="speed"]').click();
  await drag(page, [0.3, 0.3], [0.5, 0.5]);
  expect(await app<number>(page, '(s) => s.runtime.sim.maxSpeedRatio')).toBeGreaterThan(1.3);
  await openAdvanced(page);
  await page.getByTestId('courant').fill('0.7');
  await page.getByTestId('courant').press('Enter');
  await expect(page.getByTestId('toast')).toContainText('Courant number limited');
  const sigma = await app<number>(page, '(s) => Math.sqrt(s.runtime.sim.coeff)');
  const r = await app<number>(page, '(s) => s.runtime.sim.maxSpeedRatio');
  expect(sigma * r).toBeLessThanOrEqual(1 / Math.SQRT2);
  await expect(page.getByTestId('courant')).not.toHaveValue('0.7');
  // And the run stays stable.
  await page.getByTestId('run').click();
  await expect.poll(() => app<number>(page, '(s) => s.runtime.sim.step_count'), { timeout: 30_000 }).toBeGreaterThan(400);
  await page.getByTestId('run').click();
  expect(await app<boolean>(page, '(s) => s.runtime.unstable')).toBe(false);
  expect(await app<boolean>(page, '(s) => Number.isFinite(s.runtime.peak()) && s.runtime.peak() < 100')).toBe(true);
});

test('gallery thumbnails never block the main thread for more than 500 ms (B06)', async ({ page }) => {
  const errors = await open(page, '#/');
  await page.evaluate(() => {
    const w = window as unknown as { __long: number[] };
    w.__long = [];
    new PerformanceObserver((l) => l.getEntries().forEach((e) => w.__long.push(e.duration))).observe({ type: 'longtask', buffered: false });
  });
  const longest = () => page.evaluate(() => Math.max(0, ...(window as unknown as { __long: number[] }).__long));
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Gallery' }).click();
  const cards = page.locator('[data-testid^="preset-"]');
  const n = await cards.count();
  // Scroll through so every card asks for its thumbnail.
  for (let k = 0; k < n; k += 3) await cards.nth(k).scrollIntoViewIfNeeded();
  await expect(page.locator('.preset-card canvas[data-ready]')).toHaveCount(n, { timeout: 60_000 });
  expect(await longest()).toBeLessThan(500);
  // A click right away responds at once.
  const t0 = Date.now();
  await cards.nth(2).click();
  await expect(page.getByTestId('field')).toBeVisible();
  expect(Date.now() - t0).toBeLessThan(2500);
  // Coming back and toggling the theme reuse the cached fields.
  await page.goBack();
  await expect(page.locator('.preset-card canvas[data-ready]').first()).toBeVisible();
  await page.evaluate(() => ((window as unknown as { __long: number[] }).__long = []));
  await page.getByTestId('theme').click();
  await page.waitForTimeout(300);
  expect(await longest()).toBeLessThan(500);
  // Cards are title + one line.
  await expect(cards.first().locator('.body > *')).toHaveCount(2);
  expect(errors.filter((e) => !/WebGL/.test(e))).toEqual([]);
});

test.describe('phone', () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });

  test('the first-visit welcome is a slim banner that does not cover the field', async ({ page }) => {
    await page.goto('./#/sandbox');
    const hint = page.getByTestId('first-run-hint');
    await expect(hint).toBeVisible();
    const h = (await hint.boundingBox())!;
    const f = (await page.getByTestId('field').boundingBox())!;
    const overlap = !(h.y >= f.y + f.height || h.y + h.height <= f.y || h.x >= f.x + f.width || h.x + h.width <= f.x);
    expect(overlap).toBe(false);
    expect(h.height).toBeLessThan(130);
    await hint.getByRole('button', { name: 'Got it' }).click();
    await expect(hint).toHaveCount(0);
    await page.reload();
    await expect(page.getByTestId('field')).toBeVisible();
    await expect(page.getByTestId('first-run-hint')).toHaveCount(0);
  });

  test('choosing a document shows the article, not the list (B31)', async ({ page }) => {
    await open(page, '#/docs/docs/physics');
    const top = async () => (await page.getByTestId('doc-body').boundingBox())!.y;
    await expect(page.getByTestId('doc-body')).toContainText('Physics model and verification');
    expect(await top()).toBeLessThan(250);
    await page.getByTestId('docs-list-toggle').click();
    await expect(page.getByRole('link', { name: 'Overview', exact: true })).toBeVisible();
    await page.getByTestId('docs-group-developer').locator('summary').click();
    await page.getByRole('link', { name: 'waveforms', exact: true }).click();
    await expect(page).toHaveURL(/#\/docs\/docs\/waveforms$/);
    await expect(page.getByTestId('doc-body')).toContainText('waveforms.py');
    await expect(page.getByRole('link', { name: 'Overview', exact: true })).toBeHidden();
    expect(await top()).toBeLessThan(250);
    expect(await page.locator('.page').evaluate((el) => el.scrollTop)).toBe(0);
  });
});

test('docs sidebar is grouped, with module notes folded away', async ({ page }) => {
  await open(page, '#/docs');
  const nav = page.getByRole('navigation', { name: 'Documents' });
  await expect(nav.getByRole('heading', { name: 'Start here' })).toBeVisible();
  await expect(nav.getByRole('link', { name: 'Overview', exact: true })).toHaveAttribute('aria-current', 'page');
  await expect(nav.getByRole('link', { name: 'What can a laptop hear?' })).toBeVisible();
  await expect(nav.getByRole('heading', { name: 'Guides' })).toBeVisible();
  await expect(nav.getByRole('link', { name: 'calculate', exact: true })).toBeHidden();
  await page.getByTestId('docs-group-developer').locator('summary').click();
  await expect(nav.getByRole('link', { name: 'calculate', exact: true })).toBeVisible();
});

test('research lists reports by title under a collapsed "All reports", newest first', async ({ page }) => {
  await open(page, '#/research');
  await expect(page.getByTestId('writeup-card')).toHaveCount(4);
  const reports = page.getByTestId('all-reports');
  await expect(reports.locator('li').first()).toBeHidden();
  await reports.locator('summary').click();
  await expect(reports.getByRole('link', { name: 'Physics-based room imaging without ML (Phase 6: 6.1, 6.2, 6.4, 6.5)' })).toBeVisible();
  const dates = await reports.locator('li .mono').allInnerTexts();
  expect(dates.map((d) => d.trim())).toEqual([...dates.map((d) => d.trim())].sort().reverse());
});
