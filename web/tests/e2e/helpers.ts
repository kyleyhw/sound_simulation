import { expect, type Page } from '@playwright/test';

export async function open(page: Page, hash = '#/'): Promise<string[]> {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  await page.addInitScript(() => {
    try {
      localStorage.setItem('hint-dismissed', '1');
    } catch {
      /* ignore */
    }
  });
  await page.goto(`/${hash}`);
  await expect(page.getByTestId('field').or(page.locator('.content'))).toBeVisible();
  return errors;
}

/** Read a value from the app store. */
export function app<T>(page: Page, fn: string): Promise<T> {
  return page.evaluate(`(() => { const s = window.__app.getState(); return (${fn})(s); })()`) as Promise<T>;
}

export async function fieldBox(page: Page) {
  const box = await page.getByTestId('field').boundingBox();
  if (!box) throw new Error('no field');
  return box;
}

/** Drag across the field between fractional positions. */
export async function drag(page: Page, a: [number, number], b: [number, number], opts: { shift?: boolean } = {}) {
  const box = await fieldBox(page);
  const x0 = box.x + a[0] * box.width;
  const y0 = box.y + a[1] * box.height;
  const x1 = box.x + b[0] * box.width;
  const y1 = box.y + b[1] * box.height;
  if (opts.shift) await page.keyboard.down('Shift');
  await page.mouse.move(x0, y0);
  await page.mouse.down();
  for (let i = 1; i <= 8; i++) await page.mouse.move(x0 + ((x1 - x0) * i) / 8, y0 + ((y1 - y0) * i) / 8);
  await page.mouse.up();
  if (opts.shift) await page.keyboard.up('Shift');
}

export async function clickField(page: Page, fx: number, fy: number) {
  const box = await fieldBox(page);
  await page.mouse.click(box.x + fx * box.width, box.y + fy * box.height);
}

export const wallCount = (page: Page) => app<number>(page, '(s) => s.scene.materials.reduce((a, m) => a + (m ? 1 : 0), 0)');
