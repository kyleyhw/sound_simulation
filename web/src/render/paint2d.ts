/**
 * Canvas2D painting of a signed 2D pressure field, for small static or
 * low-rate views (gallery thumbnails, the home-page hero) that should not
 * each hold a WebGL context. Same colormaps and wall colours as the
 * sandbox's FieldRenderer.
 */

import { colormapLut } from './colormaps';
import { MATERIAL_RGB } from './fieldRenderer';

export type Theme = 'dark' | 'light';

/** Paint `field` (rows x cols, row-major) into `canvas`, values mapped to [-scale, scale]. */
export function paintField(canvas: HTMLCanvasElement, field: ArrayLike<number>, material: ArrayLike<number> | null, rows: number, cols: number, scale: number, theme: Theme): void {
  if (canvas.width !== cols || canvas.height !== rows) {
    canvas.width = cols;
    canvas.height = rows;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(cols, rows);
  const lut = colormapLut(theme === 'dark' ? 'icefire' : 'balance');
  const walls = MATERIAL_RGB[theme].map((c) => c.map((v) => Math.round(v * 255)));
  const inv = 1 / Math.max(scale, 1e-20);
  const d = img.data;
  for (let q = 0; q < rows * cols; q++) {
    const o = q * 4;
    const m = material ? material[q] : 0;
    if (m > 0) {
      const c = walls[Math.min(m, walls.length - 1)];
      d[o] = c[0];
      d[o + 1] = c[1];
      d[o + 2] = c[2];
    } else {
      const t = Math.max(0, Math.min(1, 0.5 + 0.5 * field[q] * inv));
      const k = Math.round(t * 255) * 4;
      d[o] = lut[k];
      d[o + 1] = lut[k + 1];
      d[o + 2] = lut[k + 2];
    }
    d[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
