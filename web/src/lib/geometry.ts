/** Rasterisation helpers for the drawing tools (2D cell space, rows x cols). */

export type Cell = [number, number];

export function discCells(r0: number, c0: number, radius: number, rows: number, cols: number): Cell[] {
  const out: Cell[] = [];
  const rr = Math.max(0, radius - 0.5);
  const R = Math.ceil(rr);
  for (let r = r0 - R; r <= r0 + R; r++)
    for (let c = c0 - R; c <= c0 + R; c++) {
      if (r < 0 || c < 0 || r >= rows || c >= cols) continue;
      if ((r - r0) ** 2 + (c - c0) ** 2 <= rr * rr + 0.25) out.push([r, c]);
    }
  return out;
}

/** Thick line from a to b (stamped disc along a Bresenham-like walk). */
export function lineCells(a: Cell, b: Cell, radius: number, rows: number, cols: number): Cell[] {
  const seen = new Set<number>();
  const out: Cell[] = [];
  const steps = Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]), 1);
  for (let s = 0; s <= steps; s++) {
    const r = Math.round(a[0] + ((b[0] - a[0]) * s) / steps);
    const c = Math.round(a[1] + ((b[1] - a[1]) * s) / steps);
    for (const cell of discCells(r, c, radius, rows, cols)) {
      const k = cell[0] * cols + cell[1];
      if (!seen.has(k)) {
        seen.add(k);
        out.push(cell);
      }
    }
  }
  return out;
}

/** Rectangle outline (thickness t) or filled when fill = true. */
export function rectCells(a: Cell, b: Cell, t: number, fill: boolean, rows: number, cols: number): Cell[] {
  const r0 = Math.max(0, Math.min(a[0], b[0]));
  const r1 = Math.min(rows - 1, Math.max(a[0], b[0]));
  const c0 = Math.max(0, Math.min(a[1], b[1]));
  const c1 = Math.min(cols - 1, Math.max(a[1], b[1]));
  const out: Cell[] = [];
  for (let r = r0; r <= r1; r++)
    for (let c = c0; c <= c1; c++) {
      if (fill || r - r0 < t || r1 - r < t || c - c0 < t || c1 - c < t) out.push([r, c]);
    }
  return out;
}

/** Ellipse inscribed in the a-b box: filled, or a ring of thickness t. */
export function ellipseCells(a: Cell, b: Cell, t: number, fill: boolean, rows: number, cols: number): Cell[] {
  const cr = (a[0] + b[0]) / 2;
  const cc = (a[1] + b[1]) / 2;
  const ry = Math.abs(b[0] - a[0]) / 2 + 0.5;
  const rx = Math.abs(b[1] - a[1]) / 2 + 0.5;
  const out: Cell[] = [];
  for (let r = Math.floor(cr - ry); r <= Math.ceil(cr + ry); r++)
    for (let c = Math.floor(cc - rx); c <= Math.ceil(cc + rx); c++) {
      if (r < 0 || c < 0 || r >= rows || c >= cols) continue;
      const q = ((r - cr) / ry) ** 2 + ((c - cc) / rx) ** 2;
      if (q > 1) continue;
      if (fill) out.push([r, c]);
      else {
        const inner = ((r - cr) / Math.max(0.5, ry - t)) ** 2 + ((c - cc) / Math.max(0.5, rx - t)) ** 2;
        if (inner >= 1) out.push([r, c]);
      }
    }
  return out;
}
