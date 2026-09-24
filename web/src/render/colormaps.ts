/**
 * Colormap lookup tables (256 x RGBA8). Diverging maps for signed pressure,
 * sequential maps for magnitudes (dB, RMS). Control points are
 * perceptually ordered; interpolation is linear in sRGB, adequate at 256
 * steps for a field display.
 */

export type ColormapName = 'icefire' | 'balance' | 'magma' | 'viridis' | 'gray';

export const DIVERGING: ColormapName[] = ['icefire', 'balance', 'gray'];
export const SEQUENTIAL: ColormapName[] = ['magma', 'viridis', 'gray'];

const STOPS: Record<ColormapName, [number, number, number, number][]> = {
  // dark-centred diverging (reads well on dark backgrounds)
  icefire: [
    [0.0, 189, 229, 255],
    [0.18, 70, 160, 230],
    [0.38, 30, 70, 140],
    [0.5, 14, 16, 26],
    [0.62, 130, 35, 45],
    [0.82, 225, 90, 55],
    [1.0, 255, 220, 170],
  ],
  // light-centred diverging (reads well on light backgrounds)
  balance: [
    [0.0, 24, 40, 110],
    [0.2, 45, 105, 190],
    [0.4, 165, 200, 230],
    [0.5, 248, 248, 246],
    [0.6, 240, 185, 160],
    [0.8, 200, 70, 55],
    [1.0, 110, 15, 30],
  ],
  magma: [
    [0.0, 0, 0, 4],
    [0.25, 80, 18, 123],
    [0.5, 182, 54, 121],
    [0.75, 251, 136, 97],
    [1.0, 252, 253, 191],
  ],
  viridis: [
    [0.0, 68, 1, 84],
    [0.25, 59, 82, 139],
    [0.5, 33, 145, 140],
    [0.75, 94, 201, 98],
    [1.0, 253, 231, 37],
  ],
  gray: [
    [0.0, 0, 0, 0],
    [1.0, 255, 255, 255],
  ],
};

const cache = new Map<ColormapName, Uint8Array>();

export function colormapLut(name: ColormapName): Uint8Array {
  const hit = cache.get(name);
  if (hit) return hit;
  const stops = STOPS[name];
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, r0, g0, b0] = stops[k];
    const [t1, r1, g1, b1] = stops[k + 1];
    const a = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
    lut[i * 4] = Math.round(r0 + a * (r1 - r0));
    lut[i * 4 + 1] = Math.round(g0 + a * (g1 - g0));
    lut[i * 4 + 2] = Math.round(b0 + a * (b1 - b0));
    lut[i * 4 + 3] = 255;
  }
  cache.set(name, lut);
  return lut;
}

/** CSS linear-gradient for a legend bar. */
export function colormapCss(name: ColormapName): string {
  const lut = colormapLut(name);
  const parts: string[] = [];
  for (let i = 0; i <= 8; i++) {
    const k = Math.round((i / 8) * 255) * 4;
    parts.push(`rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]}) ${(i / 8) * 100}%`);
  }
  return `linear-gradient(to right, ${parts.join(', ')})`;
}
