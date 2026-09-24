/**
 * Render a scene to an animated GIF, headless (plan 10.7).
 *
 *   npm run render -- <preset-id | scene.json> [out.gif] [--steps 900] [--every 6]
 *                     [--scale 2] [--colormap icefire] [--db 0] [--colors 256]
 *
 * Runs the browser engine in Node (the same code as the app), colours each
 * frame with the app's colormaps (symmetric, auto-scaled with a slowly
 * decaying peak, or in dB), draws walls grey and sources/probes as dots,
 * and encodes with gifenc. Scene files are the JSON the app saves (Save).
 */

import { readFileSync, writeFileSync } from 'node:fs';
import * as gifencNs from 'gifenc';
import { presetById, PRESETS } from '../src/engine/presets';
import { buildSimulation, type Scene, validateScene } from '../src/engine/scene';
import { colormapLut, type ColormapName } from '../src/render/colormaps';

// gifenc is CommonJS: under Node its exports sit on the default export.
const gifenc = ((gifencNs as unknown as { default?: typeof gifencNs }).default ?? gifencNs) as typeof gifencNs;
const { applyPalette, GIFEncoder, quantize } = gifenc;

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : def;
}

const positional = process.argv.slice(2).filter((a, i, all) => !a.startsWith('--') && !all[i - 1]?.startsWith('--'));
const src = positional[0];
if (!src) {
  console.error(`usage: render <preset-id | scene.json> [out.gif]\npresets: ${PRESETS.map((p) => p.id).join(', ')}`);
  process.exit(2);
}
const scene: Scene = src.endsWith('.json') ? validateScene(JSON.parse(readFileSync(src, 'utf8'))) : (presetById(src)?.build() ?? (() => { throw new Error(`unknown preset ${src}`); })());
const out = positional[1] ?? `${src.replace(/\.json$/, '').replace(/.*\//, '')}.gif`;
const steps = Number(arg('steps', '900'));
const every = Number(arg('every', '6'));
const scale = Number(arg('scale', '2'));
const cmap = arg('colormap', 'icefire') as ColormapName;
const dbRange = Number(arg('db', '0'));
const colors = Number(arg('colors', '256'));

const sim = buildSimulation(scene);
if (sim.dims !== 2) throw new Error('render supports 2D scenes');
const [rows, cols] = [sim.nx, sim.ny];
const W = cols * scale;
const H = rows * scale;
const lut = colormapLut(cmap);
const gif = GIFEncoder();
let peak = 1e-9;
const rgba = new Uint8Array(W * H * 4);
const marks = [...scene.drivers.map((d) => ({ p: d.pos, c: [255, 90, 120] })), ...scene.probes.map((p) => ({ p: p.pos, c: [120, 220, 140] }))];
let frames = 0;
for (let k = 1; k <= steps; k++) {
  sim.step();
  if (k % every) continue;
  let m = 0;
  for (let i = 0; i < sim.n; i++) m = Math.max(m, Math.abs(sim.p[i]));
  peak = Math.max(m, peak * 0.97); // decaying auto-scale: steady, not flickering
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      const q = i * cols + j;
      let r: number;
      let g: number;
      let b: number;
      if (sim.material[q] !== 0) {
        [r, g, b] = [150, 152, 160];
      } else {
        const v = sim.p[q] / peak;
        let t: number;
        if (dbRange > 0) {
          const db = 20 * Math.log10(Math.abs(v) + 1e-12);
          t = 0.5 + 0.5 * Math.sign(v) * Math.max(0, 1 + db / dbRange);
        } else t = 0.5 + 0.5 * Math.max(-1, Math.min(1, v));
        const c = Math.round(t * 255) * 4;
        [r, g, b] = [lut[c], lut[c + 1], lut[c + 2]];
      }
      for (let a = 0; a < scale; a++)
        for (let bb = 0; bb < scale; bb++) {
          const o = ((i * scale + a) * W + j * scale + bb) * 4;
          rgba[o] = r;
          rgba[o + 1] = g;
          rgba[o + 2] = b;
          rgba[o + 3] = 255;
        }
    }
  for (const mk of marks) {
    const [ci, cj] = mk.p;
    for (let a = -scale; a <= scale; a++)
      for (let bb = -scale; bb <= scale; bb++) {
        const y = ci * scale + a;
        const x = cj * scale + bb;
        if (y < 0 || x < 0 || y >= H || x >= W) continue;
        rgba.set([...mk.c, 255], (y * W + x) * 4);
      }
  }
  const palette = quantize(rgba, colors);
  gif.writeFrame(applyPalette(rgba, palette), W, H, { palette, delay: 40 });
  frames++;
}
gif.finish();
writeFileSync(out, gif.bytes());
console.log(`wrote ${out}: ${frames} frames, ${W}x${H}, ${(gif.bytes().length / 1024).toFixed(0)} KB`);
