/// <reference lib="webworker" />
/**
 * Gallery thumbnails off the main thread (bug B06): run each requested
 * preset for a few hundred steps and send back the field and the walls.
 */
import { presetById } from '../engine/presets';
import { buildSimulation } from '../engine/scene';

export const THUMB_STEPS = 260;

export interface ThumbReply {
  id: string;
  rows: number;
  cols: number;
  field: Float32Array;
  material: Uint8Array;
  peak: number;
}

self.onmessage = (e: MessageEvent<{ ids: string[] }>) => {
  for (const id of e.data.ids) {
    const preset = presetById(id);
    if (!preset) continue;
    const sim = buildSimulation(preset.build());
    for (let s = 0; s < THUMB_STEPS; s++) sim.step();
    // 2D presets; for a 3D scene take the middle slice along the last axis.
    const rows = sim.nx;
    const cols = sim.ny;
    const nz = sim.nz || 1;
    const k = Math.floor(nz / 2);
    const field = new Float32Array(rows * cols);
    const material = new Uint8Array(rows * cols);
    let peak = 1e-9;
    for (let q = 0; q < rows * cols; q++) {
      const src = q * nz + k;
      field[q] = sim.p[src];
      material[q] = sim.material[src];
      peak = Math.max(peak, Math.abs(field[q]));
    }
    const reply: ThumbReply = { id, rows, cols, field, material, peak };
    self.postMessage(reply, [field.buffer, material.buffer]);
  }
};
