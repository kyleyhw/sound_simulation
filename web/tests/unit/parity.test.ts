/**
 * Browser engine vs Python engine (fixtures from scripts/make_web_fixtures.py).
 * Same numerics and ordering, so the fields must agree to float32 round-off.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { type DriverSpec, Simulation } from '../../src/engine/simulation';
import type { WaveformSpec } from '../../src/engine/waveforms';

interface Fixture {
  boundary?: 'rigid' | 'absorb' | 'mur' | 'sponge';
  outer_beta?: number;
  sponge_cells?: number;
  materials?: number[][];
  speed?: number[] | null;
  shape: number[];
  courant: number;
  obstacles: number[][];
  drivers: { pos: number[]; waveform: WaveformSpec }[];
  steps: number;
  timestep: number;
  p: number[];
}

function load(name: string): Fixture {
  return JSON.parse(readFileSync(new URL(`../fixtures/${name}.json`, import.meta.url), 'utf8'));
}

for (const name of ['parity2d', 'parity3d', 'general_rigid', 'general_absorb', 'general_mur', 'general_sponge', 'general_speed']) {
  describe(name, () => {
    it('matches the Python engine', () => {
      const fx = load(name);
      const sim = new Simulation({
        dims: fx.shape.length as 2 | 3,
        shape: fx.shape,
        courant: fx.courant,
        outer: fx.boundary ?? 'soft',
        outerBeta: fx.outer_beta ?? 1,
        spongeCells: fx.sponge_cells ?? 24,
      });
      expect(sim.dt).toBeCloseTo(fx.timestep, 12);
      sim.setCells(fx.obstacles.map((pos) => sim.index(pos)), 1);
      for (const m of fx.materials ?? []) sim.setCells([sim.index(m.slice(0, -1))], m[m.length - 1]);
      if (fx.speed) sim.setSpeedMap(Float32Array.from(fx.speed));
      const drivers: DriverSpec[] = fx.drivers.map((d, i) => ({ id: `d${i}`, pos: d.pos, waveform: d.waveform, enabled: true }));
      sim.setDrivers(drivers);
      for (let s = 0; s < fx.steps; s++) sim.step();
      let maxAbs = 0;
      let num = 0;
      let den = 0;
      for (let i = 0; i < sim.n; i++) {
        const d = sim.p[i] - fx.p[i];
        maxAbs = Math.max(maxAbs, Math.abs(d));
        num += d * d;
        den += fx.p[i] * fx.p[i];
      }
      const rel = Math.sqrt(num / den);
      // Same gate as tests/perf/check_simulate.py.
      expect(maxAbs).toBeLessThan(1e-4);
      expect(rel).toBeLessThan(1e-4);
    });
  });
}
