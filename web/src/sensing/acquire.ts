/**
 * The v2 acquisition protocol in the browser engine: a linear chirp from a
 * driver cell, recorded by a mic pair, p = 0 walls and obstacles on a
 * 64 x 64 grid (simulation/dataset.py, scripts/generate_active_sensing.py).
 */
import { Simulation } from '../engine/simulation';
import type { ModelManifest } from './model';

export type Protocol = ModelManifest['protocol'];

/** dataset.synthetic_chirp: u(t) = sin(2 pi (f0 t + k t^2 / 2)), float32. */
export function chirp(p: Protocol, dt: number): Float32Array {
  const n = Math.floor(p.sample_rate * p.duration * dt);
  const T = n / p.sample_rate;
  const k = (p.f_end - p.f_start) / T;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / p.sample_rate;
    out[i] = Math.sin(2 * Math.PI * (p.f_start * t + 0.5 * k * t * t));
  }
  return out;
}

export function protocolSim(p: Protocol, mask: Uint8Array): Simulation {
  const sim = new Simulation({ dims: 2, shape: [p.grid, p.grid], courant: p.courant });
  sim.setMaterialMap(mask.map((v) => (v ? 1 : 0)));
  return sim;
}

/** Record one pose: returns the two mic series (duration samples each). */
export function recordPose(sim: Simulation, p: Protocol, src: Float32Array, driver: number[], mics: number[][]): [Float32Array, Float32Array] {
  sim.reset();
  sim.setDrivers([{ id: 'src', pos: driver, waveform: { type: 'samples', amplitude: p.amplitude, rate: p.sample_rate, delay: 0, data: Array.from(src) }, enabled: true }]);
  const i0 = sim.index(mics[0]);
  const i1 = sim.index(mics[1]);
  const a = new Float32Array(p.duration);
  const b = new Float32Array(p.duration);
  for (let k = 0; k < p.duration; k++) {
    sim.step();
    a[k] = sim.p[i0];
    b[k] = sim.p[i1];
  }
  return [a, b];
}

/** Random free interior cell and a mic pair spacing apart (dataset.pick_mic_positions, v3 exclusion). */
export function randomPose(mask: Uint8Array, grid: number, spacing: number, rnd: () => number = Math.random, margin = 2): { driver: number[]; mics: number[][] } {
  const free = (i: number, j: number) => i >= margin && j >= margin && i < grid - margin && j < grid - margin && !mask[i * grid + j];
  const cell = () => {
    for (let t = 0; t < 10000; t++) {
      const i = Math.floor(rnd() * grid);
      const j = Math.floor(rnd() * grid);
      if (free(i, j)) return [i, j];
    }
    throw new Error('no free cell');
  };
  const driver = cell();
  for (let t = 0; t < 2000; t++) {
    const c = cell();
    const th = rnd() * 2 * Math.PI;
    const d = [(spacing / 2) * Math.cos(th), (spacing / 2) * Math.sin(th)];
    const m1 = [Math.round(c[0] + d[0]), Math.round(c[1] + d[1])];
    const m2 = [Math.round(c[0] - d[0]), Math.round(c[1] - d[1])];
    if (free(m1[0], m1[1]) && free(m2[0], m2[1]) && !(m1[0] === driver[0] && m1[1] === driver[1]) && !(m2[0] === driver[0] && m2[1] === driver[1]))
      return { driver, mics: [m1, m2] };
  }
  throw new Error('could not place the microphones');
}
