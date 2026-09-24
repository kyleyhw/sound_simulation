/**
 * Preset scenes: each demonstrates one piece of wave physics. All are 2D,
 * in grid units (c = dx = 1, dt = 0.5).
 */

import type { WaveformSpec } from './waveforms';
import { type DriverSpec, type ProbeSpec, type SimParams, DEFAULT_PARAMS } from './simulation';
import { encodeRle, type Scene } from './scene';

export interface Preset {
  id: string;
  title: string;
  blurb: string;
  /** Short physics explanation shown in the gallery and the scene panel. */
  physics: string;
  build: () => Scene;
}

const SOFT = 1;
const RIGID = 2;
const ABSORBER = 5;

class Canvas {
  readonly m: Uint8Array;
  constructor(
    readonly nx: number,
    readonly ny: number,
  ) {
    this.m = new Uint8Array(nx * ny);
  }
  set(i: number, j: number, v: number): void {
    if (i >= 0 && i < this.nx && j >= 0 && j < this.ny) this.m[i * this.ny + j] = v;
  }
  rect(i0: number, j0: number, h: number, w: number, v: number): void {
    for (let i = i0; i < i0 + h; i++) for (let j = j0; j < j0 + w; j++) this.set(i, j, v);
  }
  disc(ci: number, cj: number, r: number, v: number): void {
    for (let i = Math.floor(ci - r); i <= ci + r; i++)
      for (let j = Math.floor(cj - r); j <= cj + r; j++) if ((i - ci) ** 2 + (j - cj) ** 2 <= r * r) this.set(i, j, v);
  }
  /** Border of the domain, `t` cells thick. */
  frame(t: number, v: number): void {
    this.rect(0, 0, t, this.ny, v);
    this.rect(this.nx - t, 0, t, this.ny, v);
    this.rect(0, 0, this.nx, t, v);
    this.rect(0, this.ny - t, this.nx, t, v);
  }
}

function scene(
  name: string,
  params: Partial<SimParams>,
  canvas: Canvas,
  drivers: DriverSpec[],
  probes: ProbeSpec[],
  description: string,
  speed?: Float32Array,
): Scene {
  let speedRle: string | undefined;
  if (speed) {
    const q = new Uint8Array(speed.length);
    for (let i = 0; i < q.length; i++) q[i] = speed[i] === 1 ? 0 : Math.max(1, Math.min(255, Math.round(speed[i] * 100)));
    speedRle = encodeRle(q);
  }
  return {
    speed: speedRle,
    version: 1,
    name,
    params: { ...DEFAULT_PARAMS, ...params, shape: [canvas.nx, canvas.ny] },
    materials: encodeRle(canvas.m),
    drivers,
    probes,
    units: 'grid',
    description,
  };
}

const ricker = (f = 0.1, amp = 5, delay = 20): WaveformSpec => ({ type: 'ricker', amplitude: amp, frequency: f, delay });
const tone = (f: number, amp = 1): WaveformSpec => ({ type: 'cosine', amplitude: amp, frequency: f });
const drv = (id: string, i: number, j: number, w: WaveformSpec, extra: Partial<DriverSpec> = {}): DriverSpec => ({
  id,
  pos: [i, j],
  waveform: w,
  enabled: true,
  ...extra,
});
const probe = (id: string, i: number, j: number, label: string): ProbeSpec => ({ id, pos: [i, j], label });

export const PRESETS: Preset[] = [
  {
    id: 'pulse-room',
    title: 'Pulse in a room',
    blurb: 'A single broadband pulse bouncing around a room with two obstacles.',
    physics:
      'A Ricker pulse (mean-zero, dominant frequency 0.1) spreads as a circular wavefront. Each obstacle re-radiates it; the outer walls hold p = 0, so every reflection flips sign.',
    build: () => {
      const c = new Canvas(200, 200);
      c.rect(60, 120, 40, 20, SOFT);
      c.disc(140, 60, 14, SOFT);
      return scene('Pulse in a room', {}, c, [drv('d1', 100, 70, ricker())], [probe('p1', 150, 150, 'Listener')], 'The default scene.');
    },
  },
  {
    id: 'double-slit',
    title: 'Double slit',
    blurb: 'A plane wave through two slits builds an interference pattern.',
    physics:
      'Each slit acts as a new point source (Huygens). Where path lengths differ by a whole wavelength the waves add; half a wavelength, they cancel. Fringe spacing ≈ λL/d.',
    build: () => {
      const c = new Canvas(220, 220);
      c.rect(70, 0, 4, 220, RIGID);
      c.rect(70, 96, 4, 8, 0);
      c.rect(70, 116, 4, 8, 0);
      const drivers: DriverSpec[] = [];
      for (let j = 4; j < 216; j += 2) drivers.push(drv(`s${j}`, 30, j, tone(0.06, 0.4)));
      return scene('Double slit', { outer: 'cpml', cpmlCells: 16 }, c, drivers, [probe('p1', 180, 110, 'Screen centre')], 'Line source of phased tones, rigid barrier, absorbing edges.');
    },
  },
  {
    id: 'ellipse',
    title: 'Whispering ellipse',
    blurb: 'Sound from one focus of an ellipse re-converges at the other.',
    physics:
      'Every ray leaving one focus of an ellipse reflects through the other focus, and all such paths have equal length — so the reflected pulse arrives there all at once.',
    build: () => {
      const c = new Canvas(200, 240);
      const a = 110;
      const b = 80;
      const ci = 100;
      const cj = 120;
      for (let i = 0; i < 200; i++)
        for (let j = 0; j < 240; j++) {
          const r = ((i - ci) / b) ** 2 + ((j - cj) / a) ** 2;
          if (r >= 1) c.set(i, j, RIGID);
        }
      const f = Math.sqrt(a * a - b * b);
      return scene(
        'Whispering ellipse',
        {},
        c,
        [drv('d1', ci, Math.round(cj - f), ricker(0.12, 5, 15))],
        [probe('p1', ci, Math.round(cj + f), 'Other focus'), probe('p2', ci + 40, cj, 'Off focus')],
        'Rigid elliptical wall; source at the left focus.',
      );
    },
  },
  {
    id: 'parabola',
    title: 'Parabolic dish',
    blurb: 'A plane wave focused by a rigid parabolic reflector.',
    physics: 'A parabola maps parallel rays to its focus: the reflected wavefront converges to a point, boosting pressure there.',
    build: () => {
      const c = new Canvas(220, 220);
      const fdist = 40;
      const vertexJ = 190;
      for (let i = 0; i < 220; i++) {
        const y = i - 110;
        const j = Math.round(vertexJ - (y * y) / (4 * fdist));
        for (let t = 0; t < 3; t++) if (j + t < 220 && Math.abs(y) < 95) c.set(i, j + t, RIGID);
      }
      const drivers: DriverSpec[] = [];
      for (let i = 20; i < 200; i += 2) drivers.push(drv(`s${i}`, i, 25, ricker(0.08, 1, 15)));
      return scene(
        'Parabolic dish',
        { outer: 'cpml', cpmlCells: 16 },
        c,
        drivers,
        [probe('p1', 110, vertexJ - fdist, 'Focus')],
        'Line source launches a plane pulse towards the dish.',
      );
    },
  },
  {
    id: 'room-modes',
    title: 'Room modes',
    blurb: 'Drive a rigid room at a resonance and a standing wave appears.',
    physics:
      'A rigid rectangular room resonates at f = (c/2)·√((m/Lx)² + (n/Ly)²). Driving at the (2,1) mode frequency builds a standing pattern with fixed nodal lines.',
    build: () => {
      const c = new Canvas(162, 122);
      c.frame(1, RIGID);
      const Lx = 160;
      const Ly = 120;
      const f = 0.5 * Math.sqrt((2 / Lx) ** 2 + (1 / Ly) ** 2);
      return scene(
        'Room modes',
        {},
        c,
        [drv('d1', 5, 5, tone(f, 0.2))],
        [probe('p1', 41, 31, 'Antinode'), probe('p2', 81, 61, 'Node')],
        `Rigid box ${Lx}x${Ly}; drive at f(2,1) = ${f.toFixed(4)}.`,
      );
    },
  },
  {
    id: 'anechoic',
    title: 'Anechoic vs reverberant',
    blurb: 'The same pulse with absorbing (left) and reflecting (right) walls.',
    physics:
      'An absorbing layer (PML) swallows outgoing waves, as in an anechoic chamber. With reflecting walls the energy stays trapped and decays only through the absorber panel.',
    build: () => {
      const c = new Canvas(200, 260);
      c.rect(0, 128, 200, 4, RIGID);
      c.rect(10, 250, 180, 6, ABSORBER);
      return scene(
        'Anechoic vs reverberant',
        { outer: 'cpml', cpmlCells: 20 },
        c,
        [drv('d1', 100, 64, ricker()), drv('d2', 100, 194, ricker())],
        [probe('p1', 60, 64, 'Open side'), probe('p2', 60, 194, 'Room side')],
        'Left half: open field. Right half: rigid walls plus one absorber.',
      );
    },
  },
  {
    id: 'sonic-crystal',
    title: 'Sonic crystal',
    blurb: 'A lattice of rigid rods blocks one band of frequencies.',
    physics:
      'Bragg scattering from a periodic lattice (spacing a) opens a band gap near f ≈ c/2a: waves in the gap decay inside the crystal, others pass.',
    build: () => {
      const c = new Canvas(200, 240);
      const a = 12;
      for (let i = 20; i < 190; i += a) for (let j = 100; j < 184; j += a) c.disc(i, j, 3.5, RIGID);
      return scene(
        'Sonic crystal',
        { outer: 'cpml', cpmlCells: 16 },
        c,
        [drv('d1', 100, 40, tone(1 / (2 * a), 0.5))],
        [probe('p1', 100, 80, 'Before'), probe('p2', 100, 210, 'After')],
        `Square lattice a = ${a}; drive near the gap centre c/2a.`,
      );
    },
  },
  {
    id: 'helmholtz',
    title: 'Helmholtz resonator',
    blurb: 'A cavity with a narrow neck rings at one low frequency.',
    physics:
      'Air in the neck acts as a mass on the spring of the cavity air, so the cavity rings at f ≈ (c/2π)·√(A/(V·L)) — far below the cavity’s own modes.',
    build: () => {
      const c = new Canvas(200, 200);
      c.rect(100, 60, 70, 80, RIGID);
      c.rect(104, 64, 62, 72, 0);
      c.rect(100, 96, 4, 8, 0);
      return scene(
        'Helmholtz resonator',
        { outer: 'cpml', cpmlCells: 16 },
        c,
        [drv('d1', 50, 100, ricker(0.05, 6, 30))],
        [probe('p1', 140, 100, 'Inside cavity'), probe('p2', 60, 60, 'Outside')],
        'Rigid box with a 8-cell neck; broadband pulse from outside.',
      );
    },
  },
  {
    id: 'lens',
    title: 'Acoustic lens',
    blurb: 'A disc of slower sound focuses a plane wave.',
    physics:
      'Sound bends towards regions where it travels slower (Snell’s law). A disc whose centre is slowest acts like a converging lens: the flat wavefront curves and meets at a focus behind it.',
    build: () => {
      const c = new Canvas(200, 260);
      const speed = new Float32Array(200 * 260).fill(1);
      const ci = 100;
      const cj = 100;
      const R = 50;
      for (let i = 0; i < 200; i++)
        for (let j = 0; j < 260; j++) {
          const r = Math.hypot(i - ci, j - cj);
          // Gradient-index profile n(r) = n0 (1 - (r/R)^2 / 2): c = c0 / n.
          if (r < R) speed[i * 260 + j] = 1 / (1.6 - 0.6 * (r / R) ** 2);
        }
      const drivers: DriverSpec[] = [];
      for (let i = 20; i < 180; i += 2) drivers.push(drv(`s${i}`, i, 25, ricker(0.06, 1, 20)));
      return scene(
        'Acoustic lens',
        { outer: 'cpml', cpmlCells: 16 },
        c,
        drivers,
        [probe('p1', 100, 185, 'Focus')],
        'Graded sound-speed disc (c from 0.63 at the centre to 1 at the rim).',
        speed,
      );
    },
  },
  {
    id: 'echolocation',
    title: 'Echolocation',
    blurb: 'A chirp and its echoes: the delay encodes distance.',
    physics:
      'A source and a microphone sit side by side. Each echo arrives after a round trip, t = 2d/c, so the recorded delays map the distance to every reflector.',
    build: () => {
      const c = new Canvas(220, 220);
      c.rect(40, 160, 60, 8, RIGID);
      c.disc(160, 150, 12, RIGID);
      return scene(
        'Echolocation',
        { outer: 'cpml', cpmlCells: 16 },
        c,
        [drv('d1', 110, 40, { type: 'burst', amplitude: 3, frequency: 0.1, cycles: 3, delay: 5 })],
        [probe('p1', 110, 44, 'Microphone')],
        'Open field (absorbing edges), a wall and a pillar.',
      );
    },
  },
  {
    id: 'corridor',
    title: 'Waveguide',
    blurb: 'Sound trapped in a corridor travels without spreading.',
    physics:
      'Between two rigid walls, only discrete transverse modes propagate; below the first cut-off (f < c/2w) the wave travels as a plane wave with no geometric spreading.',
    build: () => {
      const c = new Canvas(120, 300);
      c.rect(40, 0, 3, 300, RIGID);
      c.rect(77, 0, 3, 300, RIGID);
      return scene(
        'Waveguide',
        { outer: 'cpml', cpmlCells: 16 },
        c,
        [drv('d1', 60, 30, ricker(0.03, 5, 40))],
        [probe('p1', 60, 150, 'Middle'), probe('p2', 60, 260, 'Far end')],
        'Rigid-walled channel, width 34 cells.',
      );
    },
  },
];

export function presetById(id: string): Preset | undefined {
  return PRESETS.find((p) => p.id === id);
}
