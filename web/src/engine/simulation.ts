/**
 * Browser FDTD engine: a TypeScript port of src/acoustic_system/simulation/
 * simulate.py for 2D and 3D grids.
 *
 * Numerics (identical ordering to the Python engine, so results agree to
 * float32 round-off — see tests/unit/parity.test.ts):
 *
 *   p^{n+1} = 2 p^n - p^{n-1} + (c dt/dx)^2 * Lap(p^n)     (interior)
 *   outer faces      -> boundary condition (below)
 *   obstacle cells   -> per-cell wall model (below)
 *   drivers          -> p^{n+1}[pos] += w(t_n)   (soft source, added last)
 *   rotate buffers; t += dt
 *
 * Layout: C order, index = (i * ny + j) (2D) or ((i * ny + j) * nz + k) (3D);
 * i is the row (screen y), j the column (screen x).
 *
 * Wall models (Phase 5; the default reproduces the Python engine). The
 * general path is the same formulation as src/acoustic_system/simulation/
 * physics.py (see its docstring and docs/physics.md), parity-tested:
 *  - 'soft'  : Dirichlet p = 0 (pressure-release, reflection -1). Default.
 *  - 'rigid' : Neumann dp/dn = 0 (reflection +1), staircase.
 *  - 'absorb': locally reacting impedance wall, spring-damper branch
 *              R v + K_s x = p (frequency-independent when K_s = 0).
 * Outer boundary: 'soft' | 'rigid' | 'absorb' | 'mur' (1st-order absorbing
 * edge) | 'sponge' (graded damping layer).
 */

import { evalWaveform, type WaveformSpec } from './waveforms';

export type Dims = 2 | 3;
export type WallKind = 'soft' | 'rigid' | 'absorb';
export type OuterKind = WallKind | 'mur' | 'sponge';

export interface Material {
  kind: WallKind;
  /** High-frequency normalised admittance beta = rho c / R ('absorb'). */
  beta: number;
  /** High-pass corner f_c * dx / c (0 = frequency-independent). */
  fcRel: number;
}

export interface DriverSpec {
  id: string;
  pos: number[];
  waveform: WaveformSpec;
  enabled: boolean;
  /** Extra per-driver gain and delay (array steering). */
  gain?: number;
  delay?: number;
}

export interface ProbeSpec {
  id: string;
  pos: number[];
  label?: string;
}

export interface SimParams {
  dims: Dims;
  shape: number[];
  /** Wave speed (grid units: 1; SI: 343 m/s). */
  c: number;
  /** Cell size (grid units: 1; SI: metres). */
  dx: number;
  courant: number;
  outer: OuterKind;
  /** Admittance of the outer walls when outer === 'absorb'. */
  outerBeta: number;
  /** Damping-layer thickness in cells when outer === 'sponge'. */
  spongeCells: number;
}

export const DEFAULT_PARAMS: SimParams = {
  dims: 2,
  shape: [200, 200],
  c: 1,
  dx: 1,
  courant: 0.5,
  outer: 'soft',
  outerBeta: 1,
  spongeCells: 24,
};

export const PROBE_CAPACITY = 16384;

/** Pre-built materials indexed by material id stored per cell (0 = air). */
export const MATERIALS: Material[] = [
  { kind: 'soft', beta: 0, fcRel: 0 }, // 0: air (unused for walls)
  { kind: 'soft', beta: 0, fcRel: 0 }, // 1: pressure-release (legacy default)
  { kind: 'rigid', beta: 0, fcRel: 0 }, // 2: rigid (concrete)
  { kind: 'absorb', beta: 0.1, fcRel: 0 }, // 3: plaster  (alpha ~ 0.33)
  { kind: 'absorb', beta: 0.3, fcRel: 0 }, // 4: wood     (alpha ~ 0.71)
  { kind: 'absorb', beta: 1.0, fcRel: 0 }, // 5: absorber (alpha ~ 1 at normal incidence)
  { kind: 'absorb', beta: 1.0, fcRel: 0.03 }, // 6: curtain: absorbs above f_c, reflects below
];

export const MATERIAL_NAMES = ['Air', 'Soft (p = 0)', 'Rigid', 'Plaster', 'Wood', 'Absorber', 'Curtain'];

/** Normal-incidence absorption coefficient for admittance beta. */
export function absorptionFromBeta(beta: number): number {
  const r = (1 - beta) / (1 + beta);
  return 1 - r * r;
}

export class Simulation {
  readonly params: SimParams;
  readonly dims: Dims;
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly n: number;
  readonly dt: number;
  readonly coeff: number;

  p: Float32Array;
  pPrev: Float32Array;
  private pNext: Float32Array;
  /** Per-cell material id; 0 = air. */
  readonly material: Uint8Array;
  /** Relative sound speed c(x)/c (1 = nominal). Optional heterogeneity. */
  readonly speed: Float32Array;
  private hasSpeed = false;

  time = 0;
  step_count = 0;
  drivers: DriverSpec[] = [];
  probes: ProbeSpec[] = [];
  /** Probe ring buffers, keyed by probe id. */
  readonly probeData = new Map<string, { buf: Float32Array; count: number }>();

  // Optional accumulators (enabled by views that need them).
  rmsAccum: Float32Array | null = null;
  rmsCount = 0;
  vel: Float32Array[] | null = null; // particle velocity components (for intensity)
  intensity: Float32Array[] | null = null;

  private obstacleCount = 0;
  private generalPath = false;
  private spongeSigma: Float32Array | null = null;

  constructor(params: Partial<SimParams> = {}) {
    this.params = { ...DEFAULT_PARAMS, ...params, shape: [...(params.shape ?? DEFAULT_PARAMS.shape)] };
    const p = this.params;
    this.dims = p.dims;
    if (p.shape.length !== p.dims) throw new Error(`shape ${p.shape} is not ${p.dims}D`);
    if (!(p.c > 0) || !(p.dx > 0) || !(p.courant > 0)) throw new Error('c, dx and courant must be positive');
    this.nx = p.shape[0];
    this.ny = p.shape[1];
    this.nz = p.dims === 3 ? p.shape[2] : 1;
    this.n = this.nx * this.ny * this.nz;
    // Same rule as Simulate: kappa = min(courant, 0.95 / sqrt(d)).
    const kappa = Math.min(p.courant, 0.95 / Math.sqrt(p.dims));
    this.dt = (kappa * p.dx) / p.c;
    this.coeff = Math.fround(((p.c * this.dt) / p.dx) ** 2);
    this.p = new Float32Array(this.n);
    this.pPrev = new Float32Array(this.n);
    this.pNext = new Float32Array(this.n);
    this.material = new Uint8Array(this.n);
    this.speed = new Float32Array(this.n).fill(1);
    this.refreshPaths();
  }

  // ------------------------------------------------------------------ //
  // Geometry
  // ------------------------------------------------------------------ //

  index(pos: number[]): number {
    if (this.dims === 2) return pos[0] * this.ny + pos[1];
    return (pos[0] * this.ny + pos[1]) * this.nz + pos[2];
  }

  inBounds(pos: number[]): boolean {
    if (pos.length !== this.dims) return false;
    for (let a = 0; a < this.dims; a++) {
      if (!Number.isInteger(pos[a]) || pos[a] < 0 || pos[a] >= this.params.shape[a]) return false;
    }
    return true;
  }

  /** Set material id (0 = air) on a list of flat cell indices. */
  setCells(indices: ArrayLike<number>, materialId: number): void {
    const m = this.material;
    for (let q = 0; q < indices.length; q++) {
      const idx = indices[q];
      if (idx < 0 || idx >= this.n) continue;
      m[idx] = materialId;
      if (materialId !== 0) {
        this.p[idx] = 0;
        this.pPrev[idx] = 0;
        this.pNext[idx] = 0;
      }
    }
    this.refreshPaths();
  }

  setMaterialMap(map: Uint8Array): void {
    if (map.length !== this.n) throw new Error('material map size mismatch');
    this.material.set(map);
    for (let i = 0; i < this.n; i++) {
      if (map[i] !== 0) {
        this.p[i] = 0;
        this.pPrev[i] = 0;
        this.pNext[i] = 0;
      }
    }
    this.refreshPaths();
  }

  clearObstacles(): void {
    this.material.fill(0);
    this.refreshPaths();
  }

  setSpeedMap(map: Float32Array | null): void {
    if (map === null) {
      this.speed.fill(1);
    } else {
      if (map.length !== this.n) throw new Error('speed map size mismatch');
      this.speed.set(map);
    }
    this.refreshPaths();
  }

  /** Stable timestep bound given the current speed map. */
  get maxSpeedRatio(): number {
    let m = 1;
    for (let i = 0; i < this.n; i++) if (this.speed[i] > m) m = this.speed[i];
    return m;
  }

  private refreshPaths(): void {
    let count = 0;
    let special = false;
    for (let i = 0; i < this.n; i++) {
      const id = this.material[i];
      if (id !== 0) {
        count++;
        if (id !== 1) special = true;
      }
    }
    let speedVaries = false;
    for (let i = 0; i < this.n; i++) {
      if (this.speed[i] !== 1) {
        speedVaries = true;
        break;
      }
    }
    this.hasSpeed = speedVaries;
    this.obstacleCount = count;
    // The fast path is the exact Python-parity path: soft outer walls,
    // soft (p = 0) obstacles only, uniform c. Anything else uses the
    // general kernel.
    this.generalPath = special || speedVaries || this.params.outer !== 'soft';
    if (this.params.outer === 'sponge') this.buildSponge();
    this.gK = null; // rebuilt lazily on the next general step
  }

  get obstacles(): number {
    return this.obstacleCount;
  }

  // ------------------------------------------------------------------ //
  // Drivers and probes
  // ------------------------------------------------------------------ //

  setDrivers(drivers: DriverSpec[]): void {
    this.drivers = drivers.filter((d) => this.inBounds(d.pos));
  }

  setProbes(probes: ProbeSpec[]): void {
    this.probes = probes.filter((p) => this.inBounds(p.pos));
    const keep = new Set(this.probes.map((p) => p.id));
    for (const id of [...this.probeData.keys()]) if (!keep.has(id)) this.probeData.delete(id);
    for (const p of this.probes) {
      if (!this.probeData.has(p.id)) this.probeData.set(p.id, { buf: new Float32Array(PROBE_CAPACITY), count: 0 });
    }
  }

  /** Probe samples in chronological order (at most PROBE_CAPACITY). */
  probeSeries(id: string): Float32Array {
    const d = this.probeData.get(id);
    if (!d) return new Float32Array(0);
    const len = Math.min(d.count, PROBE_CAPACITY);
    const out = new Float32Array(len);
    const start = d.count - len;
    for (let q = 0; q < len; q++) out[q] = d.buf[(start + q) % PROBE_CAPACITY];
    return out;
  }

  // ------------------------------------------------------------------ //
  // Accumulators
  // ------------------------------------------------------------------ //

  enableRms(on: boolean): void {
    this.rmsAccum = on ? new Float32Array(this.n) : null;
    this.rmsCount = 0;
  }

  enableIntensity(on: boolean): void {
    if (on) {
      this.vel = Array.from({ length: this.dims }, () => new Float32Array(this.n));
      this.intensity = Array.from({ length: this.dims }, () => new Float32Array(this.n));
    } else {
      this.vel = null;
      this.intensity = null;
    }
  }

  resetAccumulators(): void {
    this.rmsAccum?.fill(0);
    this.rmsCount = 0;
    this.vel?.forEach((v) => v.fill(0));
    this.intensity?.forEach((v) => v.fill(0));
  }

  // ------------------------------------------------------------------ //
  // Time stepping
  // ------------------------------------------------------------------ //

  reset(): void {
    this.p.fill(0);
    this.pPrev.fill(0);
    this.pNext.fill(0);
    this.time = 0;
    this.step_count = 0;
    for (const d of this.probeData.values()) d.count = 0;
    this.wallV?.fill(0);
    this.wallX?.fill(0);
    this.resetAccumulators();
  }

  step(): void {
    const { p, pPrev, pNext } = this;
    if (this.generalPath) {
      this.stepGeneral();
    } else if (this.dims === 2) {
      kernel2dSoft(p, pPrev, pNext, this.nx, this.ny, this.coeff);
    } else {
      kernel3dSoft(p, pPrev, pNext, this.nx, this.ny, this.nz, this.coeff);
    }
    // Obstacle scrub for soft (p = 0) cells on the fast path; the general
    // kernel handles every material itself.
    if (!this.generalPath && this.obstacleCount > 0) {
      const m = this.material;
      for (let i = 0; i < this.n; i++) if (m[i] !== 0) pNext[i] = 0;
    }
    // Drivers last (soft source; may overwrite a wall, as in Python).
    const t = this.time;
    for (const d of this.drivers) {
      if (!d.enabled) continue;
      const v = evalWaveform(d.waveform, t - (d.delay ?? 0)) * (d.gain ?? 1);
      pNext[this.index(d.pos)] += v;
    }
    // Rotate.
    this.pPrev = p;
    this.p = pNext;
    this.pNext = pPrev;
    this.time = t + this.dt;
    this.step_count += 1;
    this.afterStep();
  }

  private afterStep(): void {
    const p = this.p;
    for (const probe of this.probes) {
      const d = this.probeData.get(probe.id);
      if (!d) continue;
      d.buf[d.count % PROBE_CAPACITY] = p[this.index(probe.pos)];
      d.count++;
    }
    if (this.rmsAccum) {
      const a = this.rmsAccum;
      for (let i = 0; i < this.n; i++) a[i] += p[i] * p[i];
      this.rmsCount++;
    }
    if (this.vel && this.intensity) this.updateIntensity();
  }

  /**
   * Particle velocity from Euler's equation, rho dv/dt = -grad p (rho = 1),
   * integrated with the current pressure; instantaneous intensity p v is
   * accumulated (its time average is the active intensity).
   */
  private updateIntensity(): void {
    const { p, nx, ny, nz, dims, dt } = this;
    const vel = this.vel!;
    const I = this.intensity!;
    const k = dt / this.params.dx;
    const sy = nz;
    const sx = ny * nz;
    for (let i = 1; i < nx - 1; i++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let z = dims === 3 ? 1 : 0; z < (dims === 3 ? nz - 1 : 1); z++) {
          const idx = i * sx + j * sy + z;
          if (this.material[idx] !== 0) continue;
          vel[0][idx] -= k * 0.5 * (p[idx + sx] - p[idx - sx]);
          vel[1][idx] -= k * 0.5 * (p[idx + sy] - p[idx - sy]);
          I[0][idx] += p[idx] * vel[0][idx];
          I[1][idx] += p[idx] * vel[1][idx];
          if (dims === 3) {
            vel[2][idx] -= k * 0.5 * (p[idx + 1] - p[idx - 1]);
            I[2][idx] += p[idx] * vel[2][idx];
          }
        }
      }
    }
  }

  /** Mean-square pressure map since the last accumulator reset. */
  rmsMap(): Float32Array | null {
    if (!this.rmsAccum || this.rmsCount === 0) return null;
    const out = new Float32Array(this.n);
    const inv = 1 / this.rmsCount;
    for (let i = 0; i < this.n; i++) out[i] = Math.sqrt(this.rmsAccum[i] * inv);
    return out;
  }

  /** Total discrete acoustic energy (kinetic + potential proxy), for tests. */
  energy(): number {
    // E = 1/2 sum (dp/dt)^2 / c^2 + 1/2 sum |grad p|^2 (staggered in time).
    const { p, pPrev, nx, ny, nz, dt } = this;
    let kin = 0;
    let pot = 0;
    const c2 = this.params.c * this.params.c;
    const dx2 = this.params.dx * this.params.dx;
    for (let i = 0; i < this.n; i++) {
      const v = (p[i] - pPrev[i]) / dt;
      kin += (v * v) / c2;
    }
    const sx = ny * nz;
    const sy = nz;
    for (let i = 0; i < nx; i++)
      for (let j = 0; j < ny; j++)
        for (let z = 0; z < nz; z++) {
          const idx = i * sx + j * sy + z;
          if (i + 1 < nx) pot += ((p[idx + sx] - p[idx]) * (pPrev[idx + sx] - pPrev[idx])) / dx2;
          if (j + 1 < ny) pot += ((p[idx + sy] - p[idx]) * (pPrev[idx + sy] - pPrev[idx])) / dx2;
          if (this.dims === 3 && z + 1 < nz) pot += ((p[idx + 1] - p[idx]) * (pPrev[idx + 1] - pPrev[idx])) / dx2;
        }
    return 0.5 * kin + 0.5 * pot;
  }

  // ------------------------------------------------------------------ //
  // General kernel: materials, rigid/absorbing walls, c(x), PML
  // ------------------------------------------------------------------ //

  private buildSponge(): void {
    const L = Math.max(1, Math.round(this.params.spongeCells));
    const sigma = new Float32Array(this.n);
    // Graded damping sigma(d) = sigma_max ((L - d)/L)^3 inside the layer;
    // sigma_max chosen for a theoretical normal-incidence reflection of
    // ~1e-4 (see docs/physics.md).
    const R0 = 1e-4;
    const sigmaMax = (-(3 + 1) * Math.log(R0) * this.params.c) / (2 * L * this.params.dx);
    const { nx, ny, nz } = this;
    for (let i = 0; i < nx; i++)
      for (let j = 0; j < ny; j++)
        for (let z = 0; z < nz; z++) {
          const di = Math.min(i, nx - 1 - i);
          const dj = Math.min(j, ny - 1 - j);
          const dz = this.dims === 3 ? Math.min(z, nz - 1 - z) : L;
          const d = Math.min(di, dj, dz);
          if (d < L) {
            const f = (L - d) / L;
            sigma[(i * ny + j) * nz + z] = sigmaMax * f * f * f;
          }
        }
    this.spongeSigma = sigma;
  }

  // Precomputed per-cell coefficients for the general kernel (see physics.py).
  private gK: Uint8Array | null = null; // neighbours contributing -p_c (air, soft wall)
  private gC: Float32Array | null = null; // (c(x) dt / dx)^2
  private gS: Float32Array | null = null; // sigma dt (damping layer)
  private gQ: Float32Array | null = null; // rho c lambda M (impedance branch)
  private gQa: Float32Array | null = null; // Q / a
  private gInvA: Float32Array | null = null; // 1 / a, a = R + K_s dt / 2
  private gKs: Float32Array | null = null; // spring constant K_s
  private gActive: Uint8Array | null = null; // 0 = held at p = 0 (or Mur edge)
  private wallV: Float32Array | null = null; // branch velocity v^{n-1/2}
  private wallX: Float32Array | null = null; // branch displacement x^n

  /**
   * Coefficients for
   *   (1 + q/2 + s) p^{n+1} = 2p^n - p^{n-1} + C L + s p^{n-1}
   *                           - q (p^n / 2 - K_s x^n) + Q v^{n-1/2},
   *   L = sum_nbrs p_n - K p_c   (wall cells store p = 0).
   * Rigid faces drop out of K and add no damping; impedance faces combine
   * through the admittance sum; the branch uses the most absorbing face's
   * material with an effective face count M = sum(beta) / beta_rep.
   */
  private buildGeneral(): void {
    const { nx, ny, nz, dims, material, n } = this;
    const outer = this.params.outer;
    const heldEdges = outer === 'soft' || outer === 'sponge' || outer === 'mur';
    const outerBeta = outer === 'absorb' ? this.params.outerBeta : 0;
    const lam = Math.sqrt(this.coeff);
    const c = this.params.c;
    const rho = 1;
    const K = new Uint8Array(n);
    const C = new Float32Array(n);
    const S = new Float32Array(n);
    const Q = new Float32Array(n);
    const Qa = new Float32Array(n);
    const InvA = new Float32Array(n);
    const Ks = new Float32Array(n);
    const active = new Uint8Array(n);
    const sigma = this.spongeSigma;
    const sx = ny * nz;
    const sy = nz;
    for (let i = 0; i < nx; i++)
      for (let j = 0; j < ny; j++)
        for (let z = 0; z < nz; z++) {
          const idx = i * sx + j * sy + z;
          const onEdge = i === 0 || i === nx - 1 || j === 0 || j === ny - 1 || (dims === 3 && (z === 0 || z === nz - 1));
          if (material[idx] !== 0 || (heldEdges && onEdge)) continue;
          active[idx] = 1;
          const r = this.hasSpeed ? this.speed[idx] : 1;
          let k = 0;
          let betaSum = 0;
          let betaRep = 0;
          let fcRep = 0;
          for (let a = 0; a < dims; a++) {
            const stride = a === 0 ? sx : a === 1 ? sy : 1;
            const coord = a === 0 ? i : a === 1 ? j : z;
            const size = a === 0 ? nx : a === 1 ? ny : nz;
            for (let s2 = -1; s2 <= 1; s2 += 2) {
              const c2 = coord + s2;
              let beta = 0;
              let fc = 0;
              if (c2 < 0 || c2 >= size) {
                beta = outerBeta; // rigid/impedance outer wall (held edges never get here)
              } else {
                const mat = MATERIALS[material[idx + s2 * stride]] ?? MATERIALS[1];
                if (material[idx + s2 * stride] === 0 || mat.kind === 'soft') {
                  k++;
                  continue;
                }
                if (mat.kind === 'absorb') {
                  beta = mat.beta;
                  fc = mat.fcRel;
                }
              }
              betaSum += beta;
              if (beta > betaRep) {
                betaRep = beta;
                fcRep = fc;
              }
            }
          }
          K[idx] = k;
          C[idx] = this.coeff * r * r;
          S[idx] = sigma ? sigma[idx] * this.dt : 0;
          if (betaSum > 0) {
            const cl = c * r;
            const R = (rho * cl) / betaRep;
            const ks = R * 2 * Math.PI * ((fcRep * c) / this.params.dx);
            const aCoef = R + (ks * this.dt) / 2;
            const q = rho * cl * lam * r * (betaSum / betaRep);
            Q[idx] = q;
            Qa[idx] = q / aCoef;
            InvA[idx] = 1 / aCoef;
            Ks[idx] = ks;
          }
        }
    this.gK = K;
    this.gC = C;
    this.gS = S;
    this.gQ = Q;
    this.gQa = Qa;
    this.gInvA = InvA;
    this.gKs = Ks;
    this.gActive = active;
    if (!this.wallV || this.wallV.length !== n) {
      this.wallV = new Float32Array(n);
      this.wallX = new Float32Array(n);
    }
  }

  private stepGeneral(): void {
    if (!this.gK) this.buildGeneral();
    const { p, pPrev, pNext, nx, ny, nz, dims, dt } = this;
    const K = this.gK!;
    const C = this.gC!;
    const S = this.gS!;
    const Q = this.gQ!;
    const Qa = this.gQa!;
    const InvA = this.gInvA!;
    const Ks = this.gKs!;
    const act = this.gActive!;
    const V = this.wallV!;
    const X = this.wallX!;
    const sx = ny * nz;
    const sy = nz;
    const zs = dims === 3 ? nz : 1;
    for (let i = 0; i < nx; i++) {
      const iIn = i > 0 && i < nx - 1;
      for (let j = 0; j < ny; j++) {
        const jIn = j > 0 && j < ny - 1;
        for (let z = 0; z < zs; z++) {
          const idx = i * sx + j * sy + z;
          if (act[idx] === 0) {
            pNext[idx] = 0;
            continue;
          }
          const pc = p[idx];
          let acc: number;
          if (iIn && jIn && (dims === 2 || (z > 0 && z < nz - 1))) {
            acc = p[idx + sx] + p[idx - sx] + p[idx + sy] + p[idx - sy];
            if (dims === 3) acc += p[idx + 1] + p[idx - 1];
          } else {
            acc = 0;
            if (i > 0) acc += p[idx - sx];
            if (i < nx - 1) acc += p[idx + sx];
            if (j > 0) acc += p[idx - sy];
            if (j < ny - 1) acc += p[idx + sy];
            if (dims === 3) {
              if (z > 0) acc += p[idx - 1];
              if (z < nz - 1) acc += p[idx + 1];
            }
          }
          const lap = acc - K[idx] * pc;
          const sd = S[idx];
          let rhs = 2 * pc - pPrev[idx] + C[idx] * lap + sd * pPrev[idx];
          const q = Qa[idx];
          if (q !== 0) {
            rhs += -q * (0.5 * pc - Ks[idx] * X[idx]) + Q[idx] * V[idx];
            const nxt = rhs / (1 + 0.5 * q + sd);
            const vn = (0.5 * (nxt + pc) - Ks[idx] * X[idx]) * InvA[idx];
            X[idx] += dt * vn;
            V[idx] = vn;
            pNext[idx] = nxt;
          } else {
            pNext[idx] = rhs / (1 + sd);
          }
        }
      }
    }
    if (this.params.outer === 'mur') this.murEdges();
  }

  /** First-order Engquist-Majda (Mur) edges, same as physics.mur_edges. */
  private murEdges(): void {
    const { p, pNext, nx, ny, nz, dims } = this;
    const lam = Math.sqrt(this.coeff);
    const k = (lam - 1) / (lam + 1);
    const sh = [nx, ny, nz];
    const strides = [ny * nz, nz, 1];
    for (let a = 0; a < dims; a++) {
      const n = sh[a];
      for (const [face, inner] of [
        [0, 1],
        [n - 1, n - 2],
      ]) {
        // iterate over the face
        const b1 = (a + 1) % 3;
        const b2 = (a + 2) % 3;
        const n1 = dims === 2 && b1 === 2 ? 1 : sh[b1];
        const n2 = dims === 2 && b2 === 2 ? 1 : sh[b2];
        for (let u = 0; u < n1; u++)
          for (let w = 0; w < n2; w++) {
            const base = u * strides[b1] + w * strides[b2];
            const f = base + face * strides[a];
            const g = base + inner * strides[a];
            pNext[f] = p[g] + k * (pNext[g] - p[f]);
          }
      }
    }
  }
}

/** Fused 2D five-point leap-frog with p = 0 edges (Python parity path). */
export function kernel2dSoft(
  p: Float32Array,
  pp: Float32Array,
  pn: Float32Array,
  nx: number,
  ny: number,
  coeff: number,
): void {
  for (let i = 1; i < nx - 1; i++) {
    const row = i * ny;
    for (let j = 1; j < ny - 1; j++) {
      const idx = row + j;
      const c = p[idx];
      const lap = p[idx + ny] + p[idx - ny] + p[idx + 1] + p[idx - 1] - 4 * c;
      pn[idx] = 2 * c - pp[idx] + coeff * lap;
    }
  }
  for (let j = 0; j < ny; j++) {
    pn[j] = 0;
    pn[(nx - 1) * ny + j] = 0;
  }
  for (let i = 0; i < nx; i++) {
    pn[i * ny] = 0;
    pn[i * ny + ny - 1] = 0;
  }
}

/** Fused 3D seven-point leap-frog with p = 0 faces (Python parity path). */
export function kernel3dSoft(
  p: Float32Array,
  pp: Float32Array,
  pn: Float32Array,
  nx: number,
  ny: number,
  nz: number,
  coeff: number,
): void {
  const sx = ny * nz;
  for (let i = 1; i < nx - 1; i++) {
    for (let j = 1; j < ny - 1; j++) {
      const base = i * sx + j * nz;
      for (let k = 1; k < nz - 1; k++) {
        const idx = base + k;
        const c = p[idx];
        const lap = p[idx + sx] + p[idx - sx] + p[idx + nz] + p[idx - nz] + p[idx + 1] + p[idx - 1] - 6 * c;
        pn[idx] = 2 * c - pp[idx] + coeff * lap;
      }
    }
  }
  // Zero the six faces.
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++)
      for (let k = 0; k < nz; k++) {
        if (i === 0 || i === nx - 1 || j === 0 || j === ny - 1 || k === 0 || k === nz - 1) pn[i * sx + j * nz + k] = 0;
      }
}
