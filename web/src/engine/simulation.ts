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
 * Wall models (Phase 5; the default reproduces the Python engine):
 *  - 'soft'  : Dirichlet p = 0 (pressure-release, reflection -1). Default.
 *  - 'rigid' : Neumann dp/dn = 0 (reflection +1), staircase: an air cell's
 *              Laplacian only sums its air neighbours.
 *  - 'absorb': locally reacting impedance wall with normalised admittance
 *              beta (0 = rigid, 1 = matched at normal incidence).
 * Outer boundary: 'soft' | 'rigid' | 'absorb' (impedance) | 'pml' (graded
 * absorbing layer).
 */

import { evalWaveform, type WaveformSpec } from './waveforms';

export type Dims = 2 | 3;
export type WallKind = 'soft' | 'rigid' | 'absorb';
export type OuterKind = WallKind | 'pml';

export interface Material {
  kind: WallKind;
  /** Normalised specific admittance beta = rho c / Z (only for 'absorb'). */
  beta: number;
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
  /** PML thickness in cells when outer === 'pml'. */
  pmlCells: number;
}

export const DEFAULT_PARAMS: SimParams = {
  dims: 2,
  shape: [200, 200],
  c: 1,
  dx: 1,
  courant: 0.5,
  outer: 'soft',
  outerBeta: 1,
  pmlCells: 24,
};

export const PROBE_CAPACITY = 16384;

/** Pre-built materials indexed by material id stored per cell (0 = air). */
export const MATERIALS: Material[] = [
  { kind: 'soft', beta: 0 }, // 0 is unused for obstacle cells (air)
  { kind: 'soft', beta: 0 }, // 1: pressure-release (legacy default)
  { kind: 'rigid', beta: 0 }, // 2: rigid (concrete)
  { kind: 'absorb', beta: 0.1 }, // 3: hard plaster  (alpha ~ 0.33)
  { kind: 'absorb', beta: 0.3 }, // 4: wood panel    (alpha ~ 0.71)
  { kind: 'absorb', beta: 1.0 }, // 5: absorber      (alpha ~ 1 at normal incidence)
];

export const MATERIAL_NAMES = ['Air', 'Soft (p = 0)', 'Rigid', 'Plaster', 'Wood', 'Absorber'];

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
  private pmlSigma: Float32Array | null = null;

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
    if (this.params.outer === 'pml') this.buildPml();
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

  private buildPml(): void {
    const L = Math.max(1, Math.round(this.params.pmlCells));
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
    this.pmlSigma = sigma;
  }

  // Precomputed per-cell coefficients for the general kernel.
  private gK: Uint8Array | null = null; // neighbours contributing -p_c (air, soft wall)
  private gD: Float32Array | null = null; // impedance + absorbing-layer damping
  private gC: Float32Array | null = null; // (c(x) dt / dx)^2
  private gActive: Uint8Array | null = null; // 0 = held at p = 0

  /**
   * Build the general kernel's coefficient arrays.
   *
   * Wall cells always store p = 0, so for an air cell the staircase
   * Laplacian  sum_{air n}(p_n - p_c) + sum_{soft n}(0 - p_c)  equals
   *   sum_{all n} p_n  -  K p_c,   K = #(air or soft neighbours).
   * Rigid / impedance neighbours are excluded from K (zero normal
   * gradient: their ghost equals p_c). Each impedance face adds
   * lam*beta/2 to the damping factor D of the centred update
   *   (1 + D) p^{n+1} = 2 p^n + C * lap - (1 - D) p^{n-1},
   * and the absorbing layer adds sigma*dt (see docs/physics.md).
   */
  private buildGeneral(): void {
    const { nx, ny, nz, dims, material, n } = this;
    const outer = this.params.outer;
    const softOuter = outer === 'soft' || outer === 'pml';
    const outerBeta = outer === 'absorb' ? this.params.outerBeta : 0;
    const lam = Math.sqrt(this.coeff);
    const K = new Uint8Array(n);
    const D = new Float32Array(n);
    const C = new Float32Array(n);
    const active = new Uint8Array(n);
    const sigma = this.pmlSigma;
    const sx = ny * nz;
    const sy = nz;
    for (let i = 0; i < nx; i++)
      for (let j = 0; j < ny; j++)
        for (let z = 0; z < nz; z++) {
          const idx = i * sx + j * sy + z;
          const onEdge = i === 0 || i === nx - 1 || j === 0 || j === ny - 1 || (dims === 3 && (z === 0 || z === nz - 1));
          if (material[idx] !== 0 || (softOuter && onEdge)) continue;
          active[idx] = 1;
          const r = this.hasSpeed ? this.speed[idx] : 1;
          let k = 0;
          let beta = 0;
          for (let a = 0; a < dims; a++) {
            const stride = a === 0 ? sx : a === 1 ? sy : 1;
            const coord = a === 0 ? i : a === 1 ? j : z;
            const size = a === 0 ? nx : a === 1 ? ny : nz;
            for (let s2 = -1; s2 <= 1; s2 += 2) {
              const c2 = coord + s2;
              if (c2 < 0 || c2 >= size) {
                beta += outerBeta; // rigid/absorbing outer wall (soft edges are inactive)
                continue;
              }
              const m = material[idx + s2 * stride];
              if (m === 0) k++;
              else {
                const mat = MATERIALS[m] ?? MATERIALS[1];
                if (mat.kind === 'soft') k++;
                else if (mat.kind === 'absorb') beta += mat.beta;
              }
            }
          }
          K[idx] = k;
          C[idx] = this.coeff * r * r;
          D[idx] = 0.5 * lam * r * beta + (sigma ? sigma[idx] * this.dt : 0);
        }
    this.gK = K;
    this.gD = D;
    this.gC = C;
    this.gActive = active;
  }

  private stepGeneral(): void {
    if (!this.gK) this.buildGeneral();
    const { p, pPrev, pNext, nx, ny, nz, dims } = this;
    const K = this.gK!;
    const D = this.gD!;
    const C = this.gC!;
    const act = this.gActive!;
    const sx = ny * nz;
    const sy = nz;
    if (dims === 2) {
      for (let i = 1; i < nx - 1; i++) {
        const row = i * ny;
        for (let j = 1; j < ny - 1; j++) {
          const idx = row + j;
          if (act[idx] === 0) {
            pNext[idx] = 0;
            continue;
          }
          const pc = p[idx];
          const lap = p[idx + ny] + p[idx - ny] + p[idx + 1] + p[idx - 1] - K[idx] * pc;
          const d = D[idx];
          pNext[idx] = (2 * pc + C[idx] * lap - (1 - d) * pPrev[idx]) / (1 + d);
        }
      }
    } else {
      for (let i = 1; i < nx - 1; i++)
        for (let j = 1; j < ny - 1; j++) {
          const base = i * sx + j * sy;
          for (let z = 1; z < nz - 1; z++) {
            const idx = base + z;
            if (act[idx] === 0) {
              pNext[idx] = 0;
              continue;
            }
            const pc = p[idx];
            const lap = p[idx + sx] + p[idx - sx] + p[idx + sy] + p[idx - sy] + p[idx + 1] + p[idx - 1] - K[idx] * pc;
            const d = D[idx];
            pNext[idx] = (2 * pc + C[idx] * lap - (1 - d) * pPrev[idx]) / (1 + d);
          }
        }
    }
    // Domain-edge cells: same update with bounds-checked neighbours.
    const edge = (i: number, j: number, z: number) => {
      const idx = i * sx + j * sy + z;
      if (act[idx] === 0) {
        pNext[idx] = 0;
        return;
      }
      const pc = p[idx];
      let sum = 0;
      if (i > 0) sum += p[idx - sx];
      if (i < nx - 1) sum += p[idx + sx];
      if (j > 0) sum += p[idx - sy];
      if (j < ny - 1) sum += p[idx + sy];
      if (dims === 3) {
        if (z > 0) sum += p[idx - 1];
        if (z < nz - 1) sum += p[idx + 1];
      }
      const lap = sum - K[idx] * pc;
      const d = D[idx];
      pNext[idx] = (2 * pc + C[idx] * lap - (1 - d) * pPrev[idx]) / (1 + d);
    };
    const zs = dims === 3 ? nz : 1;
    for (let i = 0; i < nx; i++)
      for (let j = 0; j < ny; j++) {
        const iEdge = i === 0 || i === nx - 1 || j === 0 || j === ny - 1;
        if (iEdge) for (let z = 0; z < zs; z++) edge(i, j, z);
        else if (dims === 3) {
          edge(i, j, 0);
          edge(i, j, nz - 1);
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
