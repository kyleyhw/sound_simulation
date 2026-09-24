/**
 * WebGPU backend for the FDTD engine (plan 10.1, 10.2).
 *
 * The CPU `Simulation` stays the owner of the model: geometry, materials,
 * drivers and probes are edited there. This class mirrors its state on the
 * GPU, advances it in batches of steps with compute shaders, and reads the
 * field back after each batch, so the CPU object is always a consistent
 * snapshot. All physics comes from `Simulation.deviceState()`, the
 * precomputed coefficients of the general update
 *
 *   (1 + q/2 + s) p^{n+1} = 2p^n - p^{n-1} + C (L + ext) + s p^{n-1}
 *                           - q (p^n / 2 - K_s x^n) + Q v^{n-1/2}.
 *
 * That update also covers the fast path: every neighbour counts, and wall
 * and edge cells are held at 0. So one kernel reproduces every path of the
 * CPU engine, 2D and 3D, including impedance walls, the sponge, c(x),
 * the CPML (`ext`, two extra passes) and Mur edges (one pass per axis, in
 * the CPU's face order). Parity with the CPU engine is tested in
 * `tests/e2e/gpu.spec.ts`.
 *
 * Per step, the compute passes are:
 *   [cpml psi] [cpml zeta+ext] -> general -> [mur x, mur y, mur z]
 *   -> inject drivers -> record probes + tick -> [rms]
 * Driver values for the whole batch are computed on the CPU (the same
 * waveform code) and uploaded once per batch. A one-word counter written
 * by the tick pass indexes them.
 */

import type { Simulation } from './simulation';
import { evalWaveform } from './waveforms';

const WG = 256;

const HEADER = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, nz: u32, dims: u32,
  n: u32, cpmlAxes: u32, murMask: u32, np: u32,
  nd: u32, off0: u32, off1: u32, off2: u32,
  murK: f32, pad0: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(0) var<uniform> P: Params;

fn coords(idx: u32) -> vec3<u32> {
  let z = idx % P.nz;
  let j = (idx / P.nz) % P.ny;
  let i = idx / (P.nz * P.ny);
  return vec3<u32>(i, j, z);
}
fn stride(a: u32) -> u32 {
  if (a == 0u) { return P.ny * P.nz; }
  if (a == 1u) { return P.nz; }
  return 1u;
}
fn extent(a: u32) -> u32 {
  if (a == 0u) { return P.nx; }
  if (a == 1u) { return P.ny; }
  return P.nz;
}
fn profOff(a: u32) -> u32 {
  if (a == 0u) { return P.off0; }
  if (a == 1u) { return P.off1; }
  return P.off2;
}
fn flatId(g: vec3<u32>, n: vec3<u32>) -> u32 { return g.x + g.y * n.x * ${WG}u; }
`;

const GENERAL = /* wgsl */ `${HEADER}
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> pp: array<f32>;
@group(0) @binding(3) var<storage, read_write> pn: array<f32>;
@group(0) @binding(4) var<storage, read> coefA: array<vec4<f32>>;   // C, S, Q, Qa
@group(0) @binding(5) var<storage, read> coefB: array<vec4<f32>>;   // InvA, Ks, flags, dt
@group(0) @binding(6) var<storage, read_write> st: array<vec2<f32>>; // V, X
@group(0) @binding(7) var<storage, read> ext: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let idx = flatId(gid, nw);
  if (idx >= P.n) { return; }
  let b = coefB[idx];
  let flags = u32(b.z);
  if ((flags & 1u) == 0u) { pn[idx] = 0.0; return; }
  let K = f32((flags >> 1u) & 15u);
  let c = coords(idx);
  let pc = p[idx];
  var acc = 0.0;
  if (c.x > 0u) { acc += p[idx - stride(0u)]; }
  if (c.x < P.nx - 1u) { acc += p[idx + stride(0u)]; }
  if (c.y > 0u) { acc += p[idx - stride(1u)]; }
  if (c.y < P.ny - 1u) { acc += p[idx + stride(1u)]; }
  if (P.dims == 3u) {
    if (c.z > 0u) { acc += p[idx - 1u]; }
    if (c.z < P.nz - 1u) { acc += p[idx + 1u]; }
  }
  var lap = acc - K * pc;
  if (P.cpmlAxes != 0u) { lap += ext[idx]; }
  let a = coefA[idx];
  let sd = a.y;
  var rhs = 2.0 * pc - pp[idx] + a.x * lap + sd * pp[idx];
  let q = a.w;
  if (q != 0.0) {
    let s = st[idx];
    rhs += -q * (0.5 * pc - b.y * s.y) + a.z * s.x;
    let nxt = rhs / (1.0 + 0.5 * q + sd);
    let vn = (0.5 * (nxt + pc) - b.y * s.y) * b.x;
    st[idx] = vec2<f32>(vn, s.y + b.w * vn);
    pn[idx] = nxt;
  } else {
    pn[idx] = rhs / (1.0 + sd);
  }
}
`;

const CPML_PSI = /* wgsl */ `${HEADER}
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read_write> psi: array<f32>;
@group(0) @binding(3) var<storage, read> coefB: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> prof: array<f32>;

fn wallAt(q: u32) -> bool { return ((u32(coefB[q].z) >> 5u) & 1u) == 1u; }

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let idx = flatId(gid, nw);
  if (idx >= P.n) { return; }
  let c = coords(idx);
  for (var a = 0u; a < P.dims; a++) {
    if (((P.cpmlAxes >> a) & 1u) == 0u) { continue; }
    let h = select(select(c.z, c.y, a == 1u), c.x, a == 0u);
    let na = extent(a);
    if (h + 1u >= na) { continue; }
    let s = stride(a);
    var d = p[idx + s] - p[idx];
    if (wallAt(idx) || wallAt(idx + s)) { d = 0.0; }
    let o = profOff(a);
    let ah = prof[o + 2u * na + h];
    let bh = prof[o + 3u * na + h];
    let k = a * P.n + idx;
    psi[k] = bh * psi[k] + ah * d;
  }
}
`;

const CPML_ZETA = /* wgsl */ `${HEADER}
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> psi: array<f32>;
@group(0) @binding(3) var<storage, read_write> zeta: array<f32>;
@group(0) @binding(4) var<storage, read_write> ext: array<f32>;
@group(0) @binding(5) var<storage, read> coefB: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> prof: array<f32>;

fn wallAt(q: u32) -> bool { return ((u32(coefB[q].z) >> 5u) & 1u) == 1u; }

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let idx = flatId(gid, nw);
  if (idx >= P.n) { return; }
  let c = coords(idx);
  var e = 0.0;
  for (var a = 0u; a < P.dims; a++) {
    if (((P.cpmlAxes >> a) & 1u) == 0u) { continue; }
    let i = select(select(c.z, c.y, a == 1u), c.x, a == 0u);
    let na = extent(a);
    if (i == 0u || i + 1u >= na) { continue; }
    let s = stride(a);
    var up = p[idx + s] - p[idx];
    if (wallAt(idx) || wallAt(idx + s)) { up = 0.0; }
    var dn = p[idx] - p[idx - s];
    if (wallAt(idx - s) || wallAt(idx)) { dn = 0.0; }
    let k = a * P.n + idx;
    let dpsi = psi[k] - psi[k - s];
    let o = profOff(a);
    let an = prof[o + i];
    let bn = prof[o + na + i];
    let z = bn * zeta[k] + an * (up - dn + dpsi);
    zeta[k] = z;
    e += dpsi + z;
  }
  ext[idx] = e;
}
`;

// One pass per axis (pipeline constant AXIS), in the CPU's face order, so the
// corner cells come out the same as physics.mur_edges / murEdges().
const MUR = /* wgsl */ `${HEADER}
override AXIS: u32 = 0u;
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read_write> pn: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let idx = flatId(gid, nw);
  if (idx >= P.n) { return; }
  let c = coords(idx);
  let i = select(select(c.z, c.y, AXIS == 1u), c.x, AXIS == 0u);
  let na = extent(AXIS);
  let s = stride(AXIS);
  // High face after low face, as in the CPU loop.
  if (((P.murMask >> (2u * AXIS + 1u)) & 1u) == 1u && i == na - 1u) {
    pn[idx] = p[idx - s] + P.murK * (pn[idx - s] - p[idx]);
  } else if (((P.murMask >> (2u * AXIS)) & 1u) == 1u && i == 0u) {
    pn[idx] = p[idx + s] + P.murK * (pn[idx + s] - p[idx]);
  }
}
`;

const INJECT = /* wgsl */ `${HEADER}
@group(0) @binding(1) var<storage, read_write> pn: array<f32>;
@group(0) @binding(2) var<storage, read> drvIdx: array<u32>;
@group(0) @binding(3) var<storage, read> drvVal: array<f32>;
@group(0) @binding(4) var<storage, read> counter: array<u32>;

@compute @workgroup_size(1)
fn main() {
  let k = counter[0];
  for (var j = 0u; j < P.nd; j++) { pn[drvIdx[j]] += drvVal[k * P.nd + j]; }
}
`;

const PROBE_TICK = /* wgsl */ `${HEADER}
@group(0) @binding(1) var<storage, read> pn: array<f32>;
@group(0) @binding(2) var<storage, read> probeIdx: array<u32>;
@group(0) @binding(3) var<storage, read_write> probeOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> counter: array<u32>;

@compute @workgroup_size(1)
fn main() {
  let k = counter[0];
  for (var j = 0u; j < P.np; j++) { probeOut[k * P.np + j] = pn[probeIdx[j]]; }
  counter[0] = k + 1u;
}
`;

const RMS = /* wgsl */ `${HEADER}
@group(0) @binding(1) var<storage, read> pn: array<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let idx = flatId(gid, nw);
  if (idx >= P.n) { return; }
  acc[idx] += pn[idx] * pn[idx];
}
`;

export async function gpuAvailable(): Promise<boolean> {
  try {
    return !!(typeof navigator !== 'undefined' && navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

type Bufs = Record<string, GPUBuffer>;

export class GpuStepper {
  readonly device: GPUDevice;
  readonly sim: Simulation;
  private bufs: Bufs = {};
  private p3: GPUBuffer[] = [];
  private rot = 0; // p = p3[rot], pp = p3[(rot + 2) % 3], pn = p3[(rot + 1) % 3]
  private pipes: Record<string, GPUComputePipeline> = {};
  private batchCap = 0;
  private geometryKey: unknown = null;
  private cpmlAxes = 0;
  private murMask = 0;
  private dispatch: [number, number] = [1, 1];
  busy = false;
  needUpload = false;
  needRmsZero = false;

  private constructor(device: GPUDevice, sim: Simulation) {
    this.device = device;
    this.sim = sim;
  }

  static async create(sim: Simulation): Promise<GpuStepper> {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available in this browser');
    const need = sim.n * 4 * 3;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, Math.max(need, 128 << 20)),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, Math.max(need, 256 << 20)),
      },
    });
    const g = new GpuStepper(device, sim);
    g.build();
    g.upload();
    sim.onReset = () => (g.needUpload = true);
    sim.onResetAccumulators = () => (g.needRmsZero = true);
    return g;
  }

  private buffer(name: string, size: number, usage: number): GPUBuffer {
    this.bufs[name]?.destroy();
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
    this.bufs[name] = b;
    return b;
  }

  private pipeline(name: string, code: string, constants?: Record<string, number>): GPUComputePipeline {
    const key = constants ? `${name}:${JSON.stringify(constants)}` : name;
    if (!this.pipes[key]) {
      this.pipes[key] = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: this.device.createShaderModule({ code }), entryPoint: 'main', constants },
      });
    }
    return this.pipes[key];
  }

  /** Allocate buffers for the current grid (called once; geometry uploads are separate). */
  private build(): void {
    const { n } = this.sim;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.p3 = [0, 1, 2].map((k) => this.buffer(`p${k}`, n * 4, S));
    this.buffer('coefA', n * 16, S);
    this.buffer('coefB', n * 16, S);
    this.buffer('state', n * 8, S);
    this.buffer('ext', n * 4, S);
    this.buffer('psi', 3 * n * 4, S);
    this.buffer('zeta', 3 * n * 4, S);
    this.buffer('counter', 16, S);
    this.buffer('params', 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const groups = Math.ceil(n / WG);
    // 2D dispatch keeps each dimension under the 65535 workgroup limit.
    const gx = Math.min(groups, 65535);
    this.dispatch = [gx, Math.ceil(groups / gx)];
  }

  /** Push the CPU simulation's complete state and coefficients to the device. */
  upload(): void {
    const sim = this.sim;
    const ds = sim.deviceState();
    const n = sim.n;
    const q = this.device.queue;
    const A = new Float32Array(4 * n);
    const B = new Float32Array(4 * n);
    const wall = ds.cpml?.wall ?? null;
    for (let i = 0; i < n; i++) {
      A[4 * i] = ds.C[i];
      A[4 * i + 1] = ds.S[i];
      A[4 * i + 2] = ds.Q[i];
      A[4 * i + 3] = ds.Qa[i];
      B[4 * i] = ds.InvA[i];
      B[4 * i + 1] = ds.Ks[i];
      B[4 * i + 2] = ds.active[i] + 2 * ds.K[i] + 32 * (wall && wall[i] ? 1 : 0);
      B[4 * i + 3] = sim.dt;
    }
    q.writeBuffer(this.bufs.coefA, 0, A);
    q.writeBuffer(this.bufs.coefB, 0, B);
    const st = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
      st[2 * i] = ds.V[i];
      st[2 * i + 1] = ds.X[i];
    }
    q.writeBuffer(this.bufs.state, 0, st);
    // Rotation 0: p = p3[0], pp = p3[2], pn = p3[1].
    this.rot = 0;
    q.writeBuffer(this.p3[0], 0, sim.p);
    q.writeBuffer(this.p3[2], 0, sim.pPrev);
    // CPML profiles and memory.
    const shape = [sim.nx, sim.ny, sim.nz];
    const offs = [0, 0, 0];
    this.cpmlAxes = 0;
    const psi = new Float32Array(3 * n);
    const zeta = new Float32Array(3 * n);
    let prof = new Float32Array(4);
    if (ds.cpml) {
      let total = 0;
      for (let a = 0; a < sim.dims; a++) {
        offs[a] = total;
        if (ds.cpml.axes[a]) total += 4 * shape[a];
      }
      prof = new Float32Array(Math.max(4, total));
      ds.cpml.axes.forEach((ax, a) => {
        if (!ax) return;
        this.cpmlAxes |= 1 << a;
        const o = offs[a];
        prof.set(ax.an, o);
        prof.set(ax.bn, o + shape[a]);
        prof.set(ax.ah, o + 2 * shape[a]);
        prof.set(ax.bh, o + 3 * shape[a]);
        psi.set(ax.psi, a * n);
        zeta.set(ax.zeta, a * n);
      });
      q.writeBuffer(this.bufs.ext, 0, new Float32Array(n));
    }
    this.buffer('prof', prof.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    q.writeBuffer(this.bufs.prof, 0, prof);
    q.writeBuffer(this.bufs.psi, 0, psi);
    q.writeBuffer(this.bufs.zeta, 0, zeta);
    this.murMask = ds.murFaces.reduce((m, f, k) => (f ? m | (1 << k) : m), 0);
    const P = new ArrayBuffer(64);
    const u = new Uint32Array(P);
    const f = new Float32Array(P);
    u.set([sim.nx, sim.ny, sim.nz, sim.dims, n, this.cpmlAxes, this.murMask, 0, 0, offs[0], offs[1], offs[2]]);
    f[12] = (ds.lam - 1) / (ds.lam + 1);
    this.paramsHost = P;
    q.writeBuffer(this.bufs.params, 0, P);
    this.geometryKey = ds.geometryKey;
    if (sim.rmsAccum) {
      this.buffer('rms', n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      q.writeBuffer(this.bufs.rms, 0, sim.rmsAccum);
    } else {
      this.bufs.rms?.destroy();
      delete this.bufs.rms;
    }
  }

  private paramsHost = new ArrayBuffer(64);

  private group(pipe: GPUComputePipeline, buffers: GPUBuffer[]): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.bufs.params } }, ...buffers.map((b, k) => ({ binding: k + 1, resource: { buffer: b } }))],
    });
  }

  /** Geometry, materials or boundaries changed on the CPU: sync, then re-upload. */
  get stale(): boolean {
    return this.sim.deviceState().geometryKey !== this.geometryKey || !!this.sim.rmsAccum !== !!this.bufs.rms;
  }

  /**
   * Advance `steps` steps on the device, then read back p and p_prev (and
   * the probe samples) into the CPU simulation. With `full`, also read the
   * impedance-branch and CPML memory, so the CPU engine can continue exactly.
   */
  async run(steps: number, full = false): Promise<void> {
    if (this.needUpload) {
      // The CPU state was reset: it is authoritative again.
      this.needUpload = false;
      this.needRmsZero = false;
      this.upload();
    }
    if (this.needRmsZero && this.bufs.rms) {
      this.needRmsZero = false;
      this.device.queue.writeBuffer(this.bufs.rms, 0, new Float32Array(this.sim.n));
    }
    if (this.stale) {
      await this.advance(0, true);
      this.upload();
    }
    await this.advance(steps, full);
  }

  private async advance(steps: number, full: boolean): Promise<void> {
    const sim = this.sim;
    const n = sim.n;
    const drivers = sim.drivers.filter((d) => d.enabled);
    const nd = drivers.length;
    const np = sim.probes.length;
    // Driver values for the batch, with the CPU engine's time accumulation.
    const vals = new Float32Array(Math.max(1, steps * nd));
    let t = sim.time;
    for (let k = 0; k < steps; k++) {
      for (let j = 0; j < nd; j++) {
        const d = drivers[j];
        vals[k * nd + j] = evalWaveform(d.waveform, t - (d.delay ?? 0)) * (d.gain ?? 1);
      }
      t = t + sim.dt;
    }
    const U = GPUBufferUsage;
    if (steps > this.batchCap || !this.bufs.drvVal) {
      this.batchCap = Math.max(steps, 64);
      this.buffer('drvVal', this.batchCap * Math.max(1, nd) * 4 * 4, U.STORAGE | U.COPY_DST);
      this.buffer('probeOut', this.batchCap * Math.max(1, np) * 4 * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    }
    if ((this.bufs.drvVal.size as number) < vals.byteLength) this.buffer('drvVal', vals.byteLength, U.STORAGE | U.COPY_DST);
    if ((this.bufs.probeOut.size as number) < steps * Math.max(1, np) * 4) this.buffer('probeOut', steps * Math.max(1, np) * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    const q = this.device.queue;
    q.writeBuffer(this.bufs.drvVal, 0, vals);
    this.buffer('drvIdx', Math.max(1, nd) * 4, U.STORAGE | U.COPY_DST);
    q.writeBuffer(this.bufs.drvIdx, 0, new Uint32Array(nd ? drivers.map((d) => sim.index(d.pos)) : [0]));
    this.buffer('probeIdx', Math.max(1, np) * 4, U.STORAGE | U.COPY_DST);
    q.writeBuffer(this.bufs.probeIdx, 0, new Uint32Array(np ? sim.probes.map((p) => sim.index(p.pos)) : [0]));
    q.writeBuffer(this.bufs.counter, 0, new Uint32Array([0, 0, 0, 0]));
    const u = new Uint32Array(this.paramsHost);
    u[7] = np;
    u[8] = nd;
    q.writeBuffer(this.bufs.params, 0, this.paramsHost);

    const pipes = {
      general: this.pipeline('general', GENERAL),
      psi: this.pipeline('psi', CPML_PSI),
      zeta: this.pipeline('zeta', CPML_ZETA),
      inject: this.pipeline('inject', INJECT),
      probe: this.pipeline('probe', PROBE_TICK),
      rms: this.pipeline('rms', RMS),
      mur: [0, 1, 2].map((a) => this.pipeline('mur', MUR, { AXIS: a })),
    };
    // Bind groups for the three buffer rotations.
    const rots = [0, 1, 2].map((r) => {
      const p = this.p3[r];
      const pp = this.p3[(r + 2) % 3];
      const pn = this.p3[(r + 1) % 3];
      const B = this.bufs;
      return {
        general: this.group(pipes.general, [p, pp, pn, B.coefA, B.coefB, B.state, B.ext]),
        psi: this.cpmlAxes ? this.group(pipes.psi, [p, B.psi, B.coefB, B.prof]) : null,
        zeta: this.cpmlAxes ? this.group(pipes.zeta, [p, B.psi, B.zeta, B.ext, B.coefB, B.prof]) : null,
        mur: pipes.mur.map((m, a) => (this.murMask >> (2 * a)) & 3 ? this.group(m, [p, pn]) : null),
        inject: nd ? this.group(pipes.inject, [pn, B.drvIdx, B.drvVal, B.counter]) : null,
        probe: this.group(pipes.probe, [pn, B.probeIdx, B.probeOut, B.counter]),
        rms: B.rms ? this.group(pipes.rms, [pn, B.rms]) : null,
      };
    });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const [dx, dy] = this.dispatch;
    for (let k = 0; k < steps; k++) {
      const g = rots[this.rot];
      if (g.psi && g.zeta) {
        pass.setPipeline(pipes.psi);
        pass.setBindGroup(0, g.psi);
        pass.dispatchWorkgroups(dx, dy);
        pass.setPipeline(pipes.zeta);
        pass.setBindGroup(0, g.zeta);
        pass.dispatchWorkgroups(dx, dy);
      }
      pass.setPipeline(pipes.general);
      pass.setBindGroup(0, g.general);
      pass.dispatchWorkgroups(dx, dy);
      g.mur.forEach((m, a) => {
        if (!m) return;
        pass.setPipeline(pipes.mur[a]);
        pass.setBindGroup(0, m);
        pass.dispatchWorkgroups(dx, dy);
      });
      if (g.inject) {
        pass.setPipeline(pipes.inject);
        pass.setBindGroup(0, g.inject);
        pass.dispatchWorkgroups(1);
      }
      pass.setPipeline(pipes.probe);
      pass.setBindGroup(0, g.probe);
      pass.dispatchWorkgroups(1);
      if (g.rms) {
        pass.setPipeline(pipes.rms);
        pass.setBindGroup(0, g.rms);
        pass.dispatchWorkgroups(dx, dy);
      }
      this.rot = (this.rot + 1) % 3;
    }
    pass.end();
    // Read back the field (p, p_prev) and the probe samples.
    const reads: [GPUBuffer, number][] = [
      [this.p3[this.rot], n * 4],
      [this.p3[(this.rot + 2) % 3], n * 4],
      [this.bufs.probeOut, Math.max(1, steps * np) * 4],
    ];
    if (this.bufs.rms) reads.push([this.bufs.rms, n * 4]);
    if (full) reads.push([this.bufs.state, n * 8], [this.bufs.psi, 3 * n * 4], [this.bufs.zeta, 3 * n * 4]);
    const staging = reads.map(([src, size]) => {
      const dst = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(src, 0, dst, 0, size);
      return dst;
    });
    q.submit([enc.finish()]);
    await Promise.all(staging.map((b) => b.mapAsync(GPUMapMode.READ)));
    const out = staging.map((b) => new Float32Array(b.getMappedRange().slice(0)));
    staging.forEach((b) => b.destroy());
    let time = sim.time;
    for (let k = 0; k < steps; k++) time = time + sim.dt;
    const imp: Parameters<Simulation['importState']>[0] = { p: out[0], pPrev: out[1], time, step_count: sim.step_count + steps };
    let r = 3;
    if (this.bufs.rms && sim.rmsAccum) {
      sim.rmsAccum.set(out[r++]);
      sim.rmsCount += steps;
    }
    if (full) {
      const st = out[r++];
      const V = new Float32Array(n);
      const X = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        V[i] = st[2 * i];
        X[i] = st[2 * i + 1];
      }
      imp.V = V;
      imp.X = X;
      const psi = out[r++];
      const zeta = out[r++];
      const ds = sim.deviceState();
      const mask = ds.cpml ? ds.cpml.axes.reduce((m, ax, a) => (ax ? m | (1 << a) : m), 0) : 0;
      // Restore the CPML memory only if the layer is the one on the device.
      if (mask === this.cpmlAxes) ds.cpml?.axes.forEach((ax, a) => {
        if (!ax) return;
        ax.psi.set(psi.subarray(a * n, (a + 1) * n));
        ax.zeta.set(zeta.subarray(a * n, (a + 1) * n));
      });
    }
    sim.importState(imp);
    if (np) sim.pushProbeSamples(out[2], steps);
  }

  /** Read the complete device state back into the CPU simulation. */
  async syncState(): Promise<void> {
    await this.advance(0, true);
  }

  destroy(): void {
    if (this.sim.onReset) this.sim.onReset = undefined;
    if (this.sim.onResetAccumulators) this.sim.onResetAccumulators = undefined;
    Object.values(this.bufs).forEach((b) => b.destroy());
    this.p3.forEach((b) => b.destroy());
    this.device.destroy();
  }
}
