/**
 * Convolutional PML for the second-order wave equation: the browser twin
 * of src/acoustic_system/simulation/cpml.py (Pasalic & McGarry 2010).
 *
 *   p_tt = c^2 sum_a (d_a^2 p + d_a psi_a + zeta_a)
 *   psi_a^n  = b psi_a^{n-1}  + a (d_a p)^n                 (half points)
 *   zeta_a^n = b zeta_a^{n-1} + a (d_a^2 p + d_a psi_a)^n   (nodes)
 *   b = exp(-(d + alpha) dt),  a = d (b - 1) / (d + alpha)
 *   d(x) = d0 (x/L)^2,  d0 = -3 c ln R / (2 L),  alpha = alpha0 (1 - x/L)
 *
 * `update(p)` returns the extra term (without C) that the general kernel
 * adds to the Laplacian. Only the slabs next to CPML faces are touched.
 * Faces touching rigid/impedance cells carry no flux, as in the kernel
 * (`setWalls`).
 */

interface AxisState {
  an: Float32Array;
  bn: Float32Array;
  ah: Float32Array;
  bh: Float32Array;
  psi: Float32Array; // indexed by the lower node of each half point
  zeta: Float32Array;
  windows: [number, number][];
}

export class Cpml {
  readonly key: string;
  readonly ext: Float32Array;
  private readonly axes: (AxisState | null)[];
  private readonly shape: number[];
  private readonly strides: number[];
  private wall: Uint8Array | null = null;
  wallsFor: unknown = null;

  constructor(shape: number[], faces: boolean[], cells: number, c: number, dx: number, dt: number, r0 = 1e-5, alpha0?: number) {
    this.shape = shape.length === 2 ? [shape[0], shape[1], 1] : shape.slice();
    const dims = shape.length;
    const [nx, ny, nz] = this.shape;
    this.strides = [ny * nz, nz, 1];
    const n = nx * ny * nz;
    const L = Math.max(2, Math.round(cells));
    this.key = `${shape.join('x')}|${faces.join(',')}|${L}|${c}|${dx}|${dt}`;
    const d0 = (-3 * c * Math.log(r0)) / (2 * L * dx);
    const a0 = alpha0 ?? (Math.PI * c) / (20 * dx);
    // Float32 coefficients, as in the Python engine.
    const coef = (dist: number): [number, number] => {
      const f = Math.min(1, Math.max(0, dist / L));
      const d = d0 * f * f;
      if (!(d > 0)) return [0, 1];
      const al = a0 * (1 - f);
      const b = Math.fround(Math.exp(-(d + al) * dt));
      return [Math.fround((d * (b - 1)) / Math.max(d + al, 1e-30)), b];
    };
    this.axes = [];
    for (let a = 0; a < dims; a++) {
      const lo = faces[2 * a];
      const hi = faces[2 * a + 1];
      const na = this.shape[a];
      if (!lo && !hi) {
        this.axes.push(null);
        continue;
      }
      const an = new Float32Array(na);
      const bn = new Float32Array(na);
      const ah = new Float32Array(na);
      const bh = new Float32Array(na);
      for (let i = 0; i < na; i++) {
        let dn = 0;
        let dh = 0;
        if (lo) {
          dn = Math.max(dn, L - 1 - i);
          dh = Math.max(dh, L - 1 - (i + 0.5));
        }
        if (hi) {
          dn = Math.max(dn, i - (na - L));
          dh = Math.max(dh, i + 0.5 - (na - L));
        }
        [an[i], bn[i]] = coef((dn * L) / (L - 1));
        [ah[i], bh[i]] = coef((dh * L) / (L - 1));
      }
      const windows: [number, number][] = [];
      if (lo) windows.push([0, Math.min(na, L + 1)]);
      if (hi) windows.push([Math.max(0, na - L - 1), na]);
      this.axes.push({ an, bn, ah, bh, psi: new Float32Array(n), zeta: new Float32Array(n), windows });
    }
    this.ext = new Float32Array(n);
  }

  /** Profiles, memory variables and wall mask, for a device backend. */
  deviceState() {
    return {
      shape: this.shape,
      axes: this.axes.map((s) => (s ? { an: s.an, bn: s.bn, ah: s.ah, bh: s.bh, psi: s.psi, zeta: s.zeta } : null)),
      wall: this.wall,
    };
  }

  /** Mark no-flux cells (rigid/impedance walls); null clears them. */
  setWalls(wall: Uint8Array | null): void {
    this.wall = wall && wall.some((v) => v) ? wall : null;
  }

  reset(): void {
    for (const s of this.axes) {
      s?.psi.fill(0);
      s?.zeta.fill(0);
    }
    this.ext.fill(0);
  }

  /** Flat base index of every line along axis `a` (all other coordinates). */
  private linesOf(a: number): Int32Array {
    const others = [0, 1, 2].filter((b) => b !== a);
    const [b1, b2] = others;
    const n1 = this.shape[b1];
    const n2 = this.shape[b2];
    const out = new Int32Array(n1 * n2);
    let q = 0;
    for (let u = 0; u < n1; u++) for (let w = 0; w < n2; w++) out[q++] = u * this.strides[b1] + w * this.strides[b2];
    return out;
  }

  private lines: (Int32Array | null)[] = [];

  update(p: Float32Array): Float32Array {
    const ext = this.ext;
    const wall = this.wall;
    if (this.lines.length === 0) this.lines = this.axes.map((s, a) => (s ? this.linesOf(a) : null));
    for (let a = 0; a < this.axes.length; a++) {
      const s = this.axes[a];
      if (!s) continue;
      const st = this.strides[a];
      for (const [w0, w1] of s.windows) for (const base of this.lines[a]!) for (let k = w0; k < w1; k++) ext[base + k * st] = 0;
    }
    for (let a = 0; a < this.axes.length; a++) {
      const s = this.axes[a];
      if (!s) continue;
      const st = this.strides[a];
      const { psi, zeta, an, bn, ah, bh } = s;
      const lines = this.lines[a]!;
      for (const [w0, w1] of s.windows) {
        for (let q = 0; q < lines.length; q++) {
          const base = lines[q];
          // Half points h in [w0, w1 - 1): psi from the (masked) difference.
          for (let h = w0; h < w1 - 1; h++) {
            const idx = base + h * st;
            const d = wall !== null && (wall[idx] | wall[idx + st]) ? 0 : Math.fround(p[idx + st] - p[idx]);
            psi[idx] = bh[h] * psi[idx] + ah[h] * d;
          }
          // Interior nodes k in [w0 + 1, w1 - 1).
          for (let k = w0 + 1; k < w1 - 1; k++) {
            const idx = base + k * st;
            const up = wall !== null && (wall[idx] | wall[idx + st]) ? 0 : Math.fround(p[idx + st] - p[idx]);
            const dn = wall !== null && (wall[idx - st] | wall[idx]) ? 0 : Math.fround(p[idx] - p[idx - st]);
            const dpsi = Math.fround(psi[idx] - psi[idx - st]);
            zeta[idx] = bn[k] * zeta[idx] + an[k] * Math.fround(up - dn + dpsi);
            ext[idx] += dpsi + zeta[idx];
          }
        }
      }
    }
    return ext;
  }
}
