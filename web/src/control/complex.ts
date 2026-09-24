/** Small dense complex linear algebra for array design (N <= ~32). */

export type Cx = [number, number];
export type CMat = Cx[][];
export type CVec = Cx[];

export const cx = (re: number, im = 0): Cx => [re, im];
export const cmul = (a: Cx, b: Cx): Cx => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
export const cadd = (a: Cx, b: Cx): Cx => [a[0] + b[0], a[1] + b[1]];
export const csub = (a: Cx, b: Cx): Cx => [a[0] - b[0], a[1] - b[1]];
export const cconj = (a: Cx): Cx => [a[0], -a[1]];
export const cabs2 = (a: Cx): number => a[0] * a[0] + a[1] * a[1];
export const cscale = (a: Cx, s: number): Cx => [a[0] * s, a[1] * s];
export const cdiv = (a: Cx, b: Cx): Cx => {
  const d = cabs2(b);
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
};
export const cexp = (phi: number): Cx => [Math.cos(phi), Math.sin(phi)];

export function matvec(A: CMat, x: CVec): CVec {
  return A.map((row) => row.reduce<Cx>((acc, a, j) => cadd(acc, cmul(a, x[j])), [0, 0]));
}

/** Correlation matrix R = H^H H / M for H with M rows (points) and N columns (sources). */
export function gram(H: CMat): CMat {
  const n = H[0]?.length ?? 0;
  const m = Math.max(1, H.length);
  const R: CMat = Array.from({ length: n }, () => Array.from({ length: n }, () => cx(0)));
  for (const row of H)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) R[i][j] = cadd(R[i][j], cmul(cconj(row[i]), row[j]));
  return R.map((r) => r.map((v) => cscale(v, 1 / m)));
}

export function addDiag(A: CMat, d: number): CMat {
  return A.map((r, i) => r.map((v, j) => (i === j ? cadd(v, cx(d)) : v)));
}

export function trace(A: CMat): number {
  return A.reduce((s, r, i) => s + r[i][0], 0);
}

/** Solve A x = b by Gaussian elimination with partial pivoting. */
export function solve(A: CMat, b: CVec): CVec {
  const n = A.length;
  const M = A.map((r, i) => [...r.map((v) => [...v] as Cx), [...b[i]] as Cx]);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (cabs2(M[i][k]) > cabs2(M[p][k])) p = i;
    [M[k], M[p]] = [M[p], M[k]];
    if (cabs2(M[k][k]) === 0) throw new Error('singular matrix');
    for (let i = k + 1; i < n; i++) {
      const f = cdiv(M[i][k], M[k][k]);
      for (let j = k; j <= n; j++) M[i][j] = csub(M[i][j], cmul(f, M[k][j]));
    }
  }
  const x: CVec = Array.from({ length: n }, () => cx(0));
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s = csub(s, cmul(M[i][j], x[j]));
    x[i] = cdiv(s, M[i][i]);
  }
  return x;
}

export function norm2(x: CVec): number {
  return x.reduce((s, v) => s + cabs2(v), 0);
}

/**
 * Principal generalised eigenvector of (Rb, Rd): maximises the Rayleigh
 * quotient w^H Rb w / w^H Rd w. Power iteration on Rd^{-1} Rb (Rd Hermitian
 * positive definite after regularisation).
 */
export function principalGeneralized(Rb: CMat, Rd: CMat, iters = 300): CVec {
  const n = Rb.length;
  let w: CVec = Array.from({ length: n }, (_, i) => cexp(0.3 * i));
  for (let k = 0; k < iters; k++) {
    const y = solve(Rd, matvec(Rb, w));
    const s = Math.sqrt(norm2(y)) || 1;
    w = y.map((v) => cscale(v, 1 / s));
  }
  return w;
}
