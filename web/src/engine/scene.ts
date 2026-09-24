/**
 * Serializable scene description: everything needed to rebuild a
 * Simulation (geometry, sources, probes, parameters). Used for save/load,
 * shareable URLs, undo/redo snapshots and presets.
 */

import { DEFAULT_PARAMS, type DriverSpec, type ProbeSpec, type SimParams, Simulation } from './simulation';

export interface Scene {
  version: 1;
  name: string;
  params: SimParams;
  /** Run-length encoded material map (see encodeRle). */
  materials: string;
  /** Optional run-length encoded sound-speed map, quantised to 1/100. */
  speed?: string;
  drivers: DriverSpec[];
  probes: ProbeSpec[];
  /** Units: 'grid' (c = dx = 1) or 'si' (metres, seconds). */
  units: 'grid' | 'si';
  description?: string;
}

let idCounter = 0;
export function newId(prefix: string): string {
  idCounter += 1;
  return `${prefix}${Date.now().toString(36)}${idCounter.toString(36)}`;
}

/** RLE: "value:count,value:count,..." over the flat byte array. */
export function encodeRle(data: Uint8Array): string {
  if (data.length === 0) return '';
  const parts: string[] = [];
  let cur = data[0];
  let run = 1;
  for (let i = 1; i < data.length; i++) {
    if (data[i] === cur) run++;
    else {
      parts.push(`${cur}:${run}`);
      cur = data[i];
      run = 1;
    }
  }
  parts.push(`${cur}:${run}`);
  return parts.join(',');
}

export function decodeRle(s: string, n: number): Uint8Array {
  const out = new Uint8Array(n);
  if (!s) return out;
  let pos = 0;
  for (const part of s.split(',')) {
    const [v, c] = part.split(':').map(Number);
    if (!Number.isFinite(v) || !Number.isFinite(c) || c < 0) throw new Error('bad RLE');
    out.fill(v, pos, Math.min(n, pos + c));
    pos += c;
  }
  if (pos !== n) throw new Error(`RLE length ${pos} != ${n}`);
  return out;
}

export function emptyScene(params: Partial<SimParams> = {}, name = 'Untitled'): Scene {
  const p: SimParams = { ...DEFAULT_PARAMS, ...params, shape: [...(params.shape ?? DEFAULT_PARAMS.shape)] };
  const n = p.shape.reduce((a, b) => a * b, 1);
  return {
    version: 1,
    name,
    params: p,
    materials: encodeRle(new Uint8Array(n)),
    drivers: [],
    probes: [],
    units: 'grid',
  };
}

export function cellCount(params: SimParams): number {
  return params.shape.reduce((a, b) => a * b, 1);
}

/** Build a fresh Simulation from a scene. */
export function buildSimulation(scene: Scene): Simulation {
  const sim = new Simulation(scene.params);
  sim.setMaterialMap(decodeRle(scene.materials, sim.n));
  if (scene.speed) {
    const q = decodeRle(scene.speed, sim.n);
    const f = new Float32Array(sim.n);
    for (let i = 0; i < sim.n; i++) f[i] = q[i] === 0 ? 1 : q[i] / 100;
    sim.setSpeedMap(f);
  }
  sim.setDrivers(scene.drivers);
  sim.setProbes(scene.probes);
  return sim;
}

export function validateScene(x: unknown): Scene {
  const s = x as Scene;
  if (!s || s.version !== 1 || !s.params || !Array.isArray(s.params.shape)) throw new Error('not a scene file');
  const dims = s.params.dims;
  if (dims !== 2 && dims !== 3) throw new Error('dims must be 2 or 3');
  if (s.params.shape.length !== dims || s.params.shape.some((v) => !Number.isInteger(v) || v < 8 || v > 1024))
    throw new Error('bad grid shape');
  if (cellCount(s.params) > 8_000_000) throw new Error('grid too large');
  for (const k of ['c', 'dx', 'courant'] as const) {
    if (!(Number.isFinite(s.params[k]) && s.params[k] > 0)) throw new Error(`bad ${k}`);
  }
  decodeRle(s.materials, cellCount(s.params)); // throws if inconsistent
  if (!Array.isArray(s.drivers) || !Array.isArray(s.probes)) throw new Error('bad drivers/probes');
  return {
    ...s,
    params: { ...DEFAULT_PARAMS, ...s.params },
    units: s.units === 'si' ? 'si' : 'grid',
  };
}

// ---------------------------------------------------------------------- //
// Shareable URL encoding: JSON -> deflate-raw -> base64url
// ---------------------------------------------------------------------- //

function toBase64Url(bytes: Uint8Array): string {
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(s: string): Uint8Array {
  const b64 = s.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((s.length + 3) % 4);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function pipe(data: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const blob = new Blob([data as BlobPart]);
  const out = await new Response(blob.stream().pipeThrough(stream)).arrayBuffer();
  return new Uint8Array(out);
}

export async function encodeSceneUrl(scene: Scene): Promise<string> {
  const json = new TextEncoder().encode(JSON.stringify(scene));
  return toBase64Url(await pipe(json, new CompressionStream('deflate-raw')));
}

export async function decodeSceneUrl(token: string): Promise<Scene> {
  const bytes = await pipe(fromBase64Url(token), new DecompressionStream('deflate-raw'));
  return validateScene(JSON.parse(new TextDecoder().decode(bytes)));
}
