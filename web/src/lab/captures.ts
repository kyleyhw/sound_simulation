/**
 * Capture set (plan 9.7): measured impulse responses with their labels and
 * ground truth (tape-measured distances, room size), kept in the browser
 * and exported as JSON for scripts/eval_real_captures.py.
 */

export interface Capture {
  id: string;
  label: string;
  createdAt: string;
  sampleRate: number;
  /** Impulse response per microphone channel (base64 float32). */
  irs: string[];
  /** Tape-measured distance to the nearest reflector (m), if entered. */
  measuredDistance?: number;
  room?: { lx: number; ly: number; lz: number };
  position?: [number, number, number];
  orientation?: { alpha: number | null; beta: number | null; gamma: number | null };
  estimates: { firstEchoDistance?: number; t30?: number | null; t20?: number | null; latencyMs?: number };
  /** True when the IRs were equalised with the device calibration. */
  equalized?: boolean;
  device?: string;
}

const KEY = 'lab-captures-v1';

export function f32ToB64(a: Float32Array): string {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

export function b64ToF32(s: string): Float32Array {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function loadCaptures(): Capture[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '[]') as Capture[];
  } catch {
    return [];
  }
}

export function saveCaptures(c: Capture[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(c));
  } catch {
    /* storage full or unavailable: the export button still works */
  }
}

/** A fresh capture id, unique even for two saves in the same millisecond. */
export function newCaptureId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Append imported captures to the set. Rows are deleted by id, so an
 * imported capture whose id is already taken (re-importing an export, or a
 * file with repeated ids) gets a new id instead of aliasing the old row.
 */
export function mergeCaptures(existing: Capture[], imported: Capture[], makeId: () => string = newCaptureId): Capture[] {
  const taken = new Set(existing.map((c) => c.id));
  const out = [...existing];
  for (const c of imported) {
    let id = typeof c.id === 'string' && c.id ? c.id : makeId();
    while (taken.has(id)) id = makeId();
    taken.add(id);
    out.push({ ...c, id });
  }
  return out;
}

/** Parse the tape-measured distance field: '' is "not entered"; anything
 * else must be a positive, finite number of metres. */
export function parseDistance(text: string): { value: number | undefined; error: string | null } {
  const t = text.trim().replace(',', '.');
  if (t === '') return { value: undefined, error: null };
  const v = Number(t);
  if (!Number.isFinite(v) || v <= 0) return { value: undefined, error: 'Enter a positive distance in metres, e.g. 1.25' };
  return { value: v, error: null };
}

export function exportCaptures(c: Capture[]): Blob {
  return new Blob([JSON.stringify({ format: 'acoustic-sandbox-captures', version: 1, captures: c }, null, 1)], {
    type: 'application/json',
  });
}

/** Per-browser device calibration (plan 9.6): the direct-path response and latency. */
export interface StoredCalibration {
  sampleRate: number;
  response: string; // base64 float32
  preSamples: number;
  latencyMs: number;
  createdAt: string;
}

const CAL_KEY = 'lab-device-calibration-v1';

export function loadCalibration(): StoredCalibration | null {
  try {
    const raw = localStorage.getItem(CAL_KEY);
    return raw ? (JSON.parse(raw) as StoredCalibration) : null;
  } catch {
    return null;
  }
}

export function saveCalibration(c: StoredCalibration | null): void {
  try {
    if (c) localStorage.setItem(CAL_KEY, JSON.stringify(c));
    else localStorage.removeItem(CAL_KEY);
  } catch {
    /* unavailable storage: calibration lasts for this page only */
  }
}
