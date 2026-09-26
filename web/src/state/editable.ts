/** Mutable in-app scene representation (materials as a byte map, not RLE). */

import { DEFAULT_PARAMS, type DriverSpec, type ProbeSpec, type SimParams } from '../engine/simulation';
import { cellCount, decodeRle, encodeRle, type Scene } from '../engine/scene';

export interface EditableScene {
  name: string;
  description?: string;
  units: 'grid' | 'si';
  params: SimParams;
  materials: Uint8Array;
  /** Relative sound speed per cell (1 = nominal); null = uniform. */
  speed: Float32Array | null;
  drivers: DriverSpec[];
  probes: ProbeSpec[];
}

export function fromScene(s: Scene): EditableScene {
  const params: SimParams = { ...DEFAULT_PARAMS, ...s.params, shape: [...s.params.shape] };
  const n = cellCount(params);
  let speed: Float32Array | null = null;
  if (s.speed) {
    const q = decodeRle(s.speed, n);
    speed = new Float32Array(n);
    for (let i = 0; i < n; i++) speed[i] = q[i] === 0 ? 1 : q[i] / 100;
  }
  return {
    name: s.name,
    description: s.description,
    units: s.units,
    params,
    materials: decodeRle(s.materials, n),
    speed,
    drivers: s.drivers.map((d) => ({ ...d, pos: [...d.pos] })),
    probes: s.probes.map((p) => ({ ...p, pos: [...p.pos] })),
  };
}

export function toScene(e: EditableScene): Scene {
  let speed: string | undefined;
  if (e.speed) {
    const q = new Uint8Array(e.speed.length);
    for (let i = 0; i < q.length; i++) q[i] = e.speed[i] === 1 ? 0 : Math.max(1, Math.min(255, Math.round(e.speed[i] * 100)));
    speed = encodeRle(q);
  }
  return {
    version: 1,
    name: e.name,
    description: e.description,
    units: e.units,
    params: { ...e.params, shape: [...e.params.shape] },
    materials: encodeRle(e.materials),
    speed,
    drivers: e.drivers.map((d) => ({ ...d, pos: [...d.pos] })),
    probes: e.probes.map((p) => ({ ...p, pos: [...p.pos] })),
  };
}

function arraysEqual(a: ArrayLike<number> | null, b: ArrayLike<number> | null): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Deep equality of two scenes (used to skip no-op undo entries). */
export function sceneEquals(a: EditableScene, b: EditableScene): boolean {
  if (a === b) return true;
  return (
    a.name === b.name &&
    a.description === b.description &&
    a.units === b.units &&
    JSON.stringify(a.params) === JSON.stringify(b.params) &&
    JSON.stringify(a.drivers) === JSON.stringify(b.drivers) &&
    JSON.stringify(a.probes) === JSON.stringify(b.probes) &&
    arraysEqual(a.speed, b.speed) &&
    arraysEqual(a.materials, b.materials)
  );
}

export function cloneScene(e: EditableScene): EditableScene {
  return {
    ...e,
    params: { ...e.params, shape: [...e.params.shape] },
    materials: e.materials.slice(),
    speed: e.speed ? e.speed.slice() : null,
    drivers: e.drivers.map((d) => ({ ...d, pos: [...d.pos], waveform: { ...d.waveform } as DriverSpec['waveform'] })),
    probes: e.probes.map((p) => ({ ...p, pos: [...p.pos] })),
  };
}
