/**
 * Echo vision (#/echo): the sensing device, as data. A device is where its
 * speakers and microphones sit and what it emits, one emission (a "ping")
 * at a time. The page, the worker and the player read everything from this
 * description, so another device (for example two speakers with coded,
 * simultaneous emissions) only needs a new `Device` and, for the picture, a
 * matching imaging routine in the worker.
 */

import type { WaveformSpec } from '../engine/waveforms';
import { demoScenario } from '../loop/scenarios';
import { N } from './room';

/** One emission: the speakers that fire together, each with its waveform. */
export interface Emission {
  drivers: { speaker: number; waveform: WaveformSpec }[];
}

export interface Device {
  id: string;
  /** Speaker positions (grid cells, [row, col]). */
  speakers: number[][];
  /** Microphone positions (grid cells). */
  mics: number[][];
  /** Emissions in firing order; every microphone records each one. */
  emissions: Emission[];
  /** Pulse centre frequency (cycles per unit time) and delay of its peak (time units). */
  f0: number;
  delay: number;
}

/** Ricker ping centre frequency, as in the loop's sensing. */
export const BAR_F0 = 0.08;

/**
 * The closed loop's 8-element bar: every element is a speaker and a
 * microphone, and the speakers click one at a time (loop pingRecordings).
 */
export function barDevice(n = N): Device {
  const { array } = demoScenario(n);
  const f0 = BAR_F0;
  const delay = 1.5 / f0;
  return {
    id: 'bar',
    speakers: array,
    mics: array,
    emissions: array.map((_, s) => ({ drivers: [{ speaker: s, waveform: { type: 'ricker', amplitude: 1, frequency: f0, delay } }] })),
    f0,
    delay,
  };
}

/** Speaker indices that fire in emission `e`. */
export function activeSpeakers(d: Device, e: number): number[] {
  return d.emissions[e]?.drivers.map((x) => x.speaker) ?? [];
}

/** Devices by id (the worker and the page resolve a request's device here). */
export const DEVICES = { bar: barDevice } as const;
export type DeviceId = keyof typeof DEVICES;
