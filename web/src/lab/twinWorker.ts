/// <reference lib="webworker" />
/**
 * Runs the room-twin 3D FDTD (`simulateShoebox`) off the main thread, so
 * the Lab page stays responsive during the ~10 s simulation.
 */
import type { Shoebox } from './measure';
import { simulateShoebox, type TwinResult } from './twin';

export interface TwinRequest {
  room: Shoebox;
  pos: [number, number, number];
  maxCells: number;
  seconds: number;
}

export type TwinReply = { type: 'progress'; fraction: number } | { type: 'done'; result: TwinResult } | { type: 'error'; message: string };

const scope = self as unknown as DedicatedWorkerGlobalScope;
const post = (m: TwinReply, transfer: Transferable[] = []) => scope.postMessage(m, transfer);

scope.onmessage = async (e: MessageEvent<TwinRequest>) => {
  const { room, pos, maxCells, seconds } = e.data;
  try {
    const result = await simulateShoebox(room, pos, { maxCells, seconds, onProgress: (fraction) => post({ type: 'progress', fraction }) });
    post({ type: 'done', result }, [result.ir.buffer]);
  } catch (err) {
    post({ type: 'error', message: (err as Error).message });
  }
};
