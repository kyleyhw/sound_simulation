/// <reference lib="webworker" />
/**
 * Measures the speaker-to-zone transfer functions (`measureTransfer`) off
 * the main thread. On the main thread the ~20 s of stepping made the whole
 * sandbox stutter; here the page stays responsive and shows the progress.
 */
import { type Geometry, measureTransfer, type Rect, type Transfer } from './soundfield';

export interface TransferRequest {
  g: Geometry;
  speakers: number[][];
  bright: Rect;
  dark: Rect;
  f: number;
}

export type TransferReply = { type: 'progress'; fraction: number } | { type: 'done'; transfer: Transfer } | { type: 'error'; message: string };

const scope = self as unknown as DedicatedWorkerGlobalScope;

scope.onmessage = async (e: MessageEvent<TransferRequest>) => {
  const { g, speakers, bright, dark, f } = e.data;
  try {
    // Yield rarely: nothing else runs in this worker, and progress messages
    // are only needed a few times a second.
    const transfer = await measureTransfer(g, speakers, bright, dark, f, {
      yieldEvery: 2000,
      onProgress: (fraction) => scope.postMessage({ type: 'progress', fraction } satisfies TransferReply),
    });
    scope.postMessage({ type: 'done', transfer } satisfies TransferReply);
  } catch (err) {
    scope.postMessage({ type: 'error', message: (err as Error).message } satisfies TransferReply);
  }
};
