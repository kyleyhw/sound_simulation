/// <reference lib="webworker" />
/**
 * Echo vision worker: loads the loop's U-Net and records the empty-room
 * reference once, then runs each Listen (8 pings, migration, back-projection
 * and the network) off the main thread, streaming field frames as it goes.
 */
import { pingRecordings } from '../loop/closedLoop';
import { LoopUNet } from '../loop/learnedSensing';
import { N } from './room';
import { analyseEchoes, echoSetup, type EchoResult, F0, geometry, recordPings } from './sense';

export type EchoRequest = { type: 'listen'; id: number; materials: Uint8Array };
export type EchoReply =
  | { type: 'ready' }
  | { type: 'frame'; id: number; ping: number; pings: number; step: number; steps: number; field: Float32Array }
  | { type: 'analysing'; id: number }
  | { type: 'result'; id: number; ms: number; result: EchoResult }
  | { type: 'error'; id?: number; message: string };

const post = (m: EchoReply, transfer: Transferable[] = []) => self.postMessage(m, transfer);
const { array, steps } = echoSetup();

const ready = (async () => {
  const model = await LoopUNet.load(`${import.meta.env.BASE_URL}models/loop_unet`);
  const recEmpty = pingRecordings(geometry(new Uint8Array(N * N)), array, F0, steps);
  return { model, recEmpty };
})();
ready.then(
  () => post({ type: 'ready' }),
  (err: Error) => post({ type: 'error', message: `could not load the model: ${err.message}` }),
);

/** At most one field frame per this many ms (the page draws the latest one). */
const FRAME_MS = 30;

self.onmessage = async (e: MessageEvent<EchoRequest>) => {
  const { id, materials } = e.data;
  try {
    const { model, recEmpty } = await ready;
    const t0 = performance.now();
    const truth = geometry(materials);
    let last = -Infinity;
    const recRoom = recordPings(truth, array, F0, steps, (ping, step, p) => {
      const now = performance.now();
      if (now - last < FRAME_MS && step !== steps - 1) return;
      last = now;
      const field = p.slice();
      post({ type: 'frame', id, ping, pings: array.length, step, steps, field }, [field.buffer]);
    });
    post({ type: 'analysing', id });
    const result = analyseEchoes(recRoom, recEmpty, array, truth, model);
    post({ type: 'result', id, ms: performance.now() - t0, result });
  } catch (err) {
    post({ type: 'error', id, message: (err as Error).message });
  }
};
