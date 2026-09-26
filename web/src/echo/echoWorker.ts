/// <reference lib="webworker" />
/**
 * Echo vision worker: loads the loop's U-Net once, then runs each Listen off
 * the main thread. Each emission is simulated in the room and in the empty
 * room side by side (sense.ts listenSweep); as soon as one is done the
 * worker posts its scattered-field frames, its echo-only recordings and the
 * back-projected picture so far, so the page can start playing while the
 * rest is computed. The frame buffers are transferred, so the worker keeps
 * only the recordings (a few hundred kB).
 */
import { Simulation } from '../engine/simulation';
import { LoopUNet } from '../loop/learnedSensing';
import { DEVICES, type DeviceId } from './device';
import { analyseImages, cumulativeImages, type EchoResult, echoSetup, echoWindow, type FrameWindow, geometry, listenSweep, residuals, singleImages } from './sense';

export type EchoRequest = { type: 'listen'; id: number; materials: Uint8Array; device?: DeviceId };

/** One ping, ready to play. */
export interface PingData {
  ping: number;
  /** Gamma-coded scattered-field frames over the window (see sense.ts encodeFrame). */
  frames: Int8Array;
  scales: Float32Array;
  /** Echo-only recordings, mic by mic (mics x steps). */
  residual: Float32Array;
  /** Back-projected energy of this ping alone, and of pings 0..ping. */
  single: Float32Array;
  cumulative: Float32Array;
}

export type EchoReply =
  | { type: 'ready' }
  | { type: 'start'; id: number; device: DeviceId; pings: number; steps: number; dt: number; window: FrameWindow }
  | { type: 'ping'; id: number; data: PingData }
  | { type: 'result'; id: number; ms: number; result: EchoResult }
  | { type: 'error'; id?: number; message: string };

const post = (m: EchoReply, transfer: Transferable[] = []) => self.postMessage(m, transfer);
const { params, steps } = echoSetup();
const dt = new Simulation(params).dt;

const ready = LoopUNet.load(`${import.meta.env.BASE_URL}models/loop_unet`);
ready.then(
  () => post({ type: 'ready' }),
  (err: Error) => post({ type: 'error', message: `could not load the model: ${err.message}` }),
);

self.onmessage = async (e: MessageEvent<EchoRequest>) => {
  const { id, materials } = e.data;
  const deviceId: DeviceId = e.data.device ?? 'bar';
  try {
    const model = await ready;
    const t0 = performance.now();
    const device = DEVICES[deviceId]();
    const truth = geometry(materials);
    const win = echoWindow(materials, device, params, steps);
    post({ type: 'start', id, device: deviceId, pings: device.emissions.length, steps, dt, window: win });
    // The bar's picture: the loop's coherent migration (speakers are the mics).
    const array = device.speakers;
    const rr: Float32Array[][] = [];
    const re: Float32Array[][] = [];
    let last = null as ReturnType<typeof cumulativeImages> | null;
    listenSweep(truth, device, steps, win, (r) => {
      rr[r.emission] = r.recRoom;
      re[r.emission] = r.recEmpty;
      last = cumulativeImages(rr, re, array, params, device.f0, r.emission);
      const single = singleImages(rr, re, array, params, device.f0, r.emission).image;
      const data: PingData = { ping: r.emission, frames: r.frames, scales: r.scales, residual: residuals(r.recRoom, r.recEmpty), single, cumulative: last.image };
      post({ type: 'ping', id, data }, [data.frames.buffer, data.scales.buffer, data.residual.buffer, data.single.buffer]);
    });
    // With every ping included, the cumulative images are the loop's migration images.
    const result = analyseImages(last!, array, truth, model);
    post({ type: 'result', id, ms: performance.now() - t0, result });
  } catch (err) {
    post({ type: 'error', id, message: (err as Error).message });
  }
};
