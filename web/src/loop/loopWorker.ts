/// <reference lib="webworker" />
/** Runs the closed-loop demo off the main thread and streams epoch results. */
import { type EstimatorName, runLoop } from './closedLoop';
import { LoopUNet } from './learnedSensing';
import { demoScenario } from './scenarios';

let modelPromise: Promise<LoopUNet> | null = null;
const loadModel = () => (modelPromise ??= LoopUNet.load(`${import.meta.env.BASE_URL}models/loop_unet`));

self.onmessage = async (e: MessageEvent<{ size: number; estimator?: EstimatorName }>) => {
  const sc = demoScenario(e.data.size);
  self.postMessage({ type: 'scenario', params: sc.params, array: sc.array, frequency: sc.frequency, epochs: sc.epochs.map((ep) => ({ label: ep.label, materials: ep.truth.materials, bright: ep.bright, dark: ep.dark })) });
  try {
    let learned: LoopUNet | null = null;
    if ((e.data.estimator ?? 'learned') === 'learned') {
      learned = await loadModel();
      if (learned.shape[0] !== e.data.size) learned = null; // trained for one grid; back-projection elsewhere
    }
    self.postMessage({ type: 'estimator', estimator: learned ? 'learned' : 'backprojection' });
    await runLoop(sc.epochs, sc.array, sc.frequency, {
      learned,
      onEpoch: (r, i) => self.postMessage({ type: 'epoch', index: i, result: r }),
    });
    self.postMessage({ type: 'done' });
  } catch (err) {
    self.postMessage({ type: 'error', message: (err as Error).message });
  }
};
