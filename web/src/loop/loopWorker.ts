/// <reference lib="webworker" />
/** Runs the closed-loop demo off the main thread and streams epoch results. */
import { runLoop } from './closedLoop';
import { demoScenario } from './scenarios';

self.onmessage = async (e: MessageEvent<{ size: number }>) => {
  const sc = demoScenario(e.data.size);
  self.postMessage({ type: 'scenario', params: sc.params, array: sc.array, frequency: sc.frequency, epochs: sc.epochs.map((ep) => ({ label: ep.label, materials: ep.truth.materials, bright: ep.bright, dark: ep.dark })) });
  try {
    await runLoop(sc.epochs, sc.array, sc.frequency, {
      onEpoch: (r, i) => self.postMessage({ type: 'epoch', index: i, result: r }),
    });
    self.postMessage({ type: 'done' });
  } catch (err) {
    self.postMessage({ type: 'error', message: (err as Error).message });
  }
};
