/**
 * Echo vision (#/echo): one sensing sweep of the closed loop's room estimate
 * (loop/closedLoop.ts senseRoom, loop/learnedSensing.ts), split so the page
 * can animate the pings. `recordPings` is `pingRecordings` with a frame hook
 * (tests/unit/echo.test.ts checks the two agree exactly); `analyseEchoes`
 * is the rest of senseRoom: migration images, back-projection and the U-Net.
 */

import type { Geometry } from '../control/soundfield';
import { Simulation } from '../engine/simulation';
import { backProjectionEstimate, type LearnedEstimator, maskIou, migrationImages, senseSteps } from '../loop/closedLoop';
import { demoScenario } from '../loop/scenarios';
import { N } from './room';

/** Ricker ping centre frequency (cycles per unit time), as in the loop's sensing. */
export const F0 = 0.08;

/** The loop's sensing setup: 100 x 100 grid, CPML walls, the 8-speaker bar. */
export function echoSetup() {
  const { params, array } = demoScenario(N);
  return { params, array, steps: senseSteps(params) };
}

export function geometry(materials: Uint8Array): Geometry {
  return { params: echoSetup().params, materials, speed: null };
}

/**
 * Every speaker pings in turn while every bar position records (the loop's
 * pingRecordings). `onFrame(ping, step, p)` sees the live pressure field
 * after each step; `p` is the simulation's buffer, so copy it to keep it.
 */
export function recordPings(g: Geometry, array: number[][], f0: number, steps: number, onFrame?: (ping: number, step: number, p: Float32Array) => void): Float32Array[][] {
  const sim = new Simulation(g.params);
  sim.setMaterialMap(g.materials);
  if (g.speed) sim.setSpeedMap(g.speed);
  const delay = 1.5 / f0;
  const idx = array.map((p) => sim.index(p));
  const out: Float32Array[][] = [];
  for (let s = 0; s < array.length; s++) {
    sim.reset();
    sim.setDrivers([{ id: 'ping', pos: array[s], waveform: { type: 'ricker', amplitude: 1, frequency: f0, delay }, enabled: true }]);
    const rec = array.map(() => new Float32Array(steps));
    for (let k = 0; k < steps; k++) {
      sim.step();
      for (let m = 0; m < idx.length; m++) rec[m][k] = sim.p[idx[m]];
      onFrame?.(s, k, sim.p);
    }
    out.push(rec);
  }
  return out;
}

export interface EchoResult {
  /** Back-projected echo energy (the image back-projection thresholds). */
  image: Float32Array;
  /** Back-projection estimate: the brightest blob of `image`. */
  backprojection: Uint8Array;
  /** U-Net obstacle probability (0 outside its crop). */
  probability: Float32Array;
  /** U-Net estimate: probability above the model's threshold. */
  learned: Uint8Array;
  iouBackprojection: number;
  iouLearned: number;
}

/** Everything senseRoom does after the pings, with both estimates scored against the truth. */
export function analyseEchoes(recRoom: Float32Array[][], recEmpty: Float32Array[][], array: number[][], truth: Geometry, model: LearnedEstimator): EchoResult {
  const images = migrationImages(recRoom, recEmpty, array, truth.params, F0);
  const backprojection = backProjectionEstimate(images.image, array, truth.params.shape);
  const { estimate: learned, probability } = model.estimate(images, array);
  return {
    image: images.image,
    backprojection,
    probability,
    learned,
    iouBackprojection: maskIou(backprojection, truth.materials),
    iouLearned: maskIou(learned, truth.materials),
  };
}
