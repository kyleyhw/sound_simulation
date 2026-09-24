/**
 * Runtime: owns the live Simulation (outside React state), the
 * requestAnimationFrame run loop, and a ring buffer of recent frames for
 * pause-and-scrub. The loop steps the engine within a per-frame time
 * budget, so rendering stays at display rate regardless of grid size.
 */

import { Simulation } from '../engine/simulation';
import type { EditableScene } from './editable';

export interface FrameStats {
  step: number;
  time: number;
  fps: number;
  stepsPerSecond: number;
  peak: number;
  running: boolean;
  scrub: number | null;
  historyLength: number;
  unstable: boolean;
}

type Listener = (s: FrameStats) => void;

const HISTORY_FRAMES = 150;

export class Runtime {
  sim: Simulation;
  running = false;
  /** Target simulation steps per displayed frame (upper bound). */
  stepsPerFrame = 8;
  /** Max milliseconds of stepping per frame. */
  budgetMs = 12;
  /** Frames recorded for scrubbing: [step, time, field copy]. */
  private history: { step: number; time: number; p: Float32Array }[] = [];
  private historyEvery = 2;
  scrubIndex: number | null = null;
  private listeners = new Set<Listener>();
  private raf = 0;
  private fpsAcc = { frames: 0, steps: 0, t0: performance.now(), fps: 0, sps: 0 };
  /** Bumped whenever geometry/sim changes, so views can refresh caches. */
  version = 0;
  unstable = false;
  private frameCallbacks = new Set<() => void>();

  constructor(scene: EditableScene) {
    this.sim = this.build(scene);
  }

  private build(scene: EditableScene): Simulation {
    const sim = new Simulation(scene.params);
    sim.setMaterialMap(scene.materials);
    sim.setSpeedMap(scene.speed);
    sim.setDrivers(scene.drivers);
    sim.setProbes(scene.probes);
    return sim;
  }

  /** Rebuild from scratch (grid/params changed or a new scene was loaded). */
  load(scene: EditableScene): void {
    const rms = this.sim.rmsAccum !== null;
    const inten = this.sim.intensity !== null;
    this.sim = this.build(scene);
    this.sim.enableRms(rms);
    this.sim.enableIntensity(inten);
    this.history = [];
    this.scrubIndex = null;
    this.unstable = false;
    this.version++;
    this.emit();
  }

  syncGeometry(scene: EditableScene): void {
    this.sim.setMaterialMap(scene.materials);
    this.sim.setSpeedMap(scene.speed);
    this.version++;
  }

  syncSources(scene: EditableScene): void {
    this.sim.setDrivers(scene.drivers);
    this.sim.setProbes(scene.probes);
    this.version++;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    fn(this.stats());
    return () => this.listeners.delete(fn);
  }

  /** Called every animation frame after stepping (renderers hook in here). */
  onFrame(fn: () => void): () => void {
    this.frameCallbacks.add(fn);
    return () => this.frameCallbacks.delete(fn);
  }

  stats(): FrameStats {
    return {
      step: this.displayStep(),
      time: this.displayTime(),
      fps: this.fpsAcc.fps,
      stepsPerSecond: this.fpsAcc.sps,
      peak: this.peak(),
      running: this.running,
      scrub: this.scrubIndex,
      historyLength: this.history.length,
      unstable: this.unstable,
    };
  }

  private emit(): void {
    const s = this.stats();
    for (const l of this.listeners) l(s);
  }

  /** Field to display: live, or the scrubbed history frame. */
  displayField(): Float32Array {
    if (this.scrubIndex !== null && this.history[this.scrubIndex]) return this.history[this.scrubIndex].p;
    return this.sim.p;
  }

  displayStep(): number {
    if (this.scrubIndex !== null && this.history[this.scrubIndex]) return this.history[this.scrubIndex].step;
    return this.sim.step_count;
  }

  displayTime(): number {
    if (this.scrubIndex !== null && this.history[this.scrubIndex]) return this.history[this.scrubIndex].time;
    return this.sim.time;
  }

  peak(): number {
    const p = this.displayField();
    let m = 0;
    for (let i = 0; i < p.length; i++) {
      const v = Math.abs(p[i]);
      if (v > m) m = v;
    }
    return m;
  }

  start(): void {
    if (this.running) return;
    this.scrubIndex = null;
    this.running = true;
    this.raf = requestAnimationFrame(this.tick);
    this.emit();
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.emit();
    this.renderNow();
  }

  toggle(): void {
    if (this.running) this.stop();
    else this.start();
  }

  stepOnce(n = 1): void {
    this.scrubIndex = null;
    for (let k = 0; k < n; k++) this.advance();
    this.renderNow();
    this.emit();
  }

  reset(): void {
    this.sim.reset();
    this.history = [];
    this.scrubIndex = null;
    this.unstable = false;
    this.renderNow();
    this.emit();
  }

  scrubTo(index: number | null): void {
    if (index === null) this.scrubIndex = null;
    else {
      if (this.running) this.stop();
      this.scrubIndex = Math.max(0, Math.min(this.history.length - 1, Math.round(index)));
    }
    this.renderNow();
    this.emit();
  }

  get historySize(): number {
    return this.history.length;
  }

  renderNow(): void {
    for (const cb of this.frameCallbacks) cb();
  }

  private advance(): void {
    this.sim.step();
    if (this.sim.step_count % this.historyEvery === 0 && this.sim.n <= 1_200_000) {
      const recycled = this.history.length >= HISTORY_FRAMES ? this.history.shift()!.p : new Float32Array(this.sim.n);
      recycled.set(this.sim.p);
      this.history.push({ step: this.sim.step_count, time: this.sim.time, p: recycled });
    }
  }

  private tick = (): void => {
    if (!this.running) return;
    const t0 = performance.now();
    let steps = 0;
    while (steps < this.stepsPerFrame && performance.now() - t0 < this.budgetMs) {
      this.advance();
      steps++;
    }
    // Instability guard: a CFL-violating or runaway field shows as NaN/huge.
    const probe = this.sim.p[(this.sim.n / 2) | 0];
    if (!Number.isFinite(probe) || this.peak() > 1e12) {
      this.unstable = true;
      this.running = false;
      this.emit();
      return;
    }
    this.fpsAcc.frames++;
    this.fpsAcc.steps += steps;
    const now = performance.now();
    if (now - this.fpsAcc.t0 >= 500) {
      const dt = (now - this.fpsAcc.t0) / 1000;
      this.fpsAcc.fps = this.fpsAcc.frames / dt;
      this.fpsAcc.sps = this.fpsAcc.steps / dt;
      this.fpsAcc.frames = 0;
      this.fpsAcc.steps = 0;
      this.fpsAcc.t0 = now;
    }
    this.renderNow();
    this.emit();
    this.raf = requestAnimationFrame(this.tick);
  };
}
