/**
 * Echo vision (#/echo): the playback schedule. Compute and playback are
 * decoupled: the worker streams each ping as soon as it is simulated, and
 * the page plays the story at a readable pace on this schedule:
 *
 *   ping 1 (slow) -> trace 1 -> ping 2 -> trace 2 -> ... -> ping n -> trace n -> guess
 *
 * "ping k" plays the k-th emission's scattered field (stage 1, clicks and
 * echoes); "trace k" adds that ping to the back-projected picture (stage 2,
 * tracing echoes back); "guess" fades in the network's map (stage 3). The
 * first ping is shown slowly and in full, later ones faster.
 */

export type SegKind = 'ping' | 'trace' | 'guess';

export interface Seg {
  kind: SegKind;
  /** Ping index (the last ping for 'guess'). */
  ping: number;
  /** Start and end, in seconds of playback at 1x. */
  t0: number;
  t1: number;
}

/** Seconds at 1x for ping k, its trace, and the final guess. */
export const pingSeconds = (k: number) => (k === 0 ? 3.2 : Math.max(0.45, 1.0 * 0.7 ** (k - 1)));
export const traceSeconds = (k: number) => (k === 0 ? 0.9 : Math.max(0.25, 0.4 * 0.75 ** (k - 1)));
export const GUESS_SECONDS = 1.6;

export function schedule(pings: number): Seg[] {
  const out: Seg[] = [];
  let t = 0;
  const add = (kind: SegKind, ping: number, d: number) => {
    out.push({ kind, ping, t0: t, t1: t + d });
    t += d;
  };
  for (let k = 0; k < pings; k++) {
    add('ping', k, pingSeconds(k));
    add('trace', k, traceSeconds(k));
  }
  add('guess', Math.max(0, pings - 1), GUESS_SECONDS);
  return out;
}

export const totalSeconds = (segs: Seg[]) => (segs.length ? segs[segs.length - 1].t1 : 0);

/** The segment at time t (clamped) and the fraction f in [0, 1] through it. */
export function locate(segs: Seg[], t: number): { index: number; seg: Seg; f: number } {
  const T = totalSeconds(segs);
  const tt = Math.min(T, Math.max(0, t));
  let i = 0;
  while (i < segs.length - 1 && tt >= segs[i].t1) i++;
  const s = segs[i];
  return { index: i, seg: s, f: s.t1 > s.t0 ? Math.min(1, Math.max(0, (tt - s.t0) / (s.t1 - s.t0))) : 1 };
}

export const stageOf = (kind: SegKind): 1 | 2 | 3 => (kind === 'ping' ? 1 : kind === 'trace' ? 2 : 3);

/**
 * How far playback may go with `ready` pings computed (each ping carries its
 * frames and its traced picture) and the network's answer in or not: to the
 * end once answered, else to just before the first segment still missing.
 */
export function availableSeconds(segs: Seg[], ready: number, answered: boolean): number {
  if (answered) return totalSeconds(segs);
  let t = 0;
  for (const s of segs) {
    if (s.kind === 'guess' || s.ping >= ready) break;
    t = s.t1;
  }
  // Stop just short of the boundary, so the last computed segment stays on screen.
  return Math.max(0, t - 1e-6);
}

/**
 * Display scale per frame: the frame's own peak, or a slowly decaying
 * earlier peak, whichever is larger, but never below `floor` x the ping's
 * peak. Faint late echoes stay visible without blowing up the weak
 * multiple bounces at the end. `halfLife` is in frames.
 */
export function displayScales(scales: Float32Array, halfLife = 100, floor = 0.3): Float32Array {
  let max = 0;
  for (const s of scales) max = Math.max(max, s);
  const rho = 0.5 ** (1 / halfLife);
  const out = new Float32Array(scales.length);
  let p = 0;
  for (let f = 0; f < scales.length; f++) {
    p = Math.max(scales[f], p * rho);
    out[f] = Math.max(p, floor * max, 1e-30);
  }
  return out;
}
