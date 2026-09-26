/**
 * Gallery thumbnail cache. Fields are simulated once per session in a Web
 * Worker (gallery/thumbWorker.ts) and kept here, so revisiting the gallery
 * or switching the theme only repaints (a few ms), never re-simulates.
 * The worker gets one preset at a time, highest priority (top of the page)
 * first, so the cards fill in reading order.
 */
import type { ThumbReply } from './thumbWorker';

const cache = new Map<string, ThumbReply>();
const waiting = new Map<string, { priority: number; cbs: Set<(t: ThumbReply) => void> }>();
let worker: Worker | null = null;
let busy = false;

function pump(): void {
  if (busy) return;
  let next: string | null = null;
  let best = Infinity;
  for (const [id, w] of waiting) {
    if (w.cbs.size && w.priority < best) {
      best = w.priority;
      next = id;
    }
  }
  if (next === null) return;
  busy = true;
  getWorker().postMessage({ ids: [next] });
}

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./thumbWorker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (e: MessageEvent<ThumbReply>) => {
    const t = e.data;
    cache.set(t.id, t);
    const w = waiting.get(t.id);
    waiting.delete(t.id);
    w?.cbs.forEach((cb) => cb(t));
    busy = false;
    pump();
  };
  return worker;
}

/**
 * Ask for a preset's thumbnail field (lower `priority` first); `cb` runs now
 * if cached, else when ready. Returns a canceller.
 */
export function requestThumb(id: string, priority: number, cb: (t: ThumbReply) => void): () => void {
  const hit = cache.get(id);
  if (hit) {
    cb(hit);
    return () => {};
  }
  let w = waiting.get(id);
  if (!w) {
    w = { priority, cbs: new Set() };
    waiting.set(id, w);
  }
  w.priority = Math.min(w.priority, priority);
  w.cbs.add(cb);
  // Let the cards that come into view together all register before choosing.
  setTimeout(pump, 16);
  const entry = w;
  return () => entry.cbs.delete(cb);
}
