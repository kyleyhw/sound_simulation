/**
 * Echo vision (#/echo): the playback of one Listen. The worker streams the
 * pings (echoWorker.ts); this plays them on the schedule of timeline.ts, at
 * a readable pace that does not depend on how fast they were computed:
 *
 *   1 Clicks and echoes: the scattered field (room - empty room) of each
 *     ping, the click itself as a thin ring, the echo-only recordings under
 *     the room with a playhead, a halo on each speaker as an echo arrives.
 *   2 Tracing echoes back: after each ping its back-projection is added to
 *     the picture, which sharpens ping by ping.
 *   3 The network's guess: the U-Net's map fades in over the picture, with
 *     its IoU.
 *
 * Drawing runs in requestAnimationFrame straight into the canvases; React
 * only re-renders when the segment, the play state or the data change.
 */

import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { nearArray } from '../loop/closedLoop';
import { colormapCss, type ColormapName } from '../render/colormaps';
import { activeSpeakers, type Device } from './device';
import { css, ESTIMATE_COLOR, TRUTH_COLOR } from './draw';
import type { PingData } from './echoWorker';
import {
  amplitudeImage,
  blit,
  CellLayer,
  drawDevice,
  drawRing,
  drawTraces,
  fillPath,
  fitCanvas,
  outlinePath,
  paintFrame,
  paintMap,
  paintTint,
  strokePath,
  TRACE_GAMMA,
  wallPath,
  zeroColor,
} from './playerDraw';
import { N } from './room';
import { echoSetup, type EchoResult, FRAME_GAMMA, frameCount, type FrameWindow } from './sense';
import { availableSeconds, displayScales, locate, schedule, type SegKind, stageOf, totalSeconds } from './timeline';

/** One Listen as the page holds it; the worker's messages fill it in. */
export interface Run {
  id: number;
  device: Device;
  /** The true room of this run. */
  room: Uint8Array;
  start: { pings: number; steps: number; dt: number; window: FrameWindow } | null;
  pings: PingData[];
  result: EchoResult | null;
  ms: number;
}

export interface PlayerProgress {
  kind: SegKind;
  ping: number;
  pings: number;
  /** Playing, but held at the end of what has been computed. */
  waiting: boolean;
  ended: boolean;
}

const SPEEDS = [0.5, 1, 2] as const;
/** This ping's own arcs, drawn over the picture while they are added. */
const ARCS: [number, number, number] = [125, 211, 252];
const STAGES = ['Clicks and echoes', 'Tracing echoes back', "The network's guess"] as const;
const ease = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

/** Per-ping data derived once for drawing. */
interface PingView {
  gains: Float32Array; // S / P per frame (frame peak over display peak)
  cumulative: HTMLCanvasElement;
  single: HTMLCanvasElement;
  echo: Float32Array; // compressed |residual| per mic and step, against the ping's loudest echo
}

function readColors(theme: string, v: number) {
  return {
    v,
    map: (theme === 'light' ? 'balance' : 'icefire') as ColormapName,
    wall: css('--text-2', '#a7afc2'),
    ink: css('--text', '#e6e9f0'),
    driver: css('--driver', '#fb7185'),
    accent: css('--accent', '#38bdf8'),
  };
}

function pct(v: number) {
  return `${Math.round(100 * v)} %`;
}

export function EchoPlayer({
  run,
  version,
  hidden,
  showTruth,
  theme,
  reduced,
  onProgress,
  onShowResults,
  ref,
}: {
  ref?: React.Ref<HTMLElement>;
  run: Run;
  version: number;
  hidden: boolean;
  showTruth: boolean;
  theme: string;
  reduced: boolean;
  onProgress: (p: PlayerProgress) => void;
  onShowResults: () => void;
}) {
  const fieldRef = useRef<HTMLCanvasElement>(null);
  const tlRef = useRef<HTMLCanvasElement>(null);
  const picRef = useRef<HTMLCanvasElement>(null);
  const scrubRef = useRef<HTMLInputElement>(null);
  const bufRef = useRef<HTMLDivElement>(null);
  const { params } = useMemo(() => echoSetup(), []);
  const pings = run.start?.pings ?? run.device.emissions.length;
  const segs = useMemo(() => schedule(pings), [pings]);
  const total = totalSeconds(segs);
  const near = useMemo(() => nearArray(run.device.mics, N, N, 6), [run.device]);
  const walls = useMemo(() => wallPath(run.room, N), [run.room]);

  // Playback state lives in a ref (the frame loop reads it); React mirrors the coarse parts.
  const st = useRef({ t: 0, playing: !reduced, speed: 1, skip: false, raf: 0, last: 0, dirty: true, key: '' });
  const [view, setView] = useState<{ kind: SegKind; ping: number; state: 'playing' | 'paused' | 'waiting' | 'ended' }>({ kind: 'ping', ping: 0, state: reduced ? 'paused' : 'playing' });
  const [speed, setSpeed] = useState(1);
  const progressRef = useRef(onProgress);
  progressRef.current = onProgress;

  // ---- derived data (per ping, per result, per theme) ------------------
  const derived = useRef<{ pings: PingView[]; net: HTMLCanvasElement | null; guess: Path2D | null; guessFill: Path2D | null; picKey: string; truth: Path2D; traces: { key: string; cv: HTMLCanvasElement } }>(null!);
  if (!derived.current) derived.current = { pings: [], net: null, guess: null, guessFill: null, picKey: '', truth: outlinePath(run.room, N), traces: { key: '', cv: document.createElement('canvas') } };
  useEffect(() => {
    const d = derived.current;
    const win = run.start?.window;
    const pingsTotal = run.start?.pings ?? run.device.emissions.length;
    for (let k = d.pings.length; k < run.pings.length && win; k++) {
      const p = run.pings[k];
      const disp = displayScales(p.scales);
      const gains = new Float32Array(p.scales.length);
      for (let f = 0; f < gains.length; f++) gains[f] = p.scales[f] / disp[f];
      const steps = run.start!.steps;
      const mics = run.device.mics.length;
      let R = 0;
      for (let m = 0; m < mics; m++) for (let s = win.from; s <= win.to; s++) R = Math.max(R, Math.abs(p.residual[m * steps + s]));
      const echo = new Float32Array(p.residual.length);
      if (R > 0) for (let q = 0; q < echo.length; q++) echo[q] = (Math.abs(p.residual[q]) / R) ** TRACE_GAMMA;
      d.pings.push({
        gains,
        // The sum's peak grows about linearly with the pings (the echoes add up where something
        // is), so the first pings are shown dimmer: the picture fills in as pings arrive.
        cumulative: paintMap(new CellLayer(N), amplitudeImage(p.cumulative, near, Math.sqrt(pingsTotal / (k + 1))), 'magma'),
        single: paintTint(new CellLayer(N), amplitudeImage(p.single, near), ARCS),
        echo,
      });
    }
    if (run.result && !d.net) {
      d.net = paintMap(new CellLayer(N), run.result.probability, 'magma');
      d.guess = outlinePath(run.result.learned, N);
      d.guessFill = wallPath(run.result.learned, N);
    }
    const s = st.current;
    if (run.result && reduced && s.t < total) s.t = total; // reduced motion: go straight to the summary
    s.dirty = true;
    kick();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version]);

  // ---- drawing ------------------------------------------------------------
  const colorsRef = useRef(readColors(theme, 0));
  useEffect(() => {
    // Deferred a frame so a theme switch has reached the document's CSS variables.
    const raf = requestAnimationFrame(() => {
      colorsRef.current = readColors(theme, colorsRef.current.v + 1);
      st.current.dirty = true;
      kick();
    });
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);
  const layer = useMemo(() => new CellLayer(N), []);

  const draw = useCallback(
    (t: number) => {
      const d = derived.current;
      const colors = colorsRef.current;
      const win = run.start?.window;
      const { seg, f } = locate(segs, t);
      const data = run.pings[seg.ping];
      const pv = d.pings[seg.ping];
      const ready = !!(win && data && pv);
      const dev = run.device;
      const cellsN = win ? frameCount(win) : 0;
      // Position in the ping (frames, and the step it stands for).
      const pos = seg.kind === 'ping' ? f * (cellsN - 1) : cellsN - 1;
      const step = win ? win.from + pos * win.every : 0;

      // Field: the scattered field, the click's ring, walls and the device.
      const fc = fieldRef.current;
      if (fc && fc.clientWidth) {
        const [W] = fitCanvas(fc, true);
        const ctx = fc.getContext('2d')!;
        const cell = W / N;
        ctx.fillStyle = zeroColor(colors.map);
        ctx.fillRect(0, 0, W, W);
        let level: Float32Array | null = null;
        let active: number[] = [];
        if (ready && seg.kind !== 'guess') {
          const fa = Math.floor(pos);
          const fb = Math.min(cellsN - 1, fa + 1);
          const src = paintFrame(layer, data.frames, fa, fb, pos - fa, pv.gains[fa], pv.gains[fb], colors.map, FRAME_GAMMA);
          blit(ctx, src, W, seg.kind === 'ping' ? 1 : (1 - f) ** 1.5);
        }
        if (!hidden) fillPath(ctx, walls, cell, colors.wall);
        if (ready && seg.kind === 'ping') {
          active = activeSpeakers(dev, seg.ping);
          const r = ((step * run.start!.dt - dev.delay) * params.c) / params.dx;
          drawRing(ctx, cell, active.map((s) => dev.speakers[s]), r, (0.5 * params.c) / dev.f0 / params.dx, colors.ink);
          const k = Math.round(step);
          const steps = run.start!.steps;
          level = new Float32Array(dev.mics.length);
          for (let m = 0; m < level.length; m++) level[m] = pv.echo[m * steps + k];
        }
        if (seg.kind === 'guess' && d.guess && run.result) {
          const a = ease(f / 0.6);
          if (d.guessFill) fillPath(ctx, d.guessFill, cell, ESTIMATE_COLOR, 0.28 * a);
          strokePath(ctx, d.guess, cell, ESTIMATE_COLOR, Math.max(2, cell * 0.45), a);
          if (!hidden && showTruth) strokePath(ctx, d.truth, cell, TRUTH_COLOR, Math.max(1.5, cell * 0.3), a, [cell * 1.2, cell * 0.9]);
        }
        drawDevice(ctx, cell, dev.speakers, dev.mics, active, level, { driver: colors.driver, ink: colors.ink });
      }

      // Timeline: this ping's echo-only recordings with a playhead.
      const tc = tlRef.current;
      if (tc && tc.clientWidth) {
        const [W, H] = fitCanvas(tc, false);
        const ctx = tc.getContext('2d')!;
        ctx.clearRect(0, 0, W, H);
        if (ready) {
          const tr = d.traces;
          const key = `${run.id}:${seg.ping}:${W}x${H}:${colors.v}`;
          if (tr.key !== key) {
            tr.cv.width = W;
            tr.cv.height = H;
            drawTraces(tr.cv, data.residual, dev.mics.length, run.start!.steps, win!.from, win!.to, activeSpeakers(dev, seg.ping), colors.ink, colors.driver);
            tr.key = key;
          }
          const x = seg.kind === 'ping' ? f * W : W;
          const dimA = seg.kind === 'guess' ? 0.25 : 0.22;
          ctx.globalAlpha = dimA;
          ctx.drawImage(tr.cv, 0, 0);
          ctx.globalAlpha = seg.kind === 'guess' ? 0.35 : seg.kind === 'trace' ? 1 - 0.5 * f : 1;
          if (x > 0) ctx.drawImage(tr.cv, 0, 0, x, H, 0, 0, x, H);
          ctx.globalAlpha = 1;
          if (seg.kind === 'ping') {
            ctx.fillStyle = colors.accent;
            ctx.fillRect(Math.min(W - 2, x), 0, Math.max(1.5, W / 400), H);
          }
        }
      }

      // Picture: the back-projection so far, then the network's map.
      const pc = picRef.current;
      // It is still during a ping: redraw only when something it shows changes.
      const picKey = `${seg.kind}:${seg.ping}:${seg.kind === 'ping' ? 0 : f.toFixed(3)}:${d.pings.length}:${!!d.net}:${hidden}:${showTruth}:${colors.v}:${pc?.clientWidth}`;
      if (pc && pc.clientWidth && picKey !== d.picKey) {
        d.picKey = picKey;
        const [W] = fitCanvas(pc, true);
        const ctx = pc.getContext('2d')!;
        const cell = W / N;
        ctx.fillStyle = '#000004';
        ctx.fillRect(0, 0, W, W);
        const cum = (k: number) => (k >= 0 ? d.pings[k]?.cumulative : undefined);
        if (seg.kind === 'ping') {
          const c = cum(seg.ping - 1);
          if (c) blit(ctx, c, W);
        } else if (seg.kind === 'trace') {
          const prev = cum(seg.ping - 1);
          if (prev) blit(ctx, prev, W);
          const g = ease(f / 0.75);
          // This ping's own arcs flash in, then the sum takes over.
          if (pv) {
            blit(ctx, pv.cumulative, W, g);
            blit(ctx, pv.single, W, 0.85 * Math.sin(Math.PI * Math.min(1, f * 1.1)));
          }
        } else {
          const last = cum(pings - 1);
          if (last) blit(ctx, last, W);
          if (d.net) blit(ctx, d.net, W, ease(f / 0.6));
          if (d.guess) strokePath(ctx, d.guess, cell, ESTIMATE_COLOR, Math.max(2, cell * 0.45), ease((f - 0.3) / 0.4));
          if (d.truth && !hidden && showTruth) strokePath(ctx, d.truth, cell, TRUTH_COLOR, Math.max(1.5, cell * 0.3), ease((f - 0.3) / 0.4), [cell * 1.2, cell * 0.9]);
        }
        drawDevice(ctx, cell, dev.speakers, dev.mics, [], null, { driver: colors.driver, ink: colors.ink });
      }
    },
    [run, segs, hidden, showTruth, walls, layer, params, pings],
  );

  // ---- the frame loop -----------------------------------------------------
  const report = useCallback(
    (t: number, avail: number) => {
      const s = st.current;
      const { seg } = locate(segs, t);
      const ended = t >= total - 1e-9;
      const waiting = !ended && (s.playing || s.skip) && t >= avail - 1e-9;
      const state = ended ? 'ended' : waiting ? 'waiting' : s.playing ? 'playing' : 'paused';
      const key = `${seg.kind}:${seg.ping}:${state}`;
      if (key === s.key) return;
      s.key = key;
      setView({ kind: seg.kind, ping: seg.ping, state });
      progressRef.current({ kind: seg.kind, ping: seg.ping, pings, waiting, ended });
    },
    [segs, total, pings],
  );

  const tick = useCallback(
    (ts: number) => {
      const s = st.current;
      s.raf = 0;
      const avail = availableSeconds(segs, derived.current.pings.length, !!derived.current.net);
      const dt = s.last ? Math.min(0.1, (ts - s.last) / 1000) : 0;
      s.last = ts;
      const t0 = s.t;
      if (s.skip) {
        s.t = avail;
        if (avail >= total) s.skip = false;
      } else if (s.playing) {
        s.t = Math.min(avail, s.t + dt * s.speed);
        if (s.t >= total) s.playing = false;
      }
      if (s.t !== t0 || s.dirty) {
        s.dirty = false;
        draw(s.t);
      }
      if (scrubRef.current) scrubRef.current.value = String(Math.round((1000 * s.t) / total));
      if (bufRef.current) bufRef.current.style.width = `${(100 * avail) / total}%`;
      report(s.t, avail);
      if (s.playing || s.skip) s.raf = requestAnimationFrame(tick);
      else s.last = 0;
    },
    [segs, run, total, draw, report],
  );
  const tickRef = useRef(tick);
  tickRef.current = tick;
  const kick = useCallback(() => {
    const s = st.current;
    if (!s.raf) s.raf = requestAnimationFrame((ts) => tickRef.current(ts));
  }, []);

  useEffect(() => {
    st.current.dirty = true;
    kick();
  }, [draw, kick]);
  useEffect(() => {
    const ro = new ResizeObserver(() => {
      st.current.dirty = true;
      kick();
    });
    for (const r of [fieldRef, tlRef, picRef]) if (r.current) ro.observe(r.current);
    return () => {
      ro.disconnect();
      cancelAnimationFrame(st.current.raf);
      st.current.raf = 0;
    };
  }, [kick]);

  // ---- controls -----------------------------------------------------------
  const play = () => {
    const s = st.current;
    if (s.t >= total - 1e-9) s.t = 0;
    s.playing = true;
    s.last = 0;
    s.dirty = true;
    kick();
  };
  const pause = () => {
    const s = st.current;
    s.playing = false;
    s.skip = false;
    s.dirty = true;
    kick();
  };
  const seek = (v: number) => {
    const s = st.current;
    const avail = availableSeconds(segs, derived.current.pings.length, !!derived.current.net);
    s.t = Math.min(avail, Math.max(0, v));
    s.skip = false;
    s.dirty = true;
    kick();
  };
  const skip = () => {
    const s = st.current;
    s.skip = true;
    s.dirty = true;
    kick();
  };
  const changeSpeed = (v: number) => {
    st.current.speed = v;
    setSpeed(v);
  };

  // ---- labels --------------------------------------------------------------
  const stage = stageOf(view.kind);
  const nSpeakers = run.device.speakers.length;
  const ended = view.state === 'ended';
  let status: React.ReactNode;
  if (!run.start) status = <span className="dim">Getting ready…</span>;
  else if (view.state === 'waiting' && run.pings.length >= pings) status = <span>Turning the echoes into a map…</span>;
  else if (view.state === 'waiting') status = <span className="dim">Still listening…</span>;
  else if (view.kind === 'ping')
    status = (
      <span>
        Ping <b>{view.ping + 1}</b> of {pings}: speaker {activeSpeakers(run.device, view.ping).map((s) => s + 1).join(' and ')} clicks, all {nSpeakers} listen.
      </span>
    );
  else if (view.kind === 'trace')
    status = (
      <span>
        Tracing ping {view.ping + 1}&rsquo;s echoes back into the room{view.ping === pings - 1 ? '.' : '…'}
      </span>
    );
  else status = <span>The network reads the traced echoes: its guess.</span>;

  const traced = view.kind === 'ping' ? view.ping : view.ping + 1;
  const picTitle = view.kind === 'guess' ? "The network's guess" : traced ? `The echoes traced back: ${traced} of ${pings} pings` : 'The echoes traced back';
  const res = run.result;
  const verdict = view.kind === 'guess' && res;

  return (
    <div className="echo-player" data-testid="echo-player" data-stage={stage} data-ping={view.ping + 1} data-state={view.state}>
      <figure className="echo-stage echo-live" ref={ref}>
        <ol className="echo-stages" data-testid="echo-stages" aria-label="Stages">
          {STAGES.map((label, i) => {
            const n = i + 1;
            const state = n < stage || (n === 3 && stage === 3 && ended) ? 'done' : n === stage ? 'active' : n === 2 && stage === 1 && view.ping > 0 ? 'done' : 'todo';
            return (
              <li key={label} data-state={state} aria-current={n === stage ? 'step' : undefined}>
                <span className="echo-num sm">{n}</span>
                <span className="echo-stage-label">{label}</span>
              </li>
            );
          })}
        </ol>
        <div className="echo-status" data-testid="echo-progress" aria-live="polite">
          {status}
        </div>
        <div className="echo-canvas-wrap">
          <canvas
            ref={fieldRef}
            className="echo-canvas echo-live-canvas"
            data-testid="echo-field"
            role="img"
            aria-label={view.kind === 'guess' ? "the room with the network's guess" : 'the echoes of the current ping'}
          />
        </div>
        <div className="echo-tl">
          <div className="echo-tl-head">
            <span>What the speakers hear: echoes only</span>
            <span className="mono dim">ping {Math.min(view.ping + 1, pings)}</span>
          </div>
          <canvas ref={tlRef} className="echo-tl-canvas" data-testid="echo-timeline" role="img" aria-label="the echo-only recording of each speaker for the current ping" />
        </div>
        <div className="echo-controls">
          <button
            className="btn icon"
            onClick={view.state === 'playing' || view.state === 'waiting' ? pause : play}
            aria-label={ended ? 'Replay' : view.state === 'playing' || view.state === 'waiting' ? 'Pause' : 'Play'}
            title={ended ? 'Replay' : view.state === 'playing' || view.state === 'waiting' ? 'Pause' : 'Play'}
            data-testid="echo-play"
          >
            {ended ? <RotateCcw size={16} /> : view.state === 'playing' || view.state === 'waiting' ? <Pause size={16} /> : <Play size={16} />}
          </button>
          <div className="echo-scrub">
            <div className="echo-scrub-buf" ref={bufRef} />
            <input
              ref={scrubRef}
              type="range"
              min={0}
              max={1000}
              step={1}
              defaultValue={0}
              aria-label="Playback position"
              data-testid="echo-scrub"
              onInput={(e) => seek((Number((e.target as HTMLInputElement).value) / 1000) * total)}
            />
          </div>
          <div className="seg echo-speed" role="group" aria-label="Playback speed">
            {SPEEDS.map((v) => (
              <button key={v} aria-pressed={speed === v} onClick={() => changeSpeed(v)} data-testid={`echo-speed-${v}`}>
                {v}×
              </button>
            ))}
          </div>
          <button className="btn icon" onClick={skip} disabled={ended} aria-label="Skip to the answer" title="Skip to the answer" data-testid="echo-skip">
            <SkipForward size={16} />
          </button>
        </div>
        <figcaption className="dim echo-cap">
          <span className="echo-key ring" /> the click &nbsp; <span className="echo-key wave" style={{ background: colormapCss(theme === 'light' ? 'balance' : 'icefire') }} /> echoes: sound that bounced off something.
          {reduced && !ended && view.state === 'paused' && ' Reduced motion: press play to watch.'}
        </figcaption>
      </figure>

      <div className="echo-pic" data-testid="echo-picture-panel">
        <div className="echo-pic-title">{picTitle}</div>
        <div className="echo-pwrap">
          <canvas ref={picRef} className="echo-canvas echo-live-canvas" data-testid="echo-picture" role="img" aria-label={picTitle} />
          {verdict && (
            <span className="echo-iou mono good" data-testid="echo-player-iou">
              IoU {res.iouLearned.toFixed(2)}
            </span>
          )}
        </div>
        <div className="echo-pic-note">
          {verdict ? (
            <>
              <p className="echo-mini-verdict" data-testid="echo-player-verdict">
                The network recovered <b>{pct(res.iouLearned)}</b> of the room; the traced echoes alone, {pct(res.iouBackprojection)}.
              </p>
              <button className="linklike" onClick={onShowResults}>
                Compare all three maps ↓
              </button>
            </>
          ) : (
            <p className="dim">Each echo could have come from anywhere on an arc around the speakers. Adding the arcs of every ping, they pile up where something is.</p>
          )}
        </div>
      </div>
    </div>
  );
}
