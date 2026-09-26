/**
 * Echo vision (#/echo): a toy page for "machine learning reconstructs a room
 * from its echoes". The closed loop's sensing (8-speaker bar, CPML room,
 * coherent migration images) runs in a worker (echo/echoWorker.ts); the
 * loop's U-Net turns the images into an obstacle map, shown next to the raw
 * back-projection image and the true room. See docs/web_app.md §5b (Echo vision page).
 */

import { Ear, Eraser, Eye, EyeOff, Hand, Minus, Shuffle, Square, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { EchoReply, EchoRequest } from '../echo/echoWorker';
import { drawPanel, ESTIMATE_COLOR, type Layer, type Overlay, TRUTH_COLOR } from '../echo/draw';
import { AREA, type CellRect, dragRect, EXAMPLES, familyCheck, N, paintRect, pointToCell, randomRoom, RIGID, type Tool } from '../echo/room';
import { echoSetup, type EchoResult } from '../echo/sense';
import { nearArray } from '../loop/closedLoop';
import { useApp } from '../state/store';
import '../echo/echo.css';

const REPORT_URL = 'https://github.com/kyleyhw/sound_simulation/blob/main/tests/reports/loop_sensing_2026_09_25.md';

type Phase = 'loading' | 'idle' | 'listening' | 'analysing' | 'done' | 'error';

const TOOLS: { id: Tool; label: string; icon: React.ReactNode; hint: string }[] = [
  { id: 'look', label: 'Look', icon: <Hand size={14} />, hint: 'Look only (no drawing)' },
  { id: 'block', label: 'Block', icon: <Square size={14} />, hint: 'Drag to draw a block' },
  { id: 'wall', label: 'Wall', icon: <Minus size={14} />, hint: 'Drag to draw a thin wall' },
  { id: 'erase', label: 'Erase', icon: <Eraser size={14} />, hint: 'Drag over cells to erase them' },
];

const FAMILY_NOTE: Record<'shape' | 'size' | 'count', string> = {
  shape: 'This room has shapes the network never saw. It was trained on boxes and walls, so it draws boxes.',
  size: 'This room has an object bigger than any in training. It was trained on small boxes and thin walls, so it draws those.',
  count: 'This room has more objects than any in training (one to three), so expect misses.',
};

/** A square canvas panel that redraws when its inputs (or the theme) change. */
function Panel({ layer, overlay, label, testId }: { layer: Layer; overlay: Overlay; label: string; testId?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useApp((s) => s.theme);
  useEffect(() => {
    // Deferred a frame so a theme switch has reached the document's CSS variables.
    const raf = requestAnimationFrame(() => ref.current && drawPanel(ref.current, N, layer, overlay));
    return () => cancelAnimationFrame(raf);
  }, [layer, overlay, theme]);
  return <canvas ref={ref} className="echo-canvas" role="img" aria-label={label} data-testid={testId} />;
}

function pct(v: number) {
  return `${Math.round(100 * v)}\u00a0%`;
}

export default function EchoVision() {
  const setup = useMemo(() => echoSetup(), []);
  const theme = useApp((s) => s.theme);
  const [room, setRoom] = useState<Uint8Array>(() => EXAMPLES[0].build());
  const [source, setSource] = useState<string>(EXAMPLES[0].id); // example id, 'random:<seed>' or 'custom'
  const [tool, setTool] = useState<Tool>('look');
  const [hidden, setHidden] = useState(false);
  const [phase, setPhase] = useState<Phase>('loading');
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ ping: number; pings: number; frac: number } | null>(null);
  const [out, setOut] = useState<{ result: EchoResult; ms: number; room: Uint8Array } | null>(null);
  const [showTruth, setShowTruth] = useState(true);
  const [drag, setDrag] = useState<{ a: [number, number]; b: [number, number] } | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const idRef = useRef(0);
  const roomCanvas = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<{ field: Float32Array; ping: number } | null>(null);
  const peakRef = useRef(1e-9);
  const resultsRef = useRef<HTMLElement>(null);
  const sentRoom = useRef<Uint8Array | null>(null); // the room of the run in flight
  const nearField = useMemo(() => nearArray(setup.array, N, N, 6), [setup]);

  const busy = phase === 'listening' || phase === 'analysing';
  const empty = !room.some((v) => v);

  // ---- the worker ------------------------------------------------------
  useEffect(() => {
    const w = new Worker(new URL('../echo/echoWorker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.onmessage = (e: MessageEvent<EchoReply>) => {
      const m = e.data;
      if (m.type === 'ready') {
        setReady(true);
        setPhase((p) => (p === 'loading' ? 'idle' : p));
        return;
      }
      if (m.type === 'error') {
        setError(m.message);
        setPhase('error');
        return;
      }
      if (m.id !== idRef.current) return; // a stale run
      if (m.type === 'frame') {
        if (frameRef.current?.ping !== m.ping) peakRef.current = 1e-9;
        frameRef.current = { field: m.field, ping: m.ping };
        setProgress({ ping: m.ping, pings: m.pings, frac: (m.ping + (m.step + 1) / m.steps) / m.pings });
      } else if (m.type === 'analysing') {
        setPhase('analysing');
      } else if (m.type === 'result') {
        frameRef.current = null;
        setOut({ result: m.result, ms: m.ms, room: sentRoom.current ?? new Uint8Array(N * N) });
        setPhase('done');
        setProgress(null);
      }
    };
    return () => w.terminate();
  }, []);

  const listen = () => {
    const w = workerRef.current;
    if (!w || busy || empty) return;
    const id = ++idRef.current;
    const materials = room.slice();
    sentRoom.current = materials.slice();
    frameRef.current = null;
    setOut(null);
    setError(null);
    setProgress({ ping: 0, pings: setup.array.length, frac: 0 });
    setPhase('listening');
    setTool('look');
    w.postMessage({ type: 'listen', id, materials } satisfies EchoRequest);
  };

  // Bring the answer into view once it arrives (phones: it is below the fold).
  useEffect(() => {
    if (phase !== 'done') return;
    const el = resultsRef.current;
    if (el && el.getBoundingClientRect().top > window.innerHeight * 0.6) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [phase]);

  // ---- editing ---------------------------------------------------------
  const changeRoom = (m: Uint8Array, src: string) => {
    setRoom(m);
    setSource(src);
    setOut(null);
    if (phase === 'done') setPhase('idle');
  };
  const newRandom = () => {
    const seed = 1 + Math.floor(Math.random() * 99_999);
    changeRoom(randomRoom(seed), `random:${seed}`);
  };

  const cellAt = (e: React.PointerEvent<HTMLCanvasElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect();
    return pointToCell(e.clientX - r.left, e.clientY - r.top, r.width, r.height);
  };
  const canDraw = tool !== 'look' && !hidden && !busy;
  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!canDraw) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const c = cellAt(e);
    setDrag({ a: c, b: c });
  };
  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (drag) setDrag({ a: drag.a, b: cellAt(e) });
  };
  const onUp = () => {
    if (!drag) return;
    const rect = dragRect(tool, drag.a, drag.b);
    setDrag(null);
    if (rect) changeRoom(paintRect(room, rect, tool === 'erase' ? 0 : RIGID), 'custom');
  };
  const preview: { rect: CellRect; erase: boolean } | null = useMemo(() => {
    if (!drag) return null;
    const rect = dragRect(tool, drag.a, drag.b);
    return rect ? { rect, erase: tool === 'erase' } : null;
  }, [drag, tool]);

  // ---- the room canvas (room view, or the live field while listening) ----
  const drawRoom = useCallback(() => {
    const cv = roomCanvas.current;
    if (!cv) return;
    const f = frameRef.current;
    const walls = hidden ? null : room;
    if (f) {
      let m = 0;
      for (let q = 0; q < f.field.length; q++) if (!nearField[q]) m = Math.max(m, Math.abs(f.field[q]));
      peakRef.current = Math.max(m, peakRef.current * 0.9);
      drawPanel(cv, N, { kind: 'field', values: f.field, walls, colormap: theme === 'light' ? 'balance' : 'icefire', scale: peakRef.current }, { array: setup.array, active: f.ping });
    } else {
      drawPanel(cv, N, { kind: 'room', walls }, { array: setup.array, area: busy || hidden ? null : AREA, preview });
    }
  }, [room, hidden, theme, preview, busy, setup, nearField]);

  useEffect(() => {
    const raf = requestAnimationFrame(drawRoom);
    return () => cancelAnimationFrame(raf);
  }, [drawRoom, progress]);

  // ---- results ---------------------------------------------------------
  const view = useMemo(() => {
    if (!out) return null;
    const { result } = out;
    // The echo image as amplitude, normalised away from the bar's near field (as back-projection does).
    let max = 0;
    for (let q = 0; q < result.image.length; q++) if (!nearField[q]) max = Math.max(max, result.image[q]);
    const echo = new Float32Array(result.image.length);
    for (let q = 0; q < echo.length; q++) echo[q] = max > 0 ? Math.sqrt(Math.max(0, result.image[q]) / max) : 0;
    return { echo, family: familyCheck(out.room) };
  }, [out, nearField]);

  const truthOutline = out && showTruth && !hidden ? [{ mask: out.room, color: TRUTH_COLOR, dash: true }] : [];
  const layers = useMemo(() => {
    if (!out || !view) return null;
    return {
      truth: { kind: 'room', walls: out.room } as Layer,
      echo: { kind: 'map', values: view.echo, colormap: 'magma' } as Layer,
      net: { kind: 'map', values: out.result.probability, colormap: 'magma' } as Layer,
    };
  }, [out, view]);
  const overlays = useMemo(() => {
    if (!out) return null;
    return {
      truth: { array: setup.array } as Overlay,
      echo: { array: setup.array, outlines: [...truthOutline, { mask: out.result.backprojection, color: ESTIMATE_COLOR }] } as Overlay,
      net: { array: setup.array, outlines: [...truthOutline, { mask: out.result.learned, color: ESTIMATE_COLOR }] } as Overlay,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [out, showTruth, hidden, setup]);

  const exampleLabel = EXAMPLES.find((e) => e.id === source)?.label;
  const roomCaption = source.startsWith('random:') ? `Random room #${source.slice(7)}` : source === 'custom' ? 'Your room' : exampleLabel;

  let status: React.ReactNode = null;
  if (phase === 'loading' && !busy) status = <span className="dim">Loading the network…</span>;
  if (phase === 'listening')
    status = progress && ready ? (
      <span>
        Ping <b>{progress.ping + 1}</b> of {progress.pings}: speaker {progress.ping + 1} clicks, all eight record.
      </span>
    ) : (
      <span className="dim">Getting ready…</span>
    );
  if (phase === 'analysing') status = <span>Turning the echoes into images, then running the network…</span>;
  if (phase === 'done' && out) status = <span className="dim">Done in {(out.ms / 1000).toFixed(1)} s, in this browser.</span>;
  if (phase === 'error') status = <span className="warn">Something went wrong: {error}</span>;

  return (
    <div className="content echo" data-testid="echo" data-phase={phase}>
      <h1>Echo vision</h1>
      <p className="lede">
        Eight speakers click one at a time and listen to the echoes. A small neural network turns those echoes into a map of the room it cannot see.
      </p>

      <section className="echo-top">
        <div className="echo-s1">
          <h2 className="echo-step">
            <span className="echo-num">1</span> The room
          </h2>
          <p className="muted echo-say">Pick a room, or draw one. Grey is solid; the red dots are the speaker bar.</p>
          <div className="btn-row">
            <button className="btn" onClick={newRandom} disabled={busy} data-testid="echo-random">
              <Shuffle size={15} /> New random room
            </button>
            <button className="btn" onClick={() => changeRoom(new Uint8Array(N * N), 'custom')} disabled={busy || empty}>
              <Trash2 size={15} /> Clear
            </button>
            <button className="btn" aria-pressed={hidden} onClick={() => setHidden(!hidden)} data-testid="echo-hide" title="Hide the room and guess along">
              {hidden ? <Eye size={15} /> : <EyeOff size={15} />} {hidden ? 'Show the room' : 'Hide the room'}
            </button>
          </div>
          <div className="echo-examples" role="group" aria-label="Example rooms">
            {EXAMPLES.map((ex) => (
              <button
                key={ex.id}
                className={`btn sm${ex.hard ? ' hard' : ''}`}
                aria-pressed={source === ex.id}
                onClick={() => changeRoom(ex.build(), ex.id)}
                disabled={busy}
                data-testid={`echo-example-${ex.id}`}
              >
                {ex.label}
              </button>
            ))}
          </div>
          <div className="echo-tools">
            <span className="label">Draw</span>
            <div className="seg" role="group" aria-label="Drawing tool">
              {TOOLS.map((t) => (
                <button key={t.id} aria-pressed={tool === t.id} onClick={() => setTool(t.id)} title={t.hint} disabled={busy || hidden} data-testid={`echo-tool-${t.id}`}>
                  {t.icon} {t.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <figure className="echo-stage">
          <div className="echo-canvas-wrap">
            <canvas
              ref={roomCanvas}
              className={`echo-canvas${canDraw ? ' drawing' : ''}`}
              data-testid="echo-room"
              role="img"
              aria-label={busy ? 'live pressure field of the current ping' : 'the room'}
              onPointerDown={onDown}
              onPointerMove={onMove}
              onPointerUp={onUp}
              onPointerCancel={() => setDrag(null)}
            />
            {hidden && !busy && (
              <div className="echo-hidden-badge" aria-hidden>
                ?<span>The room is hidden. Listen, and guess along.</span>
              </div>
            )}
          </div>
          <figcaption className="dim echo-cap">
            {busy ? 'The sound field of the current ping (red and blue: pressure above and below rest).' : `${roomCaption}. ${canDraw ? 'Drag on the room to draw; the dashed box is where objects may go.' : ''}`}
          </figcaption>
        </figure>

        <div className="echo-s2">
          <h2 className="echo-step">
            <span className="echo-num">2</span> Listen
          </h2>
          <p className="muted echo-say">Each speaker clicks in turn; all eight record what comes back.</p>
          <button className="btn primary echo-listen" onClick={listen} disabled={busy || empty} data-testid="echo-listen">
            <Ear size={18} /> {busy ? 'Listening…' : phase === 'done' ? 'Listen again' : 'Listen'}
          </button>
          {empty && !busy && <p className="hint dim">The room is empty. Add something to it first.</p>}
          <div className="echo-status" data-testid="echo-progress" aria-live="polite">
            {status}
          </div>
          {busy && (
            <div className="echo-bar" aria-hidden>
              <div style={{ width: `${Math.round(100 * (phase === 'analysing' ? 1 : (progress?.frac ?? 0)))}%` }} />
            </div>
          )}
        </div>
      </section>

      {out && layers && overlays && view && (
        <section
          ref={resultsRef}
          className="echo-results"
          data-testid="echo-result"
          data-iou-learned={out.result.iouLearned.toFixed(4)}
          data-iou-bp={out.result.iouBackprojection.toFixed(4)}
          data-in-family={String(view.family.inFamily)}
        >
          <p className="echo-verdict" data-testid="echo-verdict">
            The network recovered <b>{pct(out.result.iouLearned)}</b> of the room (IoU {out.result.iouLearned.toFixed(2)}); the raw echo image, {pct(out.result.iouBackprojection)}.
          </p>
          {!view.family.inFamily && view.family.reason && (
            <p className="echo-note" data-testid="echo-family-note">
              {FAMILY_NOTE[view.family.reason]}
            </p>
          )}
          <div className="echo-panels">
            <figure>
              <figcaption className="echo-ptitle">The real room</figcaption>
              {hidden ? (
                <div className="echo-reveal">
                  <button className="btn" onClick={() => setHidden(false)} data-testid="echo-reveal">
                    <Eye size={15} /> Reveal the room
                  </button>
                </div>
              ) : (
                <Panel layer={layers.truth} overlay={overlays.truth} label="the real room" testId="echo-truth" />
              )}
              <p className="echo-pnote dim">What the speakers could not see.</p>
            </figure>
            <figure>
              <figcaption className="echo-ptitle">
                <span className="echo-num sm">3</span> What the echoes show
              </figcaption>
              <div className="echo-pwrap">
                <Panel layer={layers.echo} overlay={overlays.echo} label="back-projected echo image" testId="echo-image" />
                <span className="echo-iou mono" data-testid="echo-iou-bp">
                  IoU {out.result.iouBackprojection.toFixed(2)}
                </span>
              </div>
              <p className="echo-pnote dim">Echoes traced back to where they could have come from. Taking its brightest spot gives the outlined guess.</p>
            </figure>
            <figure>
              <figcaption className="echo-ptitle">
                <span className="echo-num sm">4</span> What the network reconstructs
              </figcaption>
              <div className="echo-pwrap">
                <Panel layer={layers.net} overlay={overlays.net} label="the network's obstacle map" testId="echo-network" />
                <span className="echo-iou mono good" data-testid="echo-iou-learned">
                  IoU {out.result.iouLearned.toFixed(2)}
                </span>
              </div>
              <p className="echo-pnote dim">The network reads the same echo images. Bright: where it thinks something is solid.</p>
            </figure>
          </div>
          <div className="echo-legend">
            <span>
              <i className="echo-line" style={{ borderColor: ESTIMATE_COLOR }} /> the guess
            </span>
            <label className={hidden ? 'dim' : undefined}>
              <input type="checkbox" checked={showTruth && !hidden} disabled={hidden} onChange={(e) => setShowTruth(e.target.checked)} data-testid="echo-show-truth" />
              <i className="echo-line dashed" style={{ borderColor: TRUTH_COLOR }} /> show the real room's outline
            </label>
          </div>
        </section>
      )}

      <details className="echo-how">
        <summary>How it works</summary>
        <div className="prose">
          <p>
            The room is a 100 × 100 grid simulated with the same wave solver as the sandbox, with absorbing (CPML) outer walls, so the only echoes come from the objects. Each of the
            eight speakers sends a short click (a Ricker pulse); all eight record for 622 time steps. Subtracting a recording of the empty room leaves only the echoes.
          </p>
          <p>
            <b>The echo image</b> (step 3) is delay-and-sum back-projection: every point of the room collects the echo samples that would have arrived if a reflector sat there. The
            classic guess keeps the brightest blob. A straight bar mostly hears the front faces of objects, so the blob is a thin arc.
          </p>
          <p>
            <b>The network</b> (step 4) is a small U-Net (120 k parameters) that reads the same echo images and outputs, per cell, the probability that the cell is solid. Its guess
            is the cells above 0.62. It was trained on 2400 random rooms with one to three boxes or walls with a door, never on the examples here, and runs in your browser.
          </p>
          <p>
            <b>IoU</b> (intersection over union) is the overlap between a guess and the real room divided by their combined area: 1 is perfect, 0 is no overlap. On 100 held-out
            rooms the network scores 0.79 on average and the echo image 0.18. On shapes it never saw (discs, L-shapes, diagonal walls) it still draws boxes.
          </p>
          <p>
            Details: <a href={REPORT_URL} target="_blank" rel="noreferrer">the loop sensing report</a>. The same network builds the digital twin in the{' '}
            <a href="#/loop">closed-loop demo</a>.
          </p>
        </div>
      </details>
    </div>
  );
}
