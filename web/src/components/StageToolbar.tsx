import { Camera, Film, HelpCircle, Link2, Pause, Play, Redo2, RotateCcw, SkipForward, Undo2 } from 'lucide-react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { encodeSceneUrl } from '../engine/scene';
import { recordGif, recordVideo, screenshot } from '../lib/exporters';
import { toScene } from '../state/editable';
import type { FrameStats } from '../state/runtime';
import { consumedState } from '../state/urlScene';
import { useApp } from '../state/store';

const SPEEDS = [1, 2, 4, 8, 16, 32, 64];

export function StageToolbar() {
  const runtime = useApp((s) => s.runtime);
  const past = useApp((s) => s.past.length);
  const future = useApp((s) => s.future.length);
  const notify = useApp((s) => s.notify);
  const [stats, setStats] = useState<FrameStats>(() => runtime.stats());
  const [speed, setSpeed] = useState(runtime.stepsPerFrame);
  const [busy, setBusy] = useState<string | null>(null);
  const [gpuOk, setGpuOk] = useState(false);
  const [backend, setBackendState] = useState(runtime.backend);
  // Record menu: a controlled popover (a <details> never closed by itself
  // and was anchored off-screen on phones).
  const [recOpen, setRecOpen] = useState(false);
  const [recPos, setRecPos] = useState<{ left: number; top: number } | null>(null);
  const recBtnRef = useRef<HTMLButtonElement>(null);
  const recMenuRef = useRef<HTMLDivElement>(null);
  const REC_W = 200;
  useLayoutEffect(() => {
    if (!recOpen) return;
    const place = () => {
      const b = recBtnRef.current?.getBoundingClientRect();
      if (!b) return;
      const vw = document.documentElement.clientWidth;
      const left = Math.max(8, Math.min(b.right - REC_W, vw - REC_W - 8));
      setRecPos({ left, top: b.bottom + 6 });
    };
    place();
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (recMenuRef.current?.contains(t) || recBtnRef.current?.contains(t)) return;
      setRecOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setRecOpen(false);
        recBtnRef.current?.focus();
      }
    };
    document.addEventListener('pointerdown', onDown, true);
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => {
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
    };
  }, [recOpen]);

  useEffect(() => {
    let live = true;
    void import('../engine/gpu').then((m) => m.gpuAvailable()).then((ok) => live && setGpuOk(ok));
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => setBackendState(runtime.backend), [runtime, stats]);

  const chooseBackend = async (kind: 'cpu' | 'gpu') => {
    if (kind === 'gpu' && useApp.getState().view.overlay === 'intensity') {
      notify('Intensity arrows need the CPU engine; switch the overlay first.', 'error');
      return;
    }
    try {
      await runtime.setBackend(kind);
      setBackendState(runtime.backend);
      notify(kind === 'gpu' ? 'Running on the GPU (WebGPU)' : 'Running on the CPU');
    } catch (e) {
      notify(`WebGPU unavailable: ${(e as Error).message}`, 'error');
    }
  };

  useEffect(() => runtime.subscribe(setStats), [runtime]);
  useEffect(() => {
    runtime.stepsPerFrame = speed;
  }, [runtime, speed]);

  const wrap = () => document.querySelector('[data-testid="field"]') as HTMLElement | null;

  const share = async () => {
    try {
      const token = await encodeSceneUrl(toScene(useApp.getState().scene));
      const url = `${location.origin}${location.pathname}#/sandbox?s=${token}`;
      // Marked as loaded, so Back/Forward onto this entry keeps later edits.
      history.replaceState(consumedState(`s=${token}`), '', `#/sandbox?s=${token}`);
      try {
        await navigator.clipboard.writeText(url);
        notify('Share link copied to the clipboard');
      } catch {
        notify('Share link is in the address bar');
      }
    } catch (e) {
      notify(`Could not encode the scene: ${(e as Error).message}`, 'error');
    }
  };

  const run = async (label: string, fn: () => Promise<void>) => {
    setBusy(label);
    try {
      await fn();
    } catch (e) {
      notify((e as Error).message, 'error');
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="stage-toolbar" role="toolbar" aria-label="Simulation controls">
      <button
        className="btn primary"
        onClick={() => runtime.toggle()}
        aria-label={stats.running ? 'Pause' : 'Run'}
        title="Run / pause (Space)"
        data-testid="run"
      >
        {stats.running ? <Pause size={16} /> : <Play size={16} />}
        {stats.running ? 'Pause' : 'Run'}
      </button>
      <button className="btn" onClick={() => runtime.stepOnce(1)} title="Single step (.)" aria-label="Step" data-testid="step">
        <SkipForward size={16} />
      </button>
      <button className="btn" onClick={() => runtime.reset()} title="Reset field (Shift+R)" aria-label="Reset" data-testid="reset">
        <RotateCcw size={16} />
      </button>
      <label className="row tight" title="Simulation steps per displayed frame">
        <span className="muted" style={{ fontSize: 12.5 }}>Speed</span>
        <select className="input" style={{ width: 76 }} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Steps per frame">
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>
      <label className="row tight" title={gpuOk ? 'Compute backend: WebGPU runs the same kernels on the graphics card' : 'WebGPU is not available in this browser'}>
        <span className="muted" style={{ fontSize: 12.5 }}>Engine</span>
        <select className="input" style={{ width: 84 }} value={backend} onChange={(e) => void chooseBackend(e.target.value as 'cpu' | 'gpu')} aria-label="Compute backend">
          <option value="cpu">CPU</option>
          <option value="gpu" disabled={!gpuOk}>
            GPU
          </option>
        </select>
      </label>
      <span style={{ flex: 1 }} />
      <button className="btn icon" onClick={() => useApp.getState().undo()} disabled={past === 0} title="Undo (Ctrl+Z)" aria-label="Undo">
        <Undo2 size={16} />
      </button>
      <button className="btn icon" onClick={() => useApp.getState().redo()} disabled={future === 0} title="Redo (Ctrl+Shift+Z)" aria-label="Redo">
        <Redo2 size={16} />
      </button>
      <button className="btn" onClick={share} title="Copy a link to this scene" data-testid="share">
        <Link2 size={16} /> Share
      </button>
      <button
        className="btn icon"
        title="Screenshot (PNG)"
        aria-label="Screenshot"
        disabled={!!busy}
        onClick={() =>
          run('png', async () => {
            const w = wrap();
            if (w) await screenshot(w, 'acoustic-sandbox.png');
          })
        }
      >
        <Camera size={16} />
      </button>
      <button
        ref={recBtnRef}
        className="btn icon"
        title={busy === 'gif' || busy === 'video' ? 'Recording…' : 'Record'}
        aria-label="Record"
        aria-haspopup="menu"
        aria-expanded={recOpen}
        data-testid="record-menu"
        style={{ position: 'relative' }}
        onClick={() => setRecOpen(!recOpen)}
      >
        <Film size={16} />
        {(busy === 'gif' || busy === 'video') && <span className="rec-dot" aria-hidden="true" />}
      </button>
      {recOpen && (
        <div
          ref={recMenuRef}
          className="card"
          role="menu"
          aria-label="Record"
          data-testid="record-popover"
          style={{ position: 'fixed', left: recPos?.left ?? 8, top: recPos?.top ?? 60, zIndex: 30, width: REC_W, visibility: recPos ? 'visible' : 'hidden' }}
        >
          <button
            className="btn sm"
            role="menuitem"
            style={{ width: '100%', marginBottom: 6 }}
            disabled={!!busy}
            onClick={() =>
              run('gif', async () => {
                setRecOpen(false);
                if (!runtime.running) runtime.start();
                if (wrap()) await recordGif(wrap()!, 60, 'acoustic-sandbox');
              })
            }
          >
            {busy === 'gif' ? 'Recording GIF…' : 'Animated GIF (60 frames)'}
          </button>
          <button
            className="btn sm"
            role="menuitem"
            style={{ width: '100%' }}
            disabled={!!busy}
            onClick={() =>
              run('video', async () => {
                setRecOpen(false);
                if (!runtime.running) runtime.start();
                const c = wrap()?.querySelector('canvas');
                if (c) await recordVideo(c, 5000, 'acoustic-sandbox');
              })
            }
          >
            {busy === 'video' ? 'Recording 5 s…' : 'Video (5 s)'}
          </button>
        </div>
      )}
      <button className="btn icon" onClick={() => useApp.getState().setShowHelp(true)} title="Keyboard shortcuts (?)" aria-label="Help">
        <HelpCircle size={16} />
      </button>
    </div>
  );
}
