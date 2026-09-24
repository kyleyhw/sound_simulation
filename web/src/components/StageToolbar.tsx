import { Camera, Film, HelpCircle, Link2, Pause, Play, Redo2, RotateCcw, SkipForward, Undo2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { encodeSceneUrl } from '../engine/scene';
import { recordGif, recordVideo, screenshot } from '../lib/exporters';
import { toScene } from '../state/editable';
import type { FrameStats } from '../state/runtime';
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
      history.replaceState(null, '', `#/sandbox?s=${token}`);
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
      <button className="btn" onClick={() => runtime.reset()} title="Reset field (R)" aria-label="Reset" data-testid="reset">
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
      <details className="export-menu" style={{ position: 'relative' }}>
        <summary className="btn icon" title="Record" aria-label="Record" style={{ listStyle: 'none' }}>
          <Film size={16} />
        </summary>
        <div className="card" style={{ position: 'absolute', right: 0, top: 36, zIndex: 5, width: 190 }}>
          <button
            className="btn sm"
            style={{ width: '100%', marginBottom: 6 }}
            disabled={!!busy}
            onClick={() =>
              run('gif', async () => {
                if (!runtime.running) runtime.start();
                if (wrap()) await recordGif(wrap()!, 60, 'acoustic-sandbox');
              })
            }
          >
            {busy === 'gif' ? 'Recording GIF…' : 'Animated GIF (60 frames)'}
          </button>
          <button
            className="btn sm"
            style={{ width: '100%' }}
            disabled={!!busy}
            onClick={() =>
              run('video', async () => {
                if (!runtime.running) runtime.start();
                const c = wrap()?.querySelector('canvas');
                if (c) await recordVideo(c, 5000, 'acoustic-sandbox');
              })
            }
          >
            {busy === 'video' ? 'Recording 5 s…' : 'Video (5 s)'}
          </button>
        </div>
      </details>
      <button className="btn icon" onClick={() => useApp.getState().setShowHelp(true)} title="Keyboard shortcuts (?)" aria-label="Help">
        <HelpCircle size={16} />
      </button>
    </div>
  );
}
