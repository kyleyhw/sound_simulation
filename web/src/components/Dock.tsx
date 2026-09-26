import { Play, Square } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { playSignal, stopPlayback } from '../lib/audio';
import { magnitudeSpectrum, spectrogram } from '../lib/dsp';
import { colormapLut } from '../render/colormaps';
import type { FrameStats } from '../state/runtime';
import { useApp } from '../state/store';

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function prepCanvas(c: HTMLCanvasElement): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(c.clientWidth * dpr));
  const h = Math.max(1, Math.round(c.clientHeight * dpr));
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  }
  return c.getContext('2d');
}

/**
 * Line plot with min/max decimation. `topPad` (CSS px, default 18) keeps the
 * trace clear of the plot title drawn over the canvas's top-left corner.
 */
export function drawSeries(c: HTMLCanvasElement, series: { data: ArrayLike<number>; color: string }[], opts: { symmetric?: boolean; topPad?: number; bottomPad?: number } = {}): void {
  const ctx = prepCanvas(c);
  if (!ctx) return;
  const { width: W, height: H } = c;
  ctx.clearRect(0, 0, W, H);
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series)
    for (let i = 0; i < s.data.length; i++) {
      lo = Math.min(lo, s.data[i]);
      hi = Math.max(hi, s.data[i]);
    }
  if (!Number.isFinite(lo)) return;
  if (opts.symmetric !== false) {
    const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-12);
    lo = -m;
    hi = m;
  }
  if (hi - lo < 1e-12) hi = lo + 1;
  const dpr = window.devicePixelRatio || 1;
  const pad = (opts.bottomPad ?? 6) * dpr;
  const padTop = Math.min(H / 3, (opts.topPad ?? 18) * dpr);
  const span = Math.max(1, H - pad - padTop);
  const yOf = (v: number) => H - pad - ((v - lo) / (hi - lo)) * span;
  ctx.strokeStyle = cssVar('--border');
  ctx.lineWidth = 1;
  const y0 = yOf(0);
  ctx.beginPath();
  ctx.moveTo(0, y0);
  ctx.lineTo(W, y0);
  ctx.stroke();
  for (const s of series) {
    const n = s.data.length;
    if (n < 2) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.4 * (window.devicePixelRatio || 1);
    ctx.beginPath();
    // Min/max decimation per pixel column keeps peaks visible.
    const cols = Math.min(n, W);
    for (let x = 0; x < cols; x++) {
      const a = Math.floor((x * n) / cols);
      const b = Math.max(a + 1, Math.floor(((x + 1) * n) / cols));
      let mn = Infinity;
      let mx = -Infinity;
      for (let i = a; i < b; i++) {
        mn = Math.min(mn, s.data[i]);
        mx = Math.max(mx, s.data[i]);
      }
      const px = (x / Math.max(1, cols - 1)) * W;
      const yA = yOf(mn);
      const yB = yOf(mx);
      if (x === 0) ctx.moveTo(px, yA);
      ctx.lineTo(px, yA);
      ctx.lineTo(px, yB);
    }
    ctx.stroke();
  }
}

function drawSpectrogram(c: HTMLCanvasElement, x: Float32Array): void {
  const ctx = prepCanvas(c);
  if (!ctx) return;
  const { width: W, height: H } = c;
  ctx.clearRect(0, 0, W, H);
  if (x.length < 128) return;
  const nfft = x.length > 4096 ? 256 : 128;
  const { frames, bins } = spectrogram(x, nfft, nfft / 4);
  if (frames.length === 0) return;
  let mx = -Infinity;
  for (const f of frames) for (let k = 0; k < bins; k++) mx = Math.max(mx, f[k]);
  const img = ctx.createImageData(frames.length, bins);
  const lut = colormapLut('magma');
  for (let t = 0; t < frames.length; t++)
    for (let k = 0; k < bins; k++) {
      const v = Math.max(0, Math.min(1, 1 + (frames[t][k] - mx) / 70));
      const q = Math.round(v * 255) * 4;
      const o = ((bins - 1 - k) * frames.length + t) * 4;
      img.data[o] = lut[q];
      img.data[o + 1] = lut[q + 1];
      img.data[o + 2] = lut[q + 2];
      img.data[o + 3] = 255;
    }
  const off = document.createElement('canvas');
  off.width = frames.length;
  off.height = bins;
  off.getContext('2d')!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, W, H);
}

export function Dock() {
  const runtime = useApp((s) => s.runtime);
  const scene = useApp((s) => s.scene);
  const selected = useApp((s) => s.selected);
  const [probeId, setProbeId] = useState<string | null>(null);
  const [mode, setMode] = useState<'spectrogram' | 'spectrum'>('spectrogram');
  const [playing, setPlaying] = useState(false);
  const [stats, setStats] = useState<FrameStats>(() => runtime.stats());
  const waveRef = useRef<HTMLCanvasElement>(null);
  const specRef = useRef<HTMLCanvasElement>(null);
  const lastDraw = useRef(0);
  // Each Listen gets an id; a timer from an earlier clip must not flip the
  // label of a newer one back to "Listen".
  const playIdRef = useRef(0);
  const playTimerRef = useRef<number | undefined>(undefined);
  useEffect(
    () => () => {
      window.clearTimeout(playTimerRef.current);
      stopPlayback();
    },
    [],
  );

  useEffect(() => runtime.subscribe(setStats), [runtime]);

  // Follow the selected probe; otherwise keep the first one.
  useEffect(() => {
    if (selected?.kind === 'probe') setProbeId(selected.id);
  }, [selected]);
  const probe = scene.probes.find((p) => p.id === probeId) ?? scene.probes[0] ?? null;

  useEffect(() => {
    const redraw = (force = false) => {
      const now = performance.now();
      if (!force && now - lastDraw.current < 90) return;
      lastDraw.current = now;
      if (!probe || !waveRef.current || !specRef.current) return;
      const series = runtime.sim.probeSeries(probe.id);
      const window = series.length > 3000 ? series.subarray(series.length - 3000) : series;
      drawSeries(waveRef.current, [{ data: window, color: cssVar('--probe') }]);
      if (mode === 'spectrogram') drawSpectrogram(specRef.current, series);
      else {
        // magnitudeSpectrum is already one-sided (0 .. Nyquist): plot all of it.
        const spec = magnitudeSpectrum(series);
        const db = Array.from(spec, (v) => 20 * Math.log10(v + 1e-9));
        drawSeries(specRef.current, [{ data: db, color: cssVar('--accent') }], { symmetric: false, bottomPad: 18 });
      }
    };
    redraw(true);
    return runtime.onFrame(() => redraw());
  }, [runtime, probe, mode]);

  const listen = async () => {
    if (!probe) return;
    const id = ++playIdRef.current;
    window.clearTimeout(playTimerRef.current);
    if (playing) {
      stopPlayback();
      setPlaying(false);
      return;
    }
    const x = runtime.sim.probeSeries(probe.id);
    if (x.length < 64) return useApp.getState().notify('Run the simulation first to record something to play.');
    try {
      const { seconds } = await playSignal(x, { dt: runtime.sim.dt, units: scene.units });
      if (id !== playIdRef.current) return;
      setPlaying(true);
      playTimerRef.current = window.setTimeout(() => {
        if (id === playIdRef.current) setPlaying(false);
      }, seconds * 1000 + 50);
    } catch (e) {
      useApp.getState().notify(`Could not play the recording: ${(e as Error).message}`, 'error');
    }
  };

  const fmtT = (t: number) => (scene.units === 'si' ? `${(t * 1000).toFixed(2)} ms` : t.toFixed(1));
  // Spectrum axis: 0 .. Nyquist, in Hz (SI) or cycles per time unit (grid).
  const nyquist = 1 / (2 * runtime.sim.dt);
  const fmtF = (f: number) =>
    scene.units === 'si' ? (f >= 1000 ? `${(f / 1000).toPrecision(3)} kHz` : `${f.toPrecision(3)} Hz`) : `${f.toPrecision(3)} cycles/unit`;

  return (
    <section className="dock" aria-label="Recordings and timeline">
      <div className="dock-head">
        <strong style={{ fontSize: 13 }}>Recordings</strong>
        {scene.probes.length > 0 ? (
          <select className="input" style={{ width: 160 }} value={probe?.id ?? ''} onChange={(e) => setProbeId(e.target.value)} aria-label="Microphone">
            {scene.probes.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label ?? p.id}
              </option>
            ))}
          </select>
        ) : null}
        <span style={{ flex: 1 }} />
        <div className="seg" role="group" aria-label="Frequency view">
          <button aria-pressed={mode === 'spectrogram'} onClick={() => setMode('spectrogram')}>
            Spectrogram
          </button>
          <button aria-pressed={mode === 'spectrum'} onClick={() => setMode('spectrum')}>
            Spectrum
          </button>
        </div>
        <button className="btn sm" onClick={listen} disabled={!probe} data-testid="listen" title={scene.units === 'si' ? 'Play at true pitch' : 'Play, pitch-mapped so the dominant frequency is 440 Hz'}>
          {playing ? <Square size={14} /> : <Play size={14} />} {playing ? 'Stop' : 'Listen'}
        </button>
      </div>
      <div className="dock-body">
        {probe ? (
          <>
            <div className="plot">
              <span className="plot-title">pressure at {probe.label ?? 'mic'}</span>
              <canvas ref={waveRef} data-testid="scope" />
            </div>
            <div className="plot">
              <span className="plot-title">{mode === 'spectrogram' ? 'spectrogram (time →, frequency ↑)' : 'spectrum (dB) vs frequency'}</span>
              <canvas ref={specRef} data-testid="spectrum" />
              {mode === 'spectrum' && (
                <>
                  <span className="plot-axis mono" style={{ left: 8 }}>
                    0
                  </span>
                  <span className="plot-axis mono" style={{ right: 8 }} data-testid="spectrum-max">
                    {fmtF(nyquist)} (Nyquist)
                  </span>
                </>
              )}
            </div>
          </>
        ) : (
          <div className="plot" style={{ gridColumn: '1 / -1' }}>
            <div className="empty">Place a microphone (M) to record the pressure at a point, see its spectrum, and listen to it.</div>
          </div>
        )}
      </div>
      <div className="timeline">
        <span className="dim" style={{ fontSize: 12 }}>
          History
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(0, stats.historyLength - 1)}
          value={stats.scrub ?? Math.max(0, stats.historyLength - 1)}
          disabled={stats.historyLength < 2}
          aria-label="Scrub recent frames"
          data-testid="scrubber"
          onChange={(e) => runtime.scrubTo(Number(e.target.value))}
        />
        <span className="mono dim" style={{ fontSize: 12, minWidth: 90, textAlign: 'right' }}>
          {stats.scrub !== null ? `t = ${fmtT(stats.time)}` : 'live'}
        </span>
        {stats.scrub !== null && (
          <button className="btn sm" onClick={() => runtime.scrubTo(null)}>
            Live
          </button>
        )}
      </div>
    </section>
  );
}
