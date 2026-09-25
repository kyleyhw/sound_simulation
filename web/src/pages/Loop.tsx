import { Play } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { Rect } from '../control/soundfield';
import type { EpochResult, EstimatorName } from '../loop/closedLoop';
import type { Route } from '../lib/router';

interface ScenarioMsg {
  params: { shape: number[] };
  array: number[][];
  frequency: number;
  epochs: { label: string; materials: Uint8Array; bright: Rect; dark: Rect }[];
}

const ESTIMATOR_LABELS: Record<EstimatorName, string> = { learned: 'learned U-Net', backprojection: 'back-projection' };

const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const SERIES: { key: keyof EpochResult; label: string; color: string }[] = [
  { key: 'oracle', label: 'Oracle (true room)', color: '#9aa0a6' },
  { key: 'guarded', label: 'Closed loop (guarded)', color: '#ffc400' },
  { key: 'adaptive', label: 'Twin design', color: '#f28b50' },
  { key: 'naive', label: 'Empty-room design', color: '#50a0ff' },
  { key: 'static', label: 'Static (designed once)', color: '#e0457b' },
];

/** Room map: walls, sensed estimate (outline), zones and the array. */
function RoomMap({ sc, index, result }: { sc: ScenarioMsg; index: number; result?: EpochResult }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const [rows, cols] = sc.params.shape;
    cv.width = cols;
    cv.height = rows;
    const ctx = cv.getContext('2d')!;
    const img = ctx.createImageData(cols, rows);
    const ep = sc.epochs[index];
    const glow = result?.probability ?? result?.image;
    const sqrtScale = !result?.probability; // energy image: show amplitude; probability: as is
    let max = 0;
    if (glow) for (const v of glow) max = Math.max(max, v);
    for (let q = 0; q < rows * cols; q++) {
      let r = 18;
      let g = 20;
      let b = 26;
      if (glow && max > 0) {
        const v = sqrtScale ? Math.sqrt(glow[q] / max) : glow[q];
        r += 90 * v;
        g += 60 * v;
        b += 150 * v;
      }
      if (ep.materials[q]) [r, g, b] = [215, 215, 220];
      if (result?.estimate[q] && !ep.materials[q]) [r, g, b] = [242, 139, 80];
      if (result?.estimate[q] && ep.materials[q]) [r, g, b] = [255, 196, 120];
      img.data.set([r, g, b, 255], q * 4);
    }
    ctx.putImageData(img, 0, 0);
    const box = (z: Rect, color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.strokeRect(z[1] + 0.5, z[0] + 0.5, z[3] - z[1], z[2] - z[0]);
    };
    box(ep.bright, '#ffc400');
    box(ep.dark, '#50a0ff');
    ctx.fillStyle = '#e0457b';
    for (const a of sc.array) ctx.fillRect(a[1] - 1, a[0] - 1, 2, 2);
  }, [sc, index, result]);
  return <canvas ref={ref} style={{ width: '100%', aspectRatio: '1', imageRendering: 'pixelated', borderRadius: 6 }} aria-label={`room map, epoch ${index + 1}`} />;
}

function ContrastChart({ results, n }: { results: EpochResult[]; n: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    cv.width = w * dpr;
    cv.height = h * dpr;
    const ctx = cv.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 38, r: 10, t: 10, b: 22 };
    const ymax = 60;
    const X = (i: number) => pad.l + ((w - pad.l - pad.r) * (i + 0.5)) / n;
    const Y = (v: number) => pad.t + (h - pad.t - pad.b) * (1 - Math.max(-10, Math.min(ymax, v)) / ymax);
    ctx.strokeStyle = css('--border') || '#444';
    ctx.fillStyle = css('--text-3') || '#999';
    ctx.font = '11px system-ui';
    for (const v of [0, 10, 20, 30, 40, 50]) {
      ctx.beginPath();
      ctx.moveTo(pad.l, Y(v));
      ctx.lineTo(w - pad.r, Y(v));
      ctx.stroke();
      ctx.fillText(`${v} dB`, 2, Y(v) + 4);
    }
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = '#7bd88f';
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(10));
    ctx.lineTo(w - pad.r, Y(10));
    ctx.stroke();
    ctx.setLineDash([]);
    for (const s of SERIES) {
      ctx.strokeStyle = s.color;
      ctx.fillStyle = s.color;
      ctx.lineWidth = s.key === 'guarded' ? 2.5 : 1.5;
      ctx.beginPath();
      results.forEach((r, i) => {
        const v = r[s.key] as number;
        if (i === 0) ctx.moveTo(X(i), Y(v));
        else ctx.lineTo(X(i), Y(v));
      });
      ctx.stroke();
      results.forEach((r, i) => {
        ctx.beginPath();
        ctx.arc(X(i), Y(r[s.key] as number), 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
    ctx.lineWidth = 1;
  }, [results, n]);
  return <canvas ref={ref} style={{ width: '100%', height: 220 }} aria-label="zone contrast per epoch" />;
}

export default function Loop(_props: { route?: Route }) {
  const [sc, setSc] = useState<ScenarioMsg | null>(null);
  const [results, setResults] = useState<EpochResult[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [size, setSize] = useState(100);
  const [estimator, setEstimator] = useState<EstimatorName>('learned');
  const [active, setActive] = useState<EstimatorName | null>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => () => workerRef.current?.terminate(), []);

  const run = () => {
    workerRef.current?.terminate();
    const w = new Worker(new URL('../loop/loopWorker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    setResults([]);
    setActive(null);
    setError(null);
    setRunning(true);
    w.onmessage = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === 'scenario') setSc(m as ScenarioMsg);
      else if (m.type === 'estimator') setActive(m.estimator as EstimatorName);
      else if (m.type === 'epoch') setResults((r) => [...r, m.result as EpochResult]);
      else if (m.type === 'done') setRunning(false);
      else if (m.type === 'error') {
        setError(m.message);
        setRunning(false);
      }
    };
    w.postMessage({ size, estimator });
  };

  const n = sc?.epochs.length ?? 4;
  return (
    <div className="content" data-testid="loop">
      <h1>Closed loop: sense → twin → control</h1>
      <p className="lede">
        A speaker bar has to keep one zone loud and another quiet while the room changes. Each epoch the bar pings the room and back-projects the echoes into images of the
        room. A small U-Net turns those images into an obstacle estimate (or, as before, the brightest blob of the image is taken). That estimate becomes a digital twin, the controller is designed on the twin, and the result is measured in the true room. The static controller is designed
        once and never updated. The oracle is designed on the true room. Five monitor mics per zone (for example a phone at the listener) let the loop fall back to the
        empty-room design when the twin is worse.
      </p>
      <div className="row tight" style={{ alignItems: 'flex-end', gap: 12, marginBottom: 12 }}>
        <div className="field">
          <label>Grid</label>
          <select className="input" value={size} onChange={(e) => setSize(Number(e.target.value))} aria-label="Grid size" disabled={running}>
            <option value={60}>60² (fast)</option>
            <option value={80}>80²</option>
            <option value={100}>100²</option>
          </select>
        </div>
        <div className="field">
          <label>Room estimate</label>
          <select className="input" value={estimator} onChange={(e) => setEstimator(e.target.value as EstimatorName)} aria-label="Room estimate" disabled={running}>
            <option value="learned">Learned U-Net (100²)</option>
            <option value="backprojection">Back-projection (brightest blob)</option>
          </select>
        </div>
        <div className="field">
          <button className="btn primary" onClick={run} disabled={running} data-testid="run-loop">
            <Play size={16} /> {running ? `Running… epoch ${results.length + 1} of ${n}` : 'Run the loop'}
          </button>
        </div>
      </div>
      {active && (
        <p className="hint" data-testid="loop-estimator" data-estimator={active}>
          Active room estimate: <b>{ESTIMATOR_LABELS[active]}</b>
          {estimator === 'learned' && active !== 'learned' && ' (the learned model is trained for the 100² grid; other grids use back-projection)'}
        </p>
      )}
      {error && <p className="warn">Loop failed: {error}</p>}

      {sc && (
        <section className="figure">
          <h2 style={{ fontSize: 17, margin: '4px 0 8px' }}>Rooms and what the loop sensed</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))', gap: 12 }}>
            {sc.epochs.map((ep, i) => (
              <figure key={i} style={{ margin: 0 }}>
                <RoomMap sc={sc} index={i} result={results[i]} />
                <figcaption className="muted" style={{ fontSize: 12.5, marginTop: 4 }}>
                  {i + 1}. {ep.label}
                  {results[i]?.sensedIou != null && ` · sensed IoU ${results[i].sensedIou!.toFixed(2)}`}
                </figcaption>
              </figure>
            ))}
          </div>
          <p className="hint dim" style={{ fontSize: 12 }}>
            White: true walls and obstacles. Orange: the obstacle estimate. Purple glow: the back-projected echo energy, or the U-Net's obstacle probability when the learned
            estimate is active. Yellow box: loud zone. Blue box: quiet zone. Red dots: speakers.
          </p>
        </section>
      )}

      {results.length > 0 && (
        <section className="figure" data-testid="loop-results">
          <h2 style={{ fontSize: 17, margin: '4px 0 8px' }}>Zone contrast in the true room</h2>
          <ContrastChart results={results} n={n} />
          <div className="row wrap tight" style={{ gap: 14, fontSize: 12.5, margin: '6px 0 10px' }}>
            {SERIES.map((s) => (
              <span key={s.key}>
                <span style={{ display: 'inline-block', width: 10, height: 10, background: s.color, borderRadius: 2, marginRight: 5 }} />
                {s.label}
              </span>
            ))}
            <span style={{ color: '#7bd88f' }}>– – 10 dB target</span>
          </div>
          <table className="data">
            <thead>
              <tr>
                <th>epoch</th>
                <th>closed loop</th>
                <th>twin</th>
                <th>empty room</th>
                <th>static</th>
                <th>oracle</th>
                <th>estimate</th>
                <th>sense (ms)</th>
                <th>design (ms)</th>
                <th>act + measure (ms)</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i}>
                  <td>{r.label}</td>
                  <td className="mono">
                    <b>{r.guarded.toFixed(1)}</b> ({r.guardChoice})
                  </td>
                  <td className="mono">{r.adaptive.toFixed(1)}</td>
                  <td className="mono">{r.naive.toFixed(1)}</td>
                  <td className="mono">{r.static.toFixed(1)}</td>
                  <td className="mono">{r.oracle.toFixed(1)}</td>
                  <td>
                    {ESTIMATOR_LABELS[r.estimator]}
                    {r.estimator === 'learned' && <span className="dim"> ({r.latencyMs.estimate.toFixed(0)} ms)</span>}
                  </td>
                  <td className="mono">{r.latencyMs.sense.toFixed(0)}</td>
                  <td className="mono">{r.latencyMs.design.toFixed(0)}</td>
                  <td className="mono">{r.latencyMs.act.toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint dim" style={{ fontSize: 12 }}>
            Latencies are measured in this browser on this device. Sensing and design are dominated by simulating the room: 2 × N pings, and N steady-state tones on the twin.
            The sense time includes the U-Net, whose own time is shown next to the estimate.
            Applying new weights is instant. The act and measure stage simulates the true room until it reaches steady state.
          </p>
        </section>
      )}
    </div>
  );
}
