import { useEffect, useRef, useState } from 'react';
import {
  contrastDb,
  design,
  type Design,
  DESIGN_LABELS,
  type Geometry,
  lineArray,
  type Rect,
  type Transfer,
  weightedDrivers,
  zoneContrastFromEnergy,
} from '../control/soundfield';
import { cx } from '../control/complex';
import type { TransferReply, TransferRequest } from '../control/transferWorker';
import { useApp } from '../state/store';
import { NumberField } from './fields';

const PREFIX = 'arr-';

const sameRect = (a: Rect | null, b: Rect | null) => !!a && !!b && a.every((v, i) => v === b[i]);

/**
 * Sound-field control (plan 7.7): paint a loud and a quiet zone, place a
 * speaker array, measure transfer functions with the engine, and apply the
 * chosen design as per-speaker gain and delay. The live readout measures
 * the zone contrast from the time-averaged loudness map.
 */
export function ControlPanel() {
  const scene = useApp((s) => s.scene);
  const runtime = useApp((s) => s.runtime);
  const zones = useApp((s) => s.zones);
  const zoneKind = useApp((s) => s.zoneKind);
  const tool = useApp((s) => s.tool);
  const setTool = useApp((s) => s.setTool);
  const setZoneKind = useApp((s) => s.setZoneKind);
  const setZone = useApp((s) => s.setZone);
  const replaceDrivers = useApp((s) => s.replaceDrivers);
  const setView = useApp((s) => s.setView);
  const notify = useApp((s) => s.notify);

  const [rows, cols] = scene.params.shape;
  const [count, setCount] = useState(8);
  const [spacing, setSpacing] = useState(4);
  const [centreRow, setCentreRow] = useState(Math.round(rows * 0.82));
  const [centreCol, setCentreCol] = useState(Math.round(cols / 2));
  const [axis, setAxis] = useState<0 | 1>(1);
  const [freq, setFreq] = useState(0.05);
  const [method, setMethod] = useState<Design>('acc');
  const [progress, setProgress] = useState<number | null>(null);
  const [transfer, setTransfer] = useState<Transfer | null>(null);
  const [predicted, setPredicted] = useState<Record<string, number> | null>(null);
  const [live, setLive] = useState<number | null>(null);
  /** Zones the current transfer functions were measured for. */
  const [measuredZones, setMeasuredZones] = useState<{ bright: Rect; dark: Rect } | null>(null);
  const workerRef = useRef<Worker | null>(null);
  useEffect(() => () => workerRef.current?.terminate(), []);

  const is2d = scene.params.dims === 2;
  const array = scene.drivers.filter((d) => d.id.startsWith(PREFIX));
  const speakers = array.map((d) => d.pos);
  const si = scene.units === 'si';
  const hz = (f: number) => (si ? `${((f * scene.params.c) / scene.params.dx).toFixed(0)} Hz` : `${f} cycles/unit`);

  // Live contrast from the loudness map, ~4 Hz.
  useEffect(() => {
    if (!zones.bright || !zones.dark) {
      setLive(null); // no zones, no contrast (not the last value)
      return;
    }
    setLive(null);
    let last = 0;
    return runtime.onFrame(() => {
      const now = performance.now();
      if (now - last < 250) return;
      last = now;
      const rms = runtime.sim.rmsMap();
      if (!rms || !zones.bright || !zones.dark) return setLive(null);
      const e = rms.map((v) => v * v);
      setLive(zoneContrastFromEnergy(e, (p) => runtime.sim.index(p), zones.bright, zones.dark));
    });
  }, [runtime, zones]);

  if (!is2d) {
    return (
      <div>
        <h3>Sound-field control</h3>
        <p className="hint">Zone control works on 2D scenes. Switch the grid to 2D in the Scene tab.</p>
      </div>
    );
  }

  const placeArray = () => {
    const pos = lineArray([centreRow, centreCol], count, spacing, axis, scene.params.shape);
    replaceDrivers(
      PREFIX,
      pos.map((p, s) => ({ id: `${PREFIX}${s}`, pos: p, waveform: { type: 'cosine', amplitude: 1, frequency: freq }, enabled: true })),
    );
    setTransfer(null);
    setPredicted(null);
  };

  // The measurement steps the engine once per speaker until steady state
  // (~20 s at 200²), so it runs in a worker; the UI stays live and it can be cancelled.
  const runDesign = () => {
    if (!zones.bright || !zones.dark) return notify('Draw a loud zone and a quiet zone first.', 'error');
    if (speakers.length < 2) return notify('Place a speaker array first.', 'error');
    const bright = zones.bright;
    const dark = zones.dark;
    const g: Geometry = { params: scene.params, materials: scene.materials, speed: scene.speed };
    const params = scene.params;
    const spk = speakers;
    const ids = array.map((d) => d.id);
    workerRef.current?.terminate();
    const w = new Worker(new URL('../control/transferWorker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    const finish = () => {
      w.terminate();
      if (workerRef.current === w) workerRef.current = null;
      setProgress(null);
    };
    w.onmessage = (e: MessageEvent<TransferReply>) => {
      const m = e.data;
      if (m.type === 'progress') return setProgress(m.fraction);
      finish();
      if (m.type === 'error') return notify(`Design failed: ${m.message}`, 'error');
      const T = m.transfer;
      try {
        setTransfer(T);
        setMeasuredZones({ bright, dark });
        const res: Record<string, number> = { uniform: contrastDb(T, spk.map(() => cx(1))) };
        for (const k of Object.keys(DESIGN_LABELS) as Design[]) res[k] = contrastDb(T, design(k, T, spk, params, bright));
        setPredicted(res);
        applyWeights(T, method, bright, spk, ids);
      } catch (err) {
        notify(`Design failed: ${(err as Error).message}`, 'error');
      }
    };
    w.onerror = (e) => {
      finish();
      notify(`Design failed: ${e.message || 'worker error'}`, 'error');
    };
    setProgress(0);
    const req: TransferRequest = { g, speakers: spk, bright, dark, f: freq };
    w.postMessage(req);
  };

  const cancelDesign = () => {
    workerRef.current?.terminate();
    workerRef.current = null;
    setProgress(null);
  };

  // Apply is only meaningful for the zones the transfer functions were measured for.
  const applyBlocked = !zones.bright || !zones.dark ? 'Draw a loud and a quiet zone first.' : !sameRect(zones.bright, measuredZones?.bright ?? null) || !sameRect(zones.dark, measuredZones?.dark ?? null) ? 'The zones changed since the measurement: measure again.' : null;

  const applyDesign = (T: Transfer, k: Design | 'uniform') => {
    if (applyBlocked || !zones.bright) return notify(applyBlocked ?? 'Draw a loud zone first.', 'error');
    applyWeights(T, k, zones.bright, speakers, array.map((d) => d.id));
  };

  const applyWeights = (T: Transfer, k: Design | 'uniform', bright: Rect, speakers: number[][], ids: string[]) => {
    const w = k === 'uniform' ? speakers.map(() => cx(1)) : design(k, T, speakers, scene.params, bright);
    replaceDrivers(
      PREFIX,
      weightedDrivers(speakers, w, T.f, 1, ids),
    );
    runtime.sim.resetAccumulators();
    setView({ overlay: 'rms' });
  };

  return (
    <div data-testid="control-panel">
      <h3>Sound-field control</h3>
      <details className="about">
        <summary>Make one zone loud and another quiet with a speaker array.</summary>
        <p className="hint">The designs use transfer functions measured in this room, walls included.</p>
      </details>

      <h3>1 · Zones</h3>
      <div className="btn-row">
        {(['bright', 'dark'] as const).map((k) => (
          <button
            key={k}
            className="btn sm"
            aria-pressed={tool === 'zone' && zoneKind === k}
            onClick={() => {
              setZoneKind(k);
              setTool('zone');
            }}
            data-testid={`zone-tool-${k}`}
          >
            {k === 'bright' ? 'Draw loud zone' : 'Draw quiet zone'} {zones[k] ? '✓' : ''}
          </button>
        ))}
        <button className="btn sm ghost" onClick={() => (setZone('bright', null), setZone('dark', null))} data-testid="clear-zones">
          Clear zones
        </button>
      </div>

      <h3>2 · Speaker array</h3>
      <div className="field-grid">
        <NumberField label="Speakers" value={count} integer min={2} max={24} onChange={setCount} testId="array-count" />
        <NumberField label="Spacing (cells)" value={spacing} integer min={1} max={40} onChange={setSpacing} />
        <NumberField label="Centre row" value={centreRow} integer min={1} max={rows - 2} onChange={setCentreRow} />
        <NumberField label="Centre column" value={centreCol} integer min={1} max={cols - 2} onChange={setCentreCol} />
        <div className="field">
          <label>Orientation</label>
          <select className="input" value={axis} onChange={(e) => setAxis(Number(e.target.value) as 0 | 1)} aria-label="Array orientation">
            <option value={1}>Horizontal</option>
            <option value={0}>Vertical</option>
          </select>
        </div>
      </div>
      <div className="btn-row">
        <button className="btn" onClick={placeArray} data-testid="place-array">
          Place array ({count} speakers)
        </button>
      </div>

      <h3>3 · Design</h3>
      <div className="field-grid">
        <NumberField label="Frequency" value={freq} min={0.005} max={0.2} onChange={setFreq} hint={hz(freq)} testId="control-freq" />
        <div className="field">
          <label>Method</label>
          <select className="input" value={method} onChange={(e) => setMethod(e.target.value as Design)} aria-label="Control method">
            {(Object.keys(DESIGN_LABELS) as Design[]).map((k) => (
              <option key={k} value={k}>
                {DESIGN_LABELS[k]}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="btn-row">
        <button className="btn primary" onClick={runDesign} disabled={progress !== null} data-testid="design">
          {progress !== null ? `Measuring… ${Math.round(progress * 100)} %` : 'Measure room & design'}
        </button>
        {progress !== null && (
          <button className="btn ghost" onClick={cancelDesign} data-testid="cancel-design">
            Cancel
          </button>
        )}
        {transfer && (
          <>
            <button className="btn" onClick={() => applyDesign(transfer, method)} disabled={!!applyBlocked} title={applyBlocked ?? undefined} data-testid="apply-design">
              Apply design
            </button>
            <button className="btn ghost" onClick={() => applyDesign(transfer, 'uniform')} disabled={!!applyBlocked} title={applyBlocked ?? undefined}>
              All in phase
            </button>
          </>
        )}
      </div>
      {transfer && applyBlocked && (
        <p className="hint" role="status" data-testid="apply-blocked" style={{ fontSize: 12 }}>
          {applyBlocked}
        </p>
      )}

      {predicted && (
        <table className="data" data-testid="predicted">
          <thead>
            <tr>
              <th>design</th>
              <th>predicted contrast</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(predicted).map(([k, v]) => (
              <tr key={k}>
                <td>{k === 'uniform' ? 'All in phase' : DESIGN_LABELS[k as Design]}</td>
                <td className="mono">{v.toFixed(1)} dB</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3>Live readout</h3>
      <div className="metric-row">
        <div className="metric" data-testid="live-contrast">
          <div className="v">{live === null ? '—' : `${live.toFixed(1)} dB`}</div>
          <div className="k">measured loud / quiet contrast (time-averaged)</div>
        </div>
      </div>
      <div className="btn-row">
        <button className="btn sm" onClick={() => runtime.sim.resetAccumulators()}>
          Reset averaging
        </button>
        <button className="btn sm" onClick={() => setView({ overlay: 'rms' })}>
          Show loudness map
        </button>
      </div>
      <details className="about">
        <summary>Run the simulation after applying a design.</summary>
        <p className="hint dim" style={{ fontSize: 12 }}>
          The averaged map settles once the tone has filled the room; press Reset averaging to drop the start-up transient.
        </p>
      </details>
    </div>
  );
}
