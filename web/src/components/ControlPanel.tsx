import { useEffect, useState } from 'react';
import {
  contrastDb,
  design,
  type Design,
  DESIGN_LABELS,
  type Geometry,
  lineArray,
  measureTransfer,
  type Transfer,
  weightedDrivers,
  zoneContrastFromEnergy,
} from '../control/soundfield';
import { cx } from '../control/complex';
import { useApp } from '../state/store';
import { NumberField } from './fields';

const PREFIX = 'arr-';

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

  const is2d = scene.params.dims === 2;
  const array = scene.drivers.filter((d) => d.id.startsWith(PREFIX));
  const speakers = array.map((d) => d.pos);
  const si = scene.units === 'si';
  const hz = (f: number) => (si ? `${((f * scene.params.c) / scene.params.dx).toFixed(0)} Hz` : `${f} cycles/unit`);

  // Live contrast from the loudness map, ~4 Hz.
  useEffect(() => {
    if (!zones.bright || !zones.dark) return;
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

  const runDesign = async () => {
    if (!zones.bright || !zones.dark) return notify('Draw a loud zone and a quiet zone first.', 'error');
    if (speakers.length < 2) return notify('Place a speaker array first.', 'error');
    const g: Geometry = { params: scene.params, materials: scene.materials, speed: scene.speed };
    setProgress(0);
    try {
      const T = await measureTransfer(g, speakers, zones.bright, zones.dark, freq, { onProgress: setProgress });
      setTransfer(T);
      const res: Record<string, number> = { uniform: contrastDb(T, speakers.map(() => cx(1))) };
      for (const k of Object.keys(DESIGN_LABELS) as Design[]) res[k] = contrastDb(T, design(k, T, speakers, scene.params, zones.bright));
      setPredicted(res);
      applyDesign(T, method);
    } catch (e) {
      notify(`Design failed: ${(e as Error).message}`, 'error');
    } finally {
      setProgress(null);
    }
  };

  const applyDesign = (T: Transfer, k: Design | 'uniform') => {
    if (!zones.bright) return;
    const w = k === 'uniform' ? speakers.map(() => cx(1)) : design(k, T, speakers, scene.params, zones.bright);
    replaceDrivers(
      PREFIX,
      weightedDrivers(speakers, w, T.f, 1, array.map((d) => d.id)),
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
        <button className="btn sm ghost" onClick={() => (setZone('bright', null), setZone('dark', null))}>
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
        {transfer && (
          <>
            <button className="btn" onClick={() => applyDesign(transfer, method)} data-testid="apply-design">
              Apply design
            </button>
            <button className="btn ghost" onClick={() => applyDesign(transfer, 'uniform')}>
              All in phase
            </button>
          </>
        )}
      </div>

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
