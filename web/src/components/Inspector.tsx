import { Download, Eye, EyeOff, Trash2, Upload } from 'lucide-react';
import { useRef } from 'react';
import { PRESETS } from '../engine/presets';
import { validateScene } from '../engine/scene';
import { absorptionFromBeta, facesOf, MATERIAL_NAMES, MATERIALS, type OuterKind } from '../engine/simulation';
import type { WaveformSpec } from '../engine/waveforms';
import { download } from '../lib/exporters';
import { DIVERGING, SEQUENTIAL } from '../render/colormaps';
import { fromScene, toScene } from '../state/editable';
import { type InspectorTab, useApp } from '../state/store';
import { ControlPanel } from './ControlPanel';
import { NumberField, WaveformEditor } from './fields';
import { SensingPanel } from './SensingPanel';

const TABS: { id: InspectorTab; label: string }[] = [
  { id: 'scene', label: 'Scene' },
  { id: 'sources', label: 'Sources' },
  { id: 'probes', label: 'Mics' },
  { id: 'view', label: 'View' },
  { id: 'sensing', label: 'Sensing' },
  { id: 'control', label: 'Control' },
];

export function Inspector() {
  const tab = useApp((s) => s.inspectorTab);
  const setTab = useApp((s) => s.setInspectorTab);
  return (
    <aside className="inspector" aria-label="Inspector">
      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)} data-testid={`tab-${t.id}`}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="panel" role="tabpanel">
        {tab === 'scene' && <ScenePanel />}
        {tab === 'sources' && <SourcesPanel />}
        {tab === 'probes' && <ProbesPanel />}
        {tab === 'view' && <ViewPanel />}
        {tab === 'sensing' && <SensingPanel />}
        {tab === 'control' && <ControlPanel />}
      </div>
    </aside>
  );
}

/** Scale all time-dimensioned waveform parameters by `k` (t_new = k t_old). */
function rescaleWaveform(w: WaveformSpec, k: number): WaveformSpec {
  switch (w.type) {
    case 'ricker':
      return { ...w, frequency: w.frequency / k, delay: w.delay * k };
    case 'gaussian':
      return { ...w, center_time: w.center_time * k, width: w.width * k };
    case 'cosine':
      return { ...w, frequency: w.frequency / k };
    case 'chirp':
      return { ...w, f0: w.f0 / k, f1: w.f1 / k, duration: w.duration * k, delay: w.delay * k };
    case 'burst':
      return { ...w, frequency: w.frequency / k, delay: w.delay * k };
    case 'noise':
      return { ...w, duration: w.duration * k };
    case 'samples':
      return { ...w, rate: w.rate / k, delay: w.delay * k };
  }
}

function ScenePanel() {
  const scene = useApp((s) => s.scene);
  const setParams = useApp((s) => s.setParams);
  const paintMaterial = useApp((s) => s.paintMaterial);
  const setPaintMaterial = useApp((s) => s.setPaintMaterial);
  const runtime = useApp((s) => s.runtime);
  const notify = useApp((s) => s.notify);
  const fileRef = useRef<HTMLInputElement>(null);
  const p = scene.params;
  const si = scene.units === 'si';

  const setUnits = (units: 'grid' | 'si') => {
    if (units === scene.units) return;
    const st = useApp.getState();
    // Grid units: c = dx = 1. SI: c = 343 m/s, dx = 2 cm by default.
    const next = units === 'si' ? { c: 343, dx: 0.02 } : { c: 1, dx: 1 };
    const kOld = p.dx / p.c; // seconds (or grid time) per unit of (dx/c)
    const kNew = next.dx / next.c;
    const k = kNew / kOld;
    st.checkpoint();
    const drivers = scene.drivers.map((d) => ({ ...d, waveform: rescaleWaveform(d.waveform, k), delay: (d.delay ?? 0) * k }));
    const loaded = { ...scene, units, drivers, params: { ...p, ...next } };
    st.loadScene(loaded);
    st.setNewWaveform(rescaleWaveform(st.newWaveform, k));
  };

  const onFile = async (f: File) => {
    try {
      const s = validateScene(JSON.parse(await f.text()));
      useApp.getState().loadScene(fromScene(s));
      notify(`Loaded “${s.name}”`);
    } catch (e) {
      notify(`Could not load scene: ${(e as Error).message}`, 'error');
    }
  };

  const shape = p.shape;
  const dt = runtime.sim.dt;
  return (
    <div>
      <h3>Scene</h3>
      <div className="field">
        <label>Name</label>
        <input className="input" value={scene.name} onChange={(e) => useApp.getState().setName(e.target.value)} aria-label="Scene name" />
      </div>
      {scene.description && <p style={{ fontSize: 12.5 }}>{scene.description}</p>}
      <div className="row">
        <select
          className="input"
          aria-label="Load preset"
          value=""
          onChange={(e) => {
            const pr = PRESETS.find((x) => x.id === e.target.value);
            if (pr) useApp.getState().loadScene(fromScene(pr.build()));
          }}
        >
          <option value="">Load a preset…</option>
          {PRESETS.map((pr) => (
            <option key={pr.id} value={pr.id}>
              {pr.title}
            </option>
          ))}
        </select>
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <button
          className="btn sm"
          onClick={() =>
            download(new Blob([JSON.stringify(toScene(useApp.getState().scene), null, 1)], { type: 'application/json' }), `${scene.name || 'scene'}.json`)
          }
        >
          <Download size={14} /> Save
        </button>
        <button className="btn sm" onClick={() => fileRef.current?.click()}>
          <Upload size={14} /> Open
        </button>
        <input ref={fileRef} type="file" accept="application/json,.json" hidden onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
      </div>

      <h3>Grid</h3>
      <div className="field">
        <label>Dimensions</label>
        <div className="seg" role="group" aria-label="Dimensions">
          {[2, 3].map((d) => (
            <button
              key={d}
              aria-pressed={p.dims === d}
              onClick={() => d !== p.dims && setParams({ dims: d as 2 | 3, shape: d === 3 ? [64, 64, 64] : [200, 200] })}
            >
              {d}D
            </button>
          ))}
        </div>
      </div>
      <div className="row">
        {shape.map((n, a) => (
          <NumberField
            key={a}
            label={['Rows', 'Cols', 'Depth'][a]}
            value={n}
            integer
            min={16}
            max={p.dims === 3 ? 160 : 1024}
            testId={`grid-${a}`}
            onChange={(v) => setParams({ shape: shape.map((x, b) => (b === a ? v : x)) })}
          />
        ))}
      </div>
      <div className="field">
        <label>Units</label>
        <div className="seg" role="group" aria-label="Units">
          <button aria-pressed={!si} onClick={() => setUnits('grid')}>
            Grid (c = Δx = 1)
          </button>
          <button aria-pressed={si} onClick={() => setUnits('si')}>
            SI (m, s, Hz)
          </button>
        </div>
      </div>
      {si && (
        <div className="row">
          <NumberField label="Cell size Δx (m)" value={p.dx} min={1e-5} max={10} onChange={(v) => setParams({ dx: v })} />
          <NumberField label="Sound speed c (m/s)" value={p.c} min={1} max={10000} onChange={(v) => setParams({ c: v })} />
        </div>
      )}
      <NumberField
        label="Courant number σ = cΔt/Δx"
        value={p.courant}
        min={0.01}
        max={1}
        hint={`Stable for σ ≤ 1/√${p.dims} = ${(1 / Math.sqrt(p.dims)).toFixed(3)}; clamped to 0.95 of that. Δt = ${si ? `${(dt * 1e6).toFixed(2)} µs` : dt.toFixed(3)}.`}
        onChange={(v) => setParams({ courant: v })}
      />
      {si && (
        <p className="hint dim" style={{ fontSize: 12 }}>
          Domain {shape.map((n) => (n * p.dx).toFixed(2)).join(' × ')} m · highest resolvable frequency ≈ {((p.c / (8 * p.dx)) / 1000).toFixed(1)} kHz (8 cells/λ)
        </p>
      )}

      <h3>Boundaries</h3>
      <div className="field">
        <label>Outer walls</label>
        <select className="input" value={p.outer} onChange={(e) => setParams({ outer: e.target.value as OuterKind })} aria-label="Outer boundary">
          <option value="soft">Pressure-release (p = 0)</option>
          <option value="rigid">Rigid (∂p/∂n = 0)</option>
          <option value="absorb">Impedance (partially absorbing)</option>
          <option value="mur">Absorbing edge (Mur, 1st order)</option>
          <option value="sponge">Absorbing layer (sponge)</option>
          <option value="cpml">Anechoic (PML, best)</option>
        </select>
      </div>
      <label className="row tight" style={{ marginBottom: 8 }}>
        <input
          type="checkbox"
          checked={!!p.faces}
          onChange={(e) => setParams({ faces: e.target.checked ? facesOf(p) : undefined })}
          aria-label="Set each face separately"
        />{' '}
        Set each face separately
      </label>
      {p.faces && (
        <div className="row wrap" style={{ alignItems: 'flex-start' }}>
          {facesOf(p).map((f, k) => (
            <div key={k} className="field" style={{ flex: '1 1 45%' }}>
              <label>{['Top', 'Bottom', 'Left', 'Right', 'Front', 'Back'][k]}</label>
              <select
                className="input"
                value={f}
                aria-label={`${['Top', 'Bottom', 'Left', 'Right', 'Front', 'Back'][k]} face`}
                onChange={(e) => setParams({ faces: facesOf(p).map((x, q) => (q === k ? (e.target.value as OuterKind) : x)) })}
              >
                <option value="soft">p = 0</option>
                <option value="rigid">Rigid</option>
                <option value="absorb">Impedance</option>
                <option value="mur">Mur edge</option>
                <option value="sponge">Absorbing layer</option>
                <option value="cpml">PML</option>
              </select>
            </div>
          ))}
        </div>
      )}
      {facesOf(p).includes('absorb') && (
        <NumberField
          label="Wall admittance β"
          value={p.outerBeta}
          min={0}
          max={1}
          hint={`Normal-incidence absorption α = ${absorptionFromBeta(p.outerBeta).toFixed(2)}`}
          onChange={(v) => setParams({ outerBeta: v })}
        />
      )}
      {facesOf(p).includes('sponge') && (
        <NumberField label="Layer thickness (cells)" value={p.spongeCells} integer min={4} max={80} onChange={(v) => setParams({ spongeCells: v })} />
      )}
      {facesOf(p).includes('cpml') && (
        <NumberField
          label="PML thickness (cells)"
          value={p.cpmlCells ?? 16}
          integer
          min={4}
          max={64}
          hint="Reflects < −45 dB up to 60° incidence at 16 cells"
          onChange={(v) => setParams({ cpmlCells: v })}
          testId="cpml-cells"
        />
      )}

      <h3>Wall material (brush)</h3>
      <div className="material-grid" role="radiogroup" aria-label="Wall material">
        {MATERIALS.map((m, id) =>
          id === 0 ? null : (
            <button
              key={id}
              className="btn sm"
              role="radio"
              aria-checked={paintMaterial === id}
              style={paintMaterial === id ? { borderColor: 'var(--accent)', color: 'var(--accent)' } : undefined}
              title={m.kind === 'absorb' ? `α ≈ ${absorptionFromBeta(m.beta).toFixed(2)}` : m.kind}
              onClick={() => setPaintMaterial(id)}
            >
              {MATERIAL_NAMES[id]}
            </button>
          ),
        )}
      </div>
      <h3>Sound speed (brush C)</h3>
      <SpeedControls />
      <button className="btn sm danger" style={{ marginTop: 10 }} onClick={() => useApp.getState().clearObstacles()}>
        <Trash2 size={14} /> Clear all walls
      </button>
    </div>
  );
}

function SpeedControls() {
  const paintSpeed = useApp((s) => s.paintSpeed);
  const setPaintSpeed = useApp((s) => s.setPaintSpeed);
  const scene = useApp((s) => s.scene);
  const runtime = useApp((s) => s.runtime);
  // Local Courant sigma * r must stay <= 1/sqrt(d) (clamped 0.99 of that).
  const sigma = Math.sqrt(runtime.sim.coeff);
  const maxRatio = Math.max(1, (0.99 / Math.sqrt(scene.params.dims)) / sigma);
  return (
    <div>
      <div className="field">
        <label>
          <span>Speed ratio c(x)/c</span>
          <span className="mono">{paintSpeed.toFixed(2)}×</span>
        </label>
        <input
          type="range"
          min={0.3}
          max={maxRatio}
          step={0.01}
          value={Math.min(paintSpeed, maxRatio)}
          onChange={(e) => setPaintSpeed(Number(e.target.value))}
          aria-label="Speed ratio"
        />
        <span className="hint">
          Slower regions bend waves towards them (a lens). Up to {maxRatio.toFixed(2)}× keeps the scheme stable at this Courant number; paint 1.00 to restore.
        </span>
      </div>
      {scene.speed && (
        <button className="btn sm" onClick={() => useApp.getState().setSpeed(null)}>
          Reset sound speed
        </button>
      )}
    </div>
  );
}

function SourcesPanel() {
  const scene = useApp((s) => s.scene);
  const selected = useApp((s) => s.selected);
  const newWaveform = useApp((s) => s.newWaveform);
  const setNewWaveform = useApp((s) => s.setNewWaveform);
  const runtime = useApp((s) => s.runtime);
  const { dt } = runtime.sim;
  const { dx, c } = scene.params;
  return (
    <div>
      <h3>New sources</h3>
      <p style={{ fontSize: 12.5 }}>
        Pick the <b>source tool</b> (S) and click the field. New sources use this waveform:
      </p>
      <WaveformEditor value={newWaveform} onChange={setNewWaveform} dt={dt} dx={dx} c={c} />
      <h3>Sources ({scene.drivers.length})</h3>
      {scene.drivers.length === 0 && <p className="dim">No sources yet.</p>}
      {scene.drivers.map((d, i) => (
        <div key={d.id} className={`card${selected?.id === d.id ? ' selected' : ''}`} data-testid="driver-card">
          <div className="card-head">
            <span className="swatch" style={{ background: 'var(--driver)' }} />
            <span className="title">
              Source {i + 1} <span className="dim mono">[{d.pos.join(', ')}]</span>
            </span>
            <button
              className="btn icon sm ghost"
              title={d.enabled ? 'Mute' : 'Unmute'}
              aria-label={d.enabled ? 'Mute source' : 'Unmute source'}
              onClick={() => useApp.getState().updateDriver(d.id, { enabled: !d.enabled })}
            >
              {d.enabled ? <Eye size={14} /> : <EyeOff size={14} />}
            </button>
            <button className="btn icon sm ghost danger" title="Remove" aria-label="Remove source" onClick={() => useApp.getState().removeDriver(d.id)}>
              <Trash2 size={14} />
            </button>
          </div>
          {selected?.id === d.id && (
            <>
              <WaveformEditor value={d.waveform} onChange={(w) => useApp.getState().updateDriver(d.id, { waveform: w })} dt={dt} dx={dx} c={c} />
              <div className="row">
                <NumberField label="Gain" value={d.gain ?? 1} min={-100} max={100} onChange={(v) => useApp.getState().updateDriver(d.id, { gain: v })} />
                <NumberField label="Extra delay" value={d.delay ?? 0} min={0} max={1e9} onChange={(v) => useApp.getState().updateDriver(d.id, { delay: v })} />
              </div>
            </>
          )}
          {selected?.id !== d.id && (
            <button className="btn sm ghost" onClick={() => useApp.getState().select({ kind: 'driver', id: d.id })}>
              Edit waveform…
            </button>
          )}
        </div>
      ))}
      {scene.drivers.length > 0 && (
        <button className="btn sm danger" onClick={() => useApp.getState().clearDrivers()}>
          <Trash2 size={14} /> Remove all sources
        </button>
      )}
    </div>
  );
}

function ProbesPanel() {
  const scene = useApp((s) => s.scene);
  const selected = useApp((s) => s.selected);
  return (
    <div>
      <h3>Microphones ({scene.probes.length})</h3>
      <p style={{ fontSize: 12.5 }}>
        Place microphones with the <b>mic tool</b> (M). Their recordings appear in the panel below the field, where you can also listen to them.
      </p>
      {scene.probes.map((p) => (
        <div key={p.id} className={`card${selected?.id === p.id ? ' selected' : ''}`} data-testid="probe-card">
          <div className="card-head">
            <span className="swatch" style={{ background: 'var(--probe)' }} />
            <input
              className="input"
              value={p.label ?? ''}
              aria-label="Microphone name"
              onChange={(e) => useApp.getState().updateProbe(p.id, { label: e.target.value })}
            />
            <span className="dim mono" style={{ flex: 'none' }}>
              [{p.pos.join(', ')}]
            </span>
            <button className="btn icon sm ghost danger" title="Remove" aria-label="Remove microphone" onClick={() => useApp.getState().removeProbe(p.id)}>
              <Trash2 size={14} />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function ViewPanel() {
  const view = useApp((s) => s.view);
  const setView = useApp((s) => s.setView);
  const scene = useApp((s) => s.scene);
  const is3d = scene.params.dims === 3;
  const maps = view.overlay === 'rms' ? SEQUENTIAL : DIVERGING;
  return (
    <div>
      <h3>Quantity</h3>
      <div className="seg" role="group" aria-label="Displayed quantity">
        {(
          [
            ['pressure', 'Pressure'],
            ['rms', 'RMS level'],
            ['intensity', 'Energy flow'],
          ] as const
        ).map(([id, label]) => (
          <button key={id} aria-pressed={view.overlay === id} onClick={() => setView({ overlay: id, colormap: id === 'rms' ? 'magma' : useApp.getState().theme === 'dark' ? 'icefire' : 'balance' })}>
            {label}
          </button>
        ))}
      </div>
      <p className="hint dim" style={{ fontSize: 12, marginTop: 6 }}>
        {view.overlay === 'rms'
          ? 'Root-mean-square pressure since the quantity was selected (or the last reset): where it is loud on average.'
          : view.overlay === 'intensity'
            ? 'Arrows show the time-averaged acoustic intensity ⟨p v⟩: the direction energy flows.'
            : 'Instantaneous pressure: positive and negative parts of the wave.'}
      </p>
      <h3>Colour</h3>
      <div className="field">
        <label>Colormap</label>
        <select className="input" value={view.colormap} onChange={(e) => setView({ colormap: e.target.value as typeof view.colormap })} aria-label="Colormap">
          {maps.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label>Scale</label>
        <div className="seg" role="group" aria-label="Scale">
          <button aria-pressed={view.mode === 'linear'} onClick={() => setView({ mode: 'linear' })}>
            Linear
          </button>
          <button aria-pressed={view.mode === 'db'} onClick={() => setView({ mode: 'db' })}>
            Decibels
          </button>
        </div>
      </div>
      {view.mode === 'db' && (
        <div className="field">
          <label>
            <span>Dynamic range</span>
            <span className="mono">{view.dbRange} dB</span>
          </label>
          <input type="range" min={10} max={100} value={view.dbRange} onChange={(e) => setView({ dbRange: Number(e.target.value) })} aria-label="Dynamic range" />
        </div>
      )}
      <label className="row tight" style={{ marginBottom: 8 }}>
        <input type="checkbox" checked={view.autoScale} onChange={(e) => setView({ autoScale: e.target.checked })} /> Auto-scale to the peak
      </label>
      {!view.autoScale && <NumberField label="Full-scale value" value={view.fixedScale} min={1e-9} max={1e9} onChange={(v) => setView({ fixedScale: v })} />}
      <label className="row tight">
        <input type="checkbox" checked={view.showMaterials} onChange={(e) => setView({ showMaterials: e.target.checked })} /> Show walls
      </label>
      {is3d && (
        <>
          <h3>3D slice</h3>
          <div className="seg" role="group" aria-label="3D view">
            <button aria-pressed={view.view3d === 'slice'} onClick={() => setView({ view3d: 'slice' })}>
              Slice
            </button>
            <button aria-pressed={view.view3d === 'volume'} onClick={() => setView({ view3d: 'volume' })}>
              Volume
            </button>
          </div>
          <div className="field" style={{ marginTop: 10 }}>
            <label>Slice axis</label>
            <div className="seg" role="group" aria-label="Slice axis">
              {(['x', 'y', 'z'] as const).map((a, i) => (
                <button
                  key={a}
                  aria-pressed={view.sliceAxis === i}
                  onClick={() => setView({ sliceAxis: i as 0 | 1 | 2, sliceIndex: Math.floor(scene.params.shape[i] / 2) })}
                >
                  {a}
                </button>
              ))}
            </div>
          </div>
          <div className="field">
            <label>
              <span>Slice position</span>
              <span className="mono">{view.sliceIndex}</span>
            </label>
            <input
              type="range"
              min={0}
              max={scene.params.shape[view.sliceAxis] - 1}
              value={view.sliceIndex}
              onChange={(e) => setView({ sliceIndex: Number(e.target.value) })}
              aria-label="Slice position"
            />
          </div>
          <p className="hint dim" style={{ fontSize: 12 }}>
            Drawing tools paint in the current slice plane.
          </p>
        </>
      )}
    </div>
  );
}
