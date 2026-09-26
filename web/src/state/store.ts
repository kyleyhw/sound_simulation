/**
 * Application state (zustand). The scene is the single source of truth for
 * geometry/sources; the Runtime mirrors it into the live Simulation. Edits
 * go through actions so undo/redo snapshots stay consistent.
 */

import { create } from 'zustand';
import { PRESETS } from '../engine/presets';
import { newId } from '../engine/scene';
import type { DriverSpec, ProbeSpec, SimParams } from '../engine/simulation';
import { defaultWaveform, type WaveformSpec } from '../engine/waveforms';
import type { ColormapName } from '../render/colormaps';
import type { ScaleMode } from '../render/fieldRenderer';
import { cloneScene, type EditableScene, fromScene, sceneEquals } from './editable';
import { Runtime } from './runtime';

export type Tool = 'select' | 'brush' | 'eraser' | 'line' | 'rect' | 'ellipse' | 'speed' | 'driver' | 'probe' | 'zone';
export type ZoneKind = 'bright' | 'dark';
/** Control zone rectangle [r0, c0, r1, c1] in 2D cells. */
export type ZoneRect = [number, number, number, number];
export type Overlay = 'pressure' | 'rms' | 'intensity';
export type Theme = 'dark' | 'light';
export type InspectorTab = 'scene' | 'sources' | 'probes' | 'view' | 'sensing' | 'control';

export interface ViewState {
  colormap: ColormapName;
  mode: ScaleMode;
  dbRange: number;
  autoScale: boolean;
  fixedScale: number;
  overlay: Overlay;
  showMaterials: boolean;
  showGrid: boolean;
  sliceAxis: 0 | 1 | 2;
  sliceIndex: number;
  view3d: 'slice' | 'volume';
}

const MAX_HISTORY = 60;

function initialScene(): EditableScene {
  return fromScene(PRESETS[0].build());
}

function readTheme(): Theme {
  try {
    const t = localStorage.getItem('theme');
    if (t === 'light' || t === 'dark') return t;
  } catch {
    /* storage unavailable */
  }
  return typeof matchMedia !== 'undefined' && matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export interface AppState {
  scene: EditableScene;
  runtime: Runtime;
  tool: Tool;
  brushSize: number;
  paintMaterial: number;
  /** Relative sound speed painted by the speed tool. */
  paintSpeed: number;
  newWaveform: WaveformSpec;
  view: ViewState;
  theme: Theme;
  inspectorTab: InspectorTab;
  selected: { kind: 'driver' | 'probe'; id: string } | null;
  past: EditableScene[];
  future: EditableScene[];
  toast: { text: string; kind: 'info' | 'error' } | null;
  showHelp: boolean;
  /** Sound-field control zones (plan 7.7.2): loud (bright) and quiet (dark). */
  zones: { bright: ZoneRect | null; dark: ZoneRect | null };
  zoneKind: ZoneKind;

  setTool: (t: Tool) => void;
  setBrushSize: (n: number) => void;
  setPaintMaterial: (m: number) => void;
  setPaintSpeed: (v: number) => void;
  /** Paint relative sound speed into cells (1 restores nominal). */
  paintSpeedCells: (indices: number[], value: number) => void;
  setNewWaveform: (w: WaveformSpec) => void;
  setView: (v: Partial<ViewState>) => void;
  setTheme: (t: Theme) => void;
  setInspectorTab: (t: InspectorTab) => void;
  select: (s: AppState['selected']) => void;
  notify: (text: string, kind?: 'info' | 'error') => void;
  setShowHelp: (v: boolean) => void;
  setZone: (kind: ZoneKind, r: ZoneRect | null) => void;
  setZoneKind: (k: ZoneKind) => void;
  /** Replace the drivers with ids starting `prefix` (array tools); undoable. */
  replaceDrivers: (prefix: string, drivers: DriverSpec[]) => void;

  /**
   * Record the current scene on the undo stack (call before an edit). A
   * snapshot identical to the last one is not recorded again. Consecutive
   * calls with the same `coalesceKey` (e.g. keystrokes of one rename) share
   * one undo entry.
   */
  checkpoint: (coalesceKey?: string) => void;
  undo: () => void;
  redo: () => void;

  loadScene: (s: EditableScene) => void;
  setParams: (p: Partial<SimParams>) => void;
  /** Paint material id into cells (flat indices); call checkpoint() first. */
  paintCells: (indices: number[], material: number) => void;
  clearObstacles: () => void;
  addDriver: (pos: number[], waveform?: WaveformSpec) => string;
  updateDriver: (id: string, patch: Partial<DriverSpec>) => void;
  removeDriver: (id: string) => void;
  clearDrivers: () => void;
  addProbe: (pos: number[], label?: string) => string;
  updateProbe: (id: string, patch: Partial<ProbeSpec>) => void;
  removeProbe: (id: string) => void;
  setSpeed: (speed: Float32Array | null) => void;
  /** Move a marker without an undo checkpoint (call checkpoint() at drag start). */
  moveMarker: (kind: 'driver' | 'probe', id: string, pos: number[]) => void;
  setName: (name: string) => void;
}

/** Zones are 2D cell rectangles of one grid: drop them when the grid changes. */
const NO_ZONES = { bright: null, dark: null };
const sameGrid = (a: EditableScene, b: EditableScene) => a.params.dims === b.params.dims && a.params.shape.join() === b.params.shape.join();

export const useApp = create<AppState>((set, get) => {
  const scene = initialScene();
  const runtime = new Runtime(scene);
  const theme = readTheme();
  /** Coalescing key of the last checkpoint (see `checkpoint`). */
  let lastKey: string | null = null;

  const commitSources = (next: EditableScene) => {
    set({ scene: next, future: [] });
    get().runtime.syncSources(next);
    get().runtime.renderNow();
  };

  return {
    scene,
    runtime,
    tool: 'brush',
    brushSize: 3,
    paintMaterial: 2,
    paintSpeed: 0.6,
    newWaveform: defaultWaveform('ricker'),
    view: {
      colormap: theme === 'dark' ? 'icefire' : 'balance',
      mode: 'linear',
      dbRange: 50,
      autoScale: true,
      fixedScale: 1,
      overlay: 'pressure',
      showMaterials: true,
      showGrid: false,
      sliceAxis: 2,
      sliceIndex: 0,
      view3d: 'slice',
    },
    theme,
    inspectorTab: 'scene',
    selected: null,
    past: [],
    future: [],
    toast: null,
    showHelp: false,

    setTool: (tool) => set({ tool }),
    zones: { bright: null, dark: null },
    zoneKind: 'bright',
    setZone: (kind, r) => set((st) => ({ zones: { ...st.zones, [kind]: r } })),
    setZoneKind: (zoneKind) => set({ zoneKind }),
    replaceDrivers: (prefix, drivers) => {
      get().checkpoint();
      const sc = get().scene;
      commitSources({ ...sc, drivers: [...sc.drivers.filter((d) => !d.id.startsWith(prefix)), ...drivers] });
    },
    setBrushSize: (brushSize) => set({ brushSize: Math.max(1, Math.min(40, Math.round(brushSize))) }),
    setPaintMaterial: (paintMaterial) => set({ paintMaterial }),
    setPaintSpeed: (paintSpeed) => set({ paintSpeed }),
    paintSpeedCells: (indices, value) => {
      const cur = get().scene;
      const speed = cur.speed ? cur.speed : new Float32Array(cur.materials.length).fill(1);
      for (const idx of indices) if (idx >= 0 && idx < speed.length) speed[idx] = value;
      let uniform = true;
      for (let i = 0; i < speed.length; i++) if (speed[i] !== 1) (uniform = false), (i = speed.length);
      const next = { ...cur, speed: uniform ? null : speed };
      set({ scene: next, future: [] });
      get().runtime.syncGeometry(next);
      get().runtime.renderNow();
    },
    setNewWaveform: (newWaveform) => set({ newWaveform }),
    setView: (v) => {
      const view = { ...get().view, ...v };
      set({ view });
      const sim = get().runtime.sim;
      if (v.overlay !== undefined) {
        sim.enableRms(view.overlay === 'rms');
        sim.enableIntensity(view.overlay === 'intensity');
        // Intensity (particle velocity) is only computed by the CPU engine.
        if (view.overlay === 'intensity' && get().runtime.backend === 'gpu') {
          void get().runtime.setBackend('cpu');
          get().notify('Intensity arrows run on the CPU engine; switched from the GPU.');
        }
      }
      get().runtime.renderNow();
    },
    setTheme: (theme) => {
      try {
        localStorage.setItem('theme', theme);
      } catch {
        /* ignore */
      }
      if (theme === 'light' && get().view.colormap === 'icefire') set({ view: { ...get().view, colormap: 'balance' } });
      if (theme === 'dark' && get().view.colormap === 'balance') set({ view: { ...get().view, colormap: 'icefire' } });
      set({ theme });
      get().runtime.renderNow();
    },
    setInspectorTab: (inspectorTab) => set({ inspectorTab }),
    select: (selected) => set({ selected }),
    notify: (text, kind = 'info') => {
      set({ toast: { text, kind } });
      window.setTimeout(() => {
        if (get().toast?.text === text) set({ toast: null });
      }, 3200);
    },
    setShowHelp: (showHelp) => set({ showHelp }),

    checkpoint: (coalesceKey) => {
      const key = typeof coalesceKey === 'string' ? coalesceKey : null;
      if (key !== null && key === lastKey) return;
      lastKey = key;
      const { past, scene } = get();
      if (past.length && sceneEquals(past[past.length - 1], scene)) return;
      const next = [...past, cloneScene(scene)];
      if (next.length > MAX_HISTORY) next.shift();
      set({ past: next, future: [] });
    },
    undo: () => {
      lastKey = null;
      const { future, scene } = get();
      let { past } = get();
      // Entries identical to the current scene (an edit that changed
      // nothing) would make this undo look like a no-op: skip them.
      while (past.length && sceneEquals(past[past.length - 1], scene)) past = past.slice(0, -1);
      if (past.length === 0) return set({ past });
      const prev = past[past.length - 1];
      set({ past: past.slice(0, -1), future: [cloneScene(scene), ...future], scene: prev });
      if (!sameGrid(prev, scene)) set({ zones: NO_ZONES });
      const rt = get().runtime;
      // Any parameter change (grid, units, c, dx, boundaries) needs a rebuild, not just a new shape.
      if (JSON.stringify(prev.params) !== JSON.stringify(scene.params)) rt.load(prev);
      else {
        rt.syncGeometry(prev);
        rt.syncSources(prev);
      }
      rt.renderNow();
    },
    redo: () => {
      lastKey = null;
      const { past, future, scene } = get();
      if (future.length === 0) return;
      const next = future[0];
      set({ past: [...past, cloneScene(scene)], future: future.slice(1), scene: next });
      if (!sameGrid(next, scene)) set({ zones: NO_ZONES });
      const rt = get().runtime;
      // Any parameter change (grid, units, c, dx, boundaries) needs a rebuild, not just a new shape.
      if (JSON.stringify(next.params) !== JSON.stringify(scene.params)) rt.load(next);
      else {
        rt.syncGeometry(next);
        rt.syncSources(next);
      }
      rt.renderNow();
    },

    loadScene: (s) => {
      get().checkpoint();
      const view = { ...get().view };
      if (s.params.dims === 3) view.sliceIndex = Math.floor(s.params.shape[2] / 2);
      // A new room: its control zones would lie off-grid or on the wrong walls.
      const zones = sameGrid(s, get().scene) && s.params.dims === 2 ? get().zones : NO_ZONES;
      set({ scene: s, selected: null, view, zones });
      get().runtime.load(s);
    },
    setParams: (p) => {
      get().checkpoint();
      const cur = get().scene;
      const params: SimParams = { ...cur.params, ...p, shape: [...(p.shape ?? cur.params.shape)] };
      const reshape = params.shape.join() !== cur.params.shape.join() || params.dims !== cur.params.dims;
      let next: EditableScene;
      if (reshape) {
        // Rescale geometry and sources into the new grid (nearest cell).
        const n = params.shape.reduce((a, b) => a * b, 1);
        const mat = new Uint8Array(n);
        const [ox, oy, oz] = [cur.params.shape[0], cur.params.shape[1], cur.params.shape[2] ?? 1];
        const [nx, ny, nz] = [params.shape[0], params.shape[1], params.shape[2] ?? 1];
        if (params.dims === cur.params.dims) {
          for (let i = 0; i < nx; i++)
            for (let j = 0; j < ny; j++)
              for (let k = 0; k < nz; k++) {
                const si = Math.min(ox - 1, Math.floor((i * ox) / nx));
                const sj = Math.min(oy - 1, Math.floor((j * oy) / ny));
                const sk = Math.min(oz - 1, Math.floor((k * oz) / nz));
                mat[(i * ny + j) * nz + k] = cur.materials[(si * oy + sj) * oz + sk];
              }
        }
        const mapPos = (pos: number[]): number[] =>
          params.shape.map((size, a) => {
            const old = cur.params.shape[a] ?? size;
            const v = pos[a] ?? Math.floor(size / 2);
            return Math.max(1, Math.min(size - 2, Math.round((v * size) / old)));
          });
        next = {
          ...cur,
          params,
          materials: mat,
          speed: null,
          drivers: cur.drivers.map((d) => ({ ...d, pos: mapPos(d.pos) })),
          probes: cur.probes.map((q) => ({ ...q, pos: mapPos(q.pos) })),
        };
        const view = { ...get().view };
        if (params.dims === 3) view.sliceIndex = Math.floor(nz / 2);
        set({ scene: next, view, zones: NO_ZONES });
      } else {
        next = { ...cur, params };
        set({ scene: next });
      }
      get().runtime.load(next);
    },
    paintCells: (indices, material) => {
      const cur = get().scene;
      const materials = cur.materials;
      for (const idx of indices) if (idx >= 0 && idx < materials.length) materials[idx] = material;
      const next = { ...cur, materials };
      set({ scene: next, future: [] });
      get().runtime.sim.setCells(indices, material);
      get().runtime.version++;
      get().runtime.renderNow();
    },
    clearObstacles: () => {
      get().checkpoint();
      const cur = get().scene;
      // Walls only: the painted sound-speed map has its own reset.
      const next = { ...cur, materials: new Uint8Array(cur.materials.length) };
      set({ scene: next });
      get().runtime.syncGeometry(next);
      get().runtime.renderNow();
    },
    addDriver: (pos, waveform) => {
      get().checkpoint();
      const id = newId('d');
      const d: DriverSpec = { id, pos, waveform: waveform ?? { ...get().newWaveform }, enabled: true };
      commitSources({ ...get().scene, drivers: [...get().scene.drivers, d] });
      set({ selected: { kind: 'driver', id } });
      return id;
    },
    updateDriver: (id, patch) => {
      get().checkpoint();
      commitSources({ ...get().scene, drivers: get().scene.drivers.map((d) => (d.id === id ? { ...d, ...patch } : d)) });
    },
    removeDriver: (id) => {
      get().checkpoint();
      commitSources({ ...get().scene, drivers: get().scene.drivers.filter((d) => d.id !== id) });
      if (get().selected?.id === id) set({ selected: null });
    },
    clearDrivers: () => {
      get().checkpoint();
      commitSources({ ...get().scene, drivers: [] });
    },
    addProbe: (pos, label) => {
      get().checkpoint();
      const id = newId('p');
      const n = get().scene.probes.length + 1;
      commitSources({ ...get().scene, probes: [...get().scene.probes, { id, pos, label: label ?? `Mic ${n}` }] });
      set({ selected: { kind: 'probe', id } });
      return id;
    },
    updateProbe: (id, patch) => {
      // Renaming fires per keystroke: one undo entry per rename, not per key.
      const keys = Object.keys(patch);
      get().checkpoint(keys.length === 1 && keys[0] === 'label' ? `probe-label:${id}` : undefined);
      commitSources({ ...get().scene, probes: get().scene.probes.map((p) => (p.id === id ? { ...p, ...patch } : p)) });
    },
    removeProbe: (id) => {
      get().checkpoint();
      commitSources({ ...get().scene, probes: get().scene.probes.filter((p) => p.id !== id) });
      if (get().selected?.id === id) set({ selected: null });
    },
    setSpeed: (speed) => {
      get().checkpoint();
      const next = { ...get().scene, speed };
      set({ scene: next });
      get().runtime.syncGeometry(next);
      get().runtime.renderNow();
    },
    setName: (name) => set({ scene: { ...get().scene, name } }),
    moveMarker: (kind, id, pos) => {
      const sc = get().scene;
      const next =
        kind === 'driver'
          ? { ...sc, drivers: sc.drivers.map((d) => (d.id === id ? { ...d, pos } : d)) }
          : { ...sc, probes: sc.probes.map((p) => (p.id === id ? { ...p, pos } : p)) };
      set({ scene: next, future: [] });
      get().runtime.syncSources(next);
      get().runtime.renderNow();
    },
  };
});

export { defaultWaveform };
