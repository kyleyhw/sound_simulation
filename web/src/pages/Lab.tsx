import { Camera, Download, Mic, Play, Smartphone, Square, Trash2, Upload } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { drawSeries } from '../components/Dock';
import { NumberField } from '../components/fields';
import { type CtcFilters, type CtcGeometry, designCtc, separationDb, stereoSeparationDb } from '../control/ctc';
import { playAndRecord, openAudio } from '../lab/audioio';
import { b64ToF32, type Capture, exportCaptures, f32ToB64, loadCalibration, loadCaptures, saveCalibration, saveCaptures, type StoredCalibration } from '../lab/captures';
import { HeadTracker, type HeadPose } from '../lab/headtrack';
import {
  alphaFromT60,
  decayMetrics,
  deconvolve,
  directPathWindow,
  type Echo,
  equalize,
  essInverse,
  essSweep,
  eyring,
  findEchoes,
  octaveBand,
  roundTripLatencyMs,
  sabine,
  shoeboxFirstEchoes,
} from '../lab/measure';
import { simulateShoebox, type TwinResult } from '../lab/twin';
import type { Route } from '../lib/router';
import { download } from '../lib/exporters';
import { useApp } from '../state/store';

interface Measurement {
  sampleRate: number;
  irs: Float32Array[];
  echoes: Echo[];
  direct: number;
  t20: number | null;
  t30: number | null;
  edt: number | null;
  bands: { fc: number; t30: number | null }[];
  peakDb: number;
  latencyMs: number;
  equalized: boolean;
}

const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function Plot({ data, title, height = 170, symmetric = true }: { data: number[] | Float32Array | null; title: string; height?: number; symmetric?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current && data) drawSeries(ref.current, [{ data, color: css('--accent') }], { symmetric });
  }, [data, symmetric]);
  return (
    <div className="plot" style={{ height }}>
      <span className="plot-title">{title}</span>
      {data ? <canvas ref={ref} /> : <div className="empty">No data yet</div>}
    </div>
  );
}

function Section({ id, title, children, lede }: { id: string; title: string; lede: string; children: React.ReactNode }) {
  return (
    <section className="figure" id={id} aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`} style={{ margin: '4px 0 6px', fontSize: 19 }}>
        {title}
      </h2>
      <p className="muted" style={{ marginTop: 0 }}>
        {lede}
      </p>
      {children}
    </section>
  );
}

export default function Lab(_props: { route?: Route }) {
  const notify = useApp((s) => s.notify);
  // ---------------- measurement ----------------
  const [f1, setF1] = useState(100);
  const [f2, setF2] = useState(16000);
  const [dur, setDur] = useState(1.5);
  const [out, setOut] = useState<'both' | 'left' | 'right'>('both');
  const [busy, setBusy] = useState<string | null>(null);
  const [meas, setMeas] = useState<Measurement | null>(null);
  const [truth, setTruth] = useState<number | ''>('');
  const [label, setLabel] = useState('');
  const [captures, setCaptures] = useState<Capture[]>(() => loadCaptures());
  const [cal, setCal] = useState<StoredCalibration | null>(() => loadCalibration());
  const [useCal, setUseCal] = useState(true);
  const [orient, setOrient] = useState<{ alpha: number | null; beta: number | null; gamma: number | null } | null>(null);

  const measure = async () => {
    setBusy('measure');
    try {
      const ctx = await openAudio();
      const spec = { f1, f2, seconds: dur, sampleRate: ctx.sampleRate };
      const sweep = essSweep(spec);
      const inv = essInverse(spec, sweep);
      const rec = await playAndRecord(sweep, { output: out, preRollS: 0.3, tailS: 1.0 });
      const len = Math.round(0.9 * rec.sampleRate);
      const irs = rec.channels.map((ch) => deconvolve(ch, inv, Math.round(0.3 * rec.sampleRate) + len).subarray(0));
      // Trim to start ~5 ms before the direct sound of channel 0.
      const h0 = irs[0];
      let direct = 0;
      for (let i = 0; i < h0.length; i++) if (Math.abs(h0[i]) > Math.abs(h0[direct])) direct = i;
      const start = Math.max(0, direct - Math.round(0.005 * rec.sampleRate));
      const latencyMs = roundTripLatencyMs(direct, Math.round(0.3 * rec.sampleRate), rec.sampleRate);
      let trimmed: Float32Array[] = irs.map((h) => h.slice(start, start + len));
      const equalized = !!(useCal && cal && cal.sampleRate === rec.sampleRate);
      if (equalized && cal) {
        const c = { response: b64ToF32(cal.response), preSamples: cal.preSamples };
        trimmed = trimmed.map((h) => equalize(h, c));
      }
      const { echoes, direct: d0 } = findEchoes(trimmed[0], rec.sampleRate, { thresholdDb: -24, minSeparationMs: 0.8 });
      const m = decayMetrics(trimmed[0], rec.sampleRate, d0);
      const bands = [250, 500, 1000, 2000, 4000].map((fc) => ({ fc, t30: decayMetrics(octaveBand(trimmed[0], rec.sampleRate, fc), rec.sampleRate, d0).t30 }));
      let peak = 0;
      for (const v of rec.channels[0]) peak = Math.max(peak, Math.abs(v));
      setMeas({ sampleRate: rec.sampleRate, irs: trimmed, echoes, direct: d0, t20: m.t20, t30: m.t30, edt: m.edt, bands, peakDb: 20 * Math.log10(peak + 1e-12), latencyMs, equalized });
      if (peak < 1e-3) notify('The microphone signal is very quiet: turn the volume up or move closer.', 'error');
    } catch (e) {
      notify(`Measurement failed: ${(e as Error).message}`, 'error');
    } finally {
      setBusy(null);
    }
  };

  const envDb = useMemo(() => {
    if (!meas) return null;
    const h = meas.irs[0];
    const pk = h.reduce((a, v) => Math.max(a, Math.abs(v)), 1e-12);
    const n = Math.min(h.length, Math.round(0.12 * meas.sampleRate));
    return Array.from(h.subarray(0, n), (v) => Math.max(-60, 20 * Math.log10(Math.abs(v) / pk + 1e-9)));
  }, [meas]);

  const addCapture = () => {
    if (!meas) return;
    const c: Capture = {
      id: `${Date.now().toString(36)}`,
      label: label || `capture ${captures.length + 1}`,
      createdAt: new Date().toISOString(),
      sampleRate: meas.sampleRate,
      irs: meas.irs.map(f32ToB64),
      measuredDistance: truth === '' ? undefined : Number(truth),
      room: room,
      position: pos,
      orientation: orient ?? undefined,
      estimates: { firstEchoDistance: meas.echoes[0]?.distance, t30: meas.t30, t20: meas.t20, latencyMs: meas.latencyMs },
      equalized: meas.equalized,
      device: navigator.userAgent,
    };
    const next = [...captures, c];
    setCaptures(next);
    saveCaptures(next);
    notify(`Saved “${c.label}” to the capture set`);
  };

  const calibrate = () => {
    if (!meas) return;
    if (meas.equalized) return notify('Measure once with equalisation off, then calibrate from that measurement.', 'error');
    const { response, preSamples } = directPathWindow(meas.irs[0], meas.direct, meas.sampleRate);
    const c: StoredCalibration = { sampleRate: meas.sampleRate, response: f32ToB64(response), preSamples, latencyMs: meas.latencyMs, createdAt: new Date().toISOString() };
    setCal(c);
    saveCalibration(c);
    notify('Saved the device calibration; later measurements are equalised with it.');
  };

  // ---------------- room twin ----------------
  const [room, setRoom] = useState({ lx: 5, ly: 4, lz: 2.7 });
  const [pos, setPos] = useState<[number, number, number]>([1.2, 2.0, 0.8]);
  const [alpha, setAlpha] = useState(0.15);
  const [twin, setTwin] = useState<TwinResult | null>(null);
  const [twinProg, setTwinProg] = useState<number | null>(null);
  const runTwin = async () => {
    setTwinProg(0);
    try {
      setTwin(await simulateShoebox({ ...room, alpha }, pos, { maxCells: 56, seconds: 0.6, onProgress: setTwinProg }));
    } finally {
      setTwinProg(null);
    }
  };
  const predicted = shoeboxFirstEchoes(room, pos);

  // ---------------- virtual headphones ----------------
  const [span, setSpan] = useState(0.3);
  const [head, setHead] = useState<{ x: number; y: number }>({ x: 0, y: 0.5 });
  const [beta, setBeta] = useState(0.005);
  const [tracking, setTracking] = useState(false);
  const [ctcOn, setCtcOn] = useState(true);
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const trackerRef = useRef<HeadTracker | null>(null);
  const playRef = useRef<{ stop: () => void; update: (f: CtcFilters, on: boolean) => void } | null>(null);
  const geo: CtcGeometry = useMemo(() => ({ speakerSpan: span, head }), [span, head]);
  const filters = useMemo(() => designCtc(geo, 48000, 1024, beta), [geo, beta]);
  const freqs = useMemo(() => Array.from({ length: 60 }, (_, k) => 150 * Math.pow(7000 / 150, k / 59)), []);
  const sep = useMemo(() => separationDb(filters, geo, geo, freqs), [filters, geo, freqs]);
  const nat = useMemo(() => stereoSeparationDb(geo, freqs), [geo, freqs]);

  useEffect(() => {
    playRef.current?.update(filters, ctcOn);
  }, [filters, ctcOn]);

  const startTracking = async () => {
    if (!videoRef.current) return;
    try {
      const t = new HeadTracker();
      t.onPose = (p: HeadPose | null) => {
        if (p) setHead({ x: Math.max(-0.4, Math.min(0.4, p.x)), y: Math.max(0.25, Math.min(1.5, p.y)) });
      };
      await t.start(videoRef.current);
      trackerRef.current = t;
      setTracking(true);
    } catch (e) {
      notify(`Head tracking unavailable: ${(e as Error).message}`, 'error');
    }
  };
  const stopTracking = () => {
    trackerRef.current?.stop();
    trackerRef.current = null;
    setTracking(false);
  };
  useEffect(() => () => trackerRef.current?.stop(), []);

  const togglePlay = async () => {
    if (playing) {
      playRef.current?.stop();
      playRef.current = null;
      setPlaying(false);
      return;
    }
    const ctx = await openAudio();
    // Test programme: left = low chirp bursts, right = high chirp bursts, alternating.
    const fs = ctx.sampleRate;
    const n = fs * 4;
    const prog = ctx.createBuffer(2, n, fs);
    const L = prog.getChannelData(0);
    const R = prog.getChannelData(1);
    for (let i = 0; i < n; i++) {
      const t = i / fs;
      const ph = t % 2;
      const env = (x: number) => (x > 0 && x < 0.8 ? Math.sin((Math.PI * x) / 0.8) ** 2 : 0);
      L[i] = 0.4 * env(ph) * Math.sin(2 * Math.PI * (400 + 300 * ph) * t);
      R[i] = 0.4 * env(ph - 1) * Math.sin(2 * Math.PI * (1200 + 600 * (ph - 1)) * t);
    }
    const src = ctx.createBufferSource();
    src.buffer = prog;
    src.loop = true;
    const split = ctx.createChannelSplitter(2);
    const merge = ctx.createChannelMerger(2);
    const convs = [0, 1].map(() => [0, 1].map(() => ctx.createConvolver()));
    const bypass = [0, 1].map(() => ctx.createGain());
    src.connect(split);
    for (let s = 0; s < 2; s++)
      for (let p = 0; p < 2; p++) {
        convs[s][p].normalize = false;
        split.connect(convs[s][p], p);
        convs[s][p].connect(merge, 0, s);
      }
    split.connect(bypass[0], 0);
    split.connect(bypass[1], 1);
    bypass[0].connect(merge, 0, 0);
    bypass[1].connect(merge, 0, 1);
    merge.connect(ctx.destination);
    const update = (f: CtcFilters, on: boolean) => {
      for (let s = 0; s < 2; s++)
        for (let p = 0; p < 2; p++) {
          const b = ctx.createBuffer(1, f.length, f.sampleRate);
          b.copyToChannel(Float32Array.from(f.taps[s][p], (v) => (on ? v : 0)), 0);
          convs[s][p].buffer = b;
        }
      bypass.forEach((g) => (g.gain.value = on ? 0 : 1));
    };
    update(filters, ctcOn);
    src.start();
    playRef.current = {
      stop: () => {
        src.stop();
        merge.disconnect();
      },
      update,
    };
    setPlaying(true);
  };

  // ---------------- phone orientation ----------------
  const enableOrientation = async () => {
    const DOE = window.DeviceOrientationEvent as unknown as { requestPermission?: () => Promise<string> } | undefined;
    try {
      if (DOE?.requestPermission) {
        const r = await DOE.requestPermission();
        if (r !== 'granted') throw new Error('permission denied');
      }
      const on = (e: DeviceOrientationEvent) => setOrient({ alpha: e.alpha, beta: e.beta, gamma: e.gamma });
      window.addEventListener('deviceorientation', on);
      notify('Orientation sensor on: move the phone and watch the readout.');
    } catch (e) {
      notify(`Orientation unavailable: ${(e as Error).message}`, 'error');
    }
  };

  const importCaptures = async (f: File) => {
    try {
      const j = JSON.parse(await f.text());
      if (j.format !== 'acoustic-sandbox-captures') throw new Error('not a capture file');
      const next = [...captures, ...(j.captures as Capture[])];
      setCaptures(next);
      saveCaptures(next);
    } catch (e) {
      notify((e as Error).message, 'error');
    }
  };

  const eyrT = eyring({ ...room, alpha });
  const sabT = sabine({ ...room, alpha });

  return (
    <div className="content" data-testid="lab">
      <h1>Lab: measure a real room</h1>
      <p className="lede">
        Use your laptop's own speakers and microphones to measure the room you are in. Everything runs in your browser: audio and
        video are processed on this device and never uploaded. Plug-in headphones must be <b>off</b> for room measurements.
      </p>

      <Section id="measure" title="1 · Impulse response, echoes and reverberation" lede="Plays a logarithmic sweep and records it. Deconvolution turns the recording into the room's impulse response; its peaks are echoes, and its energy decay gives the reverberation time.">
        <div className="row wrap" style={{ alignItems: 'flex-end' }}>
          <NumberField label="Start (Hz)" value={f1} min={20} max={2000} onChange={setF1} />
          <NumberField label="End (Hz)" value={f2} min={2000} max={22000} onChange={setF2} />
          <NumberField label="Sweep (s)" value={dur} min={0.3} max={6} onChange={setDur} />
          <div className="field">
            <label>Speaker</label>
            <select className="input" value={out} onChange={(e) => setOut(e.target.value as typeof out)} aria-label="Output speaker">
              <option value="both">Both</option>
              <option value="left">Left only</option>
              <option value="right">Right only</option>
            </select>
          </div>
          <div className="field">
            <button className="btn primary" onClick={measure} disabled={!!busy} data-testid="measure">
              <Mic size={16} /> {busy === 'measure' ? 'Measuring…' : 'Measure'}
            </button>
          </div>
          <label className="row tight" title={cal ? `calibrated ${cal.createdAt.slice(0, 10)}, latency ${cal.latencyMs.toFixed(1)} ms` : 'no device calibration yet'}>
            <input type="checkbox" checked={useCal && !!cal} disabled={!cal} onChange={(e) => setUseCal(e.target.checked)} aria-label="Equalise with device calibration" /> Equalise with
            device calibration
          </label>
          {cal && (
            <button className="btn ghost" onClick={() => (setCal(null), saveCalibration(null))} aria-label="Forget device calibration">
              Forget calibration
            </button>
          )}
        </div>
        {meas && (
          <>
            <div className="metric-row" data-testid="measure-results">
              <div className="metric">
                <div className="v">{meas.echoes[0] ? `${meas.echoes[0].distance.toFixed(2)} m` : '—'}</div>
                <div className="k">nearest reflector (first echo / 2)</div>
              </div>
              <div className="metric">
                <div className="v">{meas.t30 ? `${meas.t30.toFixed(2)} s` : '—'}</div>
                <div className="k">T30 (broadband)</div>
              </div>
              <div className="metric">
                <div className="v">{meas.t20 ? `${meas.t20.toFixed(2)} s` : '—'}</div>
                <div className="k">T20</div>
              </div>
              <div className="metric">
                <div className="v">{meas.edt ? `${meas.edt.toFixed(2)} s` : '—'}</div>
                <div className="k">EDT</div>
              </div>
              <div className="metric">
                <div className="v">{meas.peakDb.toFixed(0)} dBFS</div>
                <div className="k">recording peak</div>
              </div>
              <div className="metric">
                <div className="v">{meas.latencyMs.toFixed(1)} ms</div>
                <div className="k">round-trip latency{meas.equalized ? ' · equalised' : ''}</div>
              </div>
            </div>
            <Plot data={envDb} title="impulse response envelope (dB), first 120 ms" symmetric={false} />
            <table className="data">
              <thead>
                <tr>
                  <th>echo</th>
                  <th>delay (ms)</th>
                  <th>extra path (m)</th>
                  <th>reflector distance (m)</th>
                  <th>level (dB)</th>
                </tr>
              </thead>
              <tbody>
                {meas.echoes.map((e, k) => (
                  <tr key={k}>
                    <td>{k + 1}</td>
                    <td>{e.delayMs.toFixed(2)}</td>
                    <td>{e.extraPath.toFixed(2)}</td>
                    <td>{e.distance.toFixed(2)}</td>
                    <td>{e.levelDb.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <table className="data">
              <thead>
                <tr>
                  <th>octave band</th>
                  {meas.bands.map((b) => (
                    <th key={b.fc}>{b.fc >= 1000 ? `${b.fc / 1000} k` : b.fc}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>T30 (s)</td>
                  {meas.bands.map((b) => (
                    <td key={b.fc}>{b.t30 ? b.t30.toFixed(2) : '—'}</td>
                  ))}
                </tr>
              </tbody>
            </table>
            <div className="row wrap" style={{ alignItems: 'flex-end' }}>
              <div className="field">
                <label>Label</label>
                <input className="input" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="kitchen, laptop on table" aria-label="Capture label" />
              </div>
              <div className="field">
                <label>Tape-measured nearest wall (m)</label>
                <input className="input mono" value={truth} onChange={(e) => setTruth(e.target.value === '' ? '' : Number(e.target.value))} aria-label="Measured distance" />
              </div>
              <div className="field">
                <button className="btn" onClick={addCapture} data-testid="save-capture">
                  Save to capture set
                </button>
              </div>
              <div className="field">
                <button className="btn" onClick={calibrate} data-testid="calibrate" title="Use the direct speaker-to-mic path of this measurement as the device response">
                  Calibrate device from this
                </button>
              </div>
            </div>
          </>
        )}
      </Section>

      <Section id="twin" title="2 · Room twin" lede="Describe the room, then compare the measurement with diffuse-field theory (Sabine, Eyring) and with a 3D FDTD simulation of the same box. The first-order echo distances predicted from your position should appear in the measured echo list.">
        <div className="row wrap">
          <NumberField label="Length (m)" value={room.lx} min={1} max={40} onChange={(v) => setRoom({ ...room, lx: v })} />
          <NumberField label="Width (m)" value={room.ly} min={1} max={40} onChange={(v) => setRoom({ ...room, ly: v })} />
          <NumberField label="Height (m)" value={room.lz} min={1} max={20} onChange={(v) => setRoom({ ...room, lz: v })} />
          <NumberField label="Absorption α" value={alpha} min={0.01} max={0.99} onChange={setAlpha} />
        </div>
        <div className="row wrap">
          {(['x', 'y', 'z'] as const).map((a, k) => (
            <NumberField key={a} label={`Laptop ${a} (m)`} value={pos[k]} min={0.05} max={40} onChange={(v) => setPos(pos.map((x, q) => (q === k ? v : x)) as [number, number, number])} />
          ))}
        </div>
        <div className="metric-row">
          <div className="metric">
            <div className="v">{sabT.toFixed(2)} s</div>
            <div className="k">Sabine T60</div>
          </div>
          <div className="metric">
            <div className="v">{eyrT.toFixed(2)} s</div>
            <div className="k">Eyring T60</div>
          </div>
          <div className="metric">
            <div className="v">{meas?.t30 ? alphaFromT60(room, meas.t30).toFixed(2) : '—'}</div>
            <div className="k">α implied by the measured T30</div>
          </div>
          <div className="metric">
            <div className="v">{twin?.t30 ? `${twin.t30.toFixed(2)} s` : twinProg !== null ? `${Math.round(twinProg * 100)} %` : '—'}</div>
            <div className="k">FDTD twin T30 {twin ? `(${twin.cells.join('×')} cells)` : ''}</div>
          </div>
        </div>
        <p className="muted" style={{ fontSize: 13 }}>
          Predicted first-order reflector distances: {predicted.map((d) => d.toFixed(2)).join(', ')} m.
        </p>
        <button className="btn" onClick={runTwin} disabled={twinProg !== null} data-testid="run-twin">
          {twinProg !== null ? 'Simulating…' : 'Simulate the room (3D FDTD)'}
        </button>
      </Section>

      <Section id="headphones" title="3 · Virtual headphones" lede="Crosstalk cancellation drives both laptop speakers so that each ear hears only its own channel. It works in a small sweet spot, so the webcam tracks your head and the filters follow you. Left channel: low chirps. Right channel: high chirps.">
        <div className="row wrap" style={{ alignItems: 'flex-end' }}>
          <NumberField label="Speaker span (m)" value={span} min={0.1} max={1} onChange={setSpan} />
          <NumberField label="Regularisation β" value={beta} min={0.0001} max={1} onChange={setBeta} />
          <div className="field">
            <label>
              <span>Head x (m)</span>
              <span className="mono">{head.x.toFixed(2)}</span>
            </label>
            <input type="range" min={-0.4} max={0.4} step={0.01} value={head.x} onChange={(e) => setHead({ ...head, x: Number(e.target.value) })} aria-label="Head x" />
          </div>
          <div className="field">
            <label>
              <span>Head distance (m)</span>
              <span className="mono">{head.y.toFixed(2)}</span>
            </label>
            <input type="range" min={0.25} max={1.5} step={0.01} value={head.y} onChange={(e) => setHead({ ...head, y: Number(e.target.value) })} aria-label="Head distance" />
          </div>
        </div>
        <div className="row wrap">
          <button className="btn" onClick={tracking ? stopTracking : startTracking} data-testid="track">
            <Camera size={16} /> {tracking ? 'Stop head tracking' : 'Track my head (webcam)'}
          </button>
          <button className="btn primary" onClick={togglePlay} data-testid="ctc-play">
            {playing ? <Square size={16} /> : <Play size={16} />} {playing ? 'Stop' : 'Play test'}
          </button>
          <label className="row tight">
            <input type="checkbox" checked={ctcOn} onChange={(e) => setCtcOn(e.target.checked)} aria-label="Crosstalk cancellation" /> Crosstalk cancellation
          </label>
        </div>
        <video ref={videoRef} playsInline muted style={{ width: 200, borderRadius: 8, marginTop: 10, display: tracking ? 'block' : 'none' }} />
        <Plot data={sep} title={`channel separation at the ears (dB), 150 Hz → 7 kHz; plain stereo ≈ ${(nat.reduce((a, b) => a + b, 0) / nat.length).toFixed(1)} dB`} symmetric={false} />
      </Section>

      <Section id="captures" title="4 · Capture set" lede="Build a small labelled dataset of real rooms. Export it and run scripts/eval_real_captures.py to score the echo-distance and T60 estimators against your tape measurements.">
        <table className="data">
          <thead>
            <tr>
              <th>label</th>
              <th>first echo (m)</th>
              <th>tape (m)</th>
              <th>T30 (s)</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {captures.map((c) => (
              <tr key={c.id}>
                <td>{c.label}</td>
                <td>{c.estimates.firstEchoDistance?.toFixed(2) ?? '—'}</td>
                <td>{c.measuredDistance?.toFixed(2) ?? '—'}</td>
                <td>{c.estimates.t30?.toFixed(2) ?? '—'}</td>
                <td>
                  <button
                    className="btn icon sm ghost danger"
                    aria-label="Delete capture"
                    onClick={() => {
                      const next = captures.filter((x) => x.id !== c.id);
                      setCaptures(next);
                      saveCaptures(next);
                    }}
                  >
                    <Trash2 size={14} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="row tight">
          <button className="btn" disabled={!captures.length} onClick={() => download(exportCaptures(captures), 'captures.json')}>
            <Download size={16} /> Export JSON
          </button>
          <label className="btn">
            <Upload size={16} /> Import
            <input type="file" accept=".json" hidden onChange={(e) => e.target.files?.[0] && importCaptures(e.target.files[0])} />
          </label>
        </div>
      </Section>

      <Section id="phone" title="5 · Phone as a moving sensor" lede="On a phone, the orientation sensor gives the device's pose while you walk it around the room; each saved capture is tagged with it.">
        <button className="btn" onClick={enableOrientation}>
          <Smartphone size={16} /> Use the orientation sensor
        </button>
        {orient && (
          <p className="mono">
            heading α = {orient.alpha?.toFixed(1)}° · tilt β = {orient.beta?.toFixed(1)}° · roll γ = {orient.gamma?.toFixed(1)}°
          </p>
        )}
      </Section>
    </div>
  );
}
