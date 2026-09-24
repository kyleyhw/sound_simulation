import { ExternalLink, Pause, Play, RotateCcw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { buildSimulation, encodeSceneUrl, type Scene } from '../engine/scene';
import type { Simulation } from '../engine/simulation';
import { colormapLut } from '../render/colormaps';

export interface Variant {
  label: string;
  build: () => Scene;
  /** Open this preset in the sandbox (else the scene is shared by URL). */
  presetId?: string;
}

/**
 * A small embedded simulation for the explainers (plan 10.6). It runs only
 * while playing and on screen, auto-scales with a slowly decaying peak, and
 * can hand its scene to the full sandbox.
 */
export function LiveSim({ variants, stepsPerFrame = 3, caption, height = 300 }: { variants: Variant[]; stepsPerFrame?: number; caption?: string; height?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const simRef = useRef<Simulation | null>(null);
  const peakRef = useRef(1e-9);
  const maxRef = useRef(0);
  const [variant, setVariant] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [visible, setVisible] = useState(false);
  const [step, setStep] = useState(0);

  const draw = () => {
    const sim = simRef.current;
    const cv = canvasRef.current;
    if (!sim || !cv) return;
    const rows = sim.nx;
    const cols = sim.ny;
    if (cv.width !== cols || cv.height !== rows) {
      cv.width = cols;
      cv.height = rows;
    }
    const ctx = cv.getContext('2d')!;
    const img = ctx.createImageData(cols, rows);
    let m = 0;
    for (let i = 0; i < sim.n; i++) m = Math.max(m, Math.abs(sim.p[i]));
    peakRef.current = Math.max(m, peakRef.current * 0.97);
    maxRef.current = Math.max(maxRef.current, m);
    const lut = colormapLut('icefire');
    const pk = peakRef.current;
    for (let q = 0; q < rows * cols; q++) {
      const o = q * 4;
      if (sim.material[q] !== 0) {
        img.data[o] = 150;
        img.data[o + 1] = 153;
        img.data[o + 2] = 162;
      } else {
        const c = Math.round((0.5 + 0.5 * Math.max(-1, Math.min(1, sim.p[q] / pk))) * 255) * 4;
        img.data[o] = lut[c];
        img.data[o + 1] = lut[c + 1];
        img.data[o + 2] = lut[c + 2];
      }
      img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const dot = (pos: number[], color: string) => {
      ctx.fillStyle = color;
      ctx.fillRect(pos[1] - 1, pos[0] - 1, 3, 3);
    };
    sim.drivers.forEach((d) => dot(d.pos, '#fb7185'));
    sim.probes.forEach((p) => dot(p.pos, '#fbbf24'));
  };

  const rebuild = (k: number) => {
    simRef.current = buildSimulation(variants[k].build());
    peakRef.current = 1e-9;
    maxRef.current = 0;
    setStep(0);
    draw();
  };

  useEffect(() => {
    rebuild(variant);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [variant]);

  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    const io = new IntersectionObserver((e) => setVisible(e[0].isIntersecting), { threshold: 0.1 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!playing || !visible) return;
    let raf = 0;
    const tick = () => {
      const sim = simRef.current;
      if (sim) {
        for (let k = 0; k < stepsPerFrame; k++) sim.step();
        setStep(sim.step_count);
        draw();
        // A pulse that has left (or died away in) the domain replays by itself.
        if (sim.step_count > 200 && peakRef.current < 2e-3 * maxRef.current) rebuild(variant);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, visible, stepsPerFrame]);

  const openInSandbox = async () => {
    const v = variants[variant];
    if (v.presetId) window.location.hash = `/sandbox?preset=${v.presetId}`;
    else window.location.hash = `/sandbox?s=${await encodeSceneUrl(v.build())}`;
  };

  return (
    <figure className="livesim" ref={hostRef} data-testid="livesim">
      <div className="livesim-stage" style={{ height }}>
        <canvas ref={canvasRef} aria-label={`live simulation: ${variants[variant].label}`} />
      </div>
      <div className="btn-row" style={{ marginTop: 8, alignItems: 'center' }}>
        <button className="btn sm primary" onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause' : 'Play'} data-testid="livesim-play">
          {playing ? <Pause size={14} /> : <Play size={14} />} {playing ? 'Pause' : 'Play'}
        </button>
        <button className="btn sm" onClick={() => rebuild(variant)} aria-label="Restart">
          <RotateCcw size={14} />
        </button>
        {variants.length > 1 &&
          variants.map((v, k) => (
            <button key={v.label} className="btn sm" aria-pressed={k === variant} onClick={() => setVariant(k)} style={k === variant ? { borderColor: 'var(--accent)', color: 'var(--accent)' } : undefined}>
              {v.label}
            </button>
          ))}
        <button className="btn sm ghost" onClick={() => void openInSandbox()}>
          <ExternalLink size={14} /> Open in sandbox
        </button>
        <span className="muted mono" style={{ fontSize: 12 }}>
          step {step}
        </span>
      </div>
      {caption && <figcaption className="muted" style={{ fontSize: 13 }}>{caption}</figcaption>}
    </figure>
  );
}
