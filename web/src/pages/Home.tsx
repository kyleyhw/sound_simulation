import { ArrowRight, BookOpen, Brush, Ear } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { SECONDARY } from '../components/TopNav';
import { DEFAULT_PARAMS, Simulation } from '../engine/simulation';
import { paintField } from '../render/paint2d';
import { useApp } from '../state/store';

const ENTRIES = [
  { path: '/echo', title: 'Echo vision', text: 'Watch a neural network rebuild a room from its echoes.', icon: <Ear size={22} />, id: 'echo' },
  { path: '/sandbox', title: 'Sandbox', text: 'Draw walls and sources, watch sound move.', icon: <Brush size={22} />, id: 'sandbox' },
  { path: '/learn', title: 'Learn', text: 'How the simulation works, one live idea at a time.', icon: <BookOpen size={22} />, id: 'learn' },
];

const MORE = [{ path: '/gallery', label: 'Gallery', blurb: 'Ready-made experiments' }, ...SECONDARY];

const ROWS = 100;
const COLS = 190;
const PERIOD = 430; // steps between pings

/** The hero scene: a pulse in an anechoic space echoing off a wall, a pillar and a box. */
function heroSim(): Simulation {
  const sim = new Simulation({ ...DEFAULT_PARAMS, shape: [ROWS, COLS], outer: 'cpml', cpmlCells: 10 });
  const m = new Uint8Array(ROWS * COLS);
  const set = (i: number, j: number) => {
    if (i >= 0 && i < ROWS && j >= 0 && j < COLS) m[i * COLS + j] = 2;
  };
  for (let i = 18; i < 82; i++) for (let j = 150; j < 154; j++) set(i, j); // wall
  for (let i = 0; i < ROWS; i++)
    for (let j = 0; j < COLS; j++) if ((i - 30) ** 2 + (j - 105) ** 2 <= 81) set(i, j); // pillar
  for (let i = 64; i < 80; i++) for (let j = 92; j < 118; j++) set(i, j); // box
  sim.setMaterialMap(m);
  sim.setDrivers([{ id: 'ping', pos: [50, 42], waveform: { type: 'ricker', amplitude: 5, frequency: 0.09, delay: 16 }, enabled: true }]);
  return sim;
}

/** A small live simulation: it runs only while on screen and holds still for reduced motion. */
function HeroSim() {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useApp((s) => s.theme);
  const themeRef = useRef(theme);
  themeRef.current = theme;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const sim = heroSim();
    let peak = 1e-9;
    let steps = 0;
    const paint = () => {
      let m = 0;
      for (let i = 0; i < sim.n; i++) m = Math.max(m, Math.abs(sim.p[i]));
      peak = Math.max(m, peak * 0.985);
      paintField(cv, sim.p, sim.material, ROWS, COLS, peak * 0.7, themeRef.current);
    };
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      for (let s = 0; s < 190; s++) sim.step();
      paint();
      return;
    }
    let visible = true;
    let raf = 0;
    const tick = () => {
      raf = 0;
      if (!visible || document.hidden) return;
      for (let s = 0; s < 2; s++) {
        sim.step();
        if (++steps % PERIOD === 0) {
          sim.reset();
          peak = 1e-9;
        }
      }
      paint();
      raf = requestAnimationFrame(tick);
    };
    const start = () => {
      if (!raf && visible && !document.hidden) raf = requestAnimationFrame(tick);
    };
    const io = new IntersectionObserver((e) => {
      visible = e.some((x) => x.isIntersecting);
      start();
    });
    io.observe(cv);
    document.addEventListener('visibilitychange', start);
    paint();
    start();
    return () => {
      io.disconnect();
      document.removeEventListener('visibilitychange', start);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <a className="home-hero-sim" href="#/sandbox" aria-label="Open the sandbox">
      <canvas ref={ref} role="img" aria-label="A sound pulse echoing off a wall, a pillar and a box" data-testid="hero-sim" />
    </a>
  );
}

/** The front door (#/): one line, a live picture, three ways in. */
export function Home() {
  return (
    <div className="content home" data-testid="home">
      <section className="home-hero">
        <h1>
          Watch sound move, and watch a network <span className="accent">hear the shape of a room</span>.
        </h1>
        <HeroSim />
      </section>
      <div className="entry-grid">
        {ENTRIES.map((e) => (
          <a key={e.path} className="entry-card" href={`#${e.path}`} data-testid={`entry-${e.id}`}>
            <span className="entry-icon" aria-hidden>
              {e.icon}
            </span>
            <strong>{e.title}</strong>
            <span className="muted">{e.text}</span>
            <span className="entry-go" aria-hidden>
              Open <ArrowRight size={14} />
            </span>
          </a>
        ))}
      </div>
      <nav className="home-more" aria-label="More sections">
        {MORE.map((m) => (
          <a key={m.path} href={`#${m.path}`}>
            <strong>{m.label}</strong> <span className="dim">{m.blurb}</span>
          </a>
        ))}
      </nav>
    </div>
  );
}
