import { useEffect, useRef } from 'react';
import { PRESETS, type Preset } from '../engine/presets';
import { buildSimulation } from '../engine/scene';
import { navigate } from '../lib/router';
import { FieldRenderer } from '../render/fieldRenderer';
import { useApp } from '../state/store';

/** Render a thumbnail by running the preset for a few hundred steps. */
function Thumb({ preset }: { preset: Preset }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useApp((s) => s.theme);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    let cancelled = false;
    const io = new IntersectionObserver((entries) => {
      if (!entries.some((e) => e.isIntersecting)) return;
      io.disconnect();
      // Defer so the gallery paints first.
      setTimeout(() => {
        if (cancelled) return;
        const sim = buildSimulation(preset.build());
        const steps = 260;
        for (let s = 0; s < steps; s++) sim.step();
        canvas.width = sim.ny;
        canvas.height = sim.nx;
        let peak = 1e-9;
        for (let i = 0; i < sim.n; i++) peak = Math.max(peak, Math.abs(sim.p[i]));
        new FieldRenderer(canvas).draw(sim.p, sim.nx, sim.ny, sim.material, {
          colormap: theme === 'dark' ? 'icefire' : 'balance',
          mode: 'linear',
          scale: peak * 0.6,
          dbRange: 50,
          signed: true,
          showMaterials: true,
          theme,
        });
      }, 30);
    });
    io.observe(canvas);
    return () => {
      cancelled = true;
      io.disconnect();
    };
  }, [preset, theme]);
  return <canvas ref={ref} aria-hidden />;
}

export function Gallery() {
  return (
    <div className="content">
      <h1>Gallery</h1>
      <p className="lede">
        Ready-made experiments. Each one opens in the sandbox, where you can run it, change it, listen at the microphones, and share
        your version.
      </p>
      <div className="grid-cards">
        {PRESETS.map((p) => (
          <button key={p.id} className="preset-card" onClick={() => navigate(`/sandbox?preset=${p.id}`)} data-testid={`preset-${p.id}`}>
            <Thumb preset={p} />
            <div className="body">
              <strong>{p.title}</strong>
              <span>{p.blurb}</span>
              <p className="dim" style={{ fontSize: 12.5, margin: '8px 0 0' }}>
                {p.physics}
              </p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
