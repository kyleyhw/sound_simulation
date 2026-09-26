import { useEffect, useRef, useState } from 'react';
import { PRESETS, type Preset } from '../engine/presets';
import { requestThumb } from '../gallery/thumbs';
import type { ThumbReply } from '../gallery/thumbWorker';
import { navigate } from '../lib/router';
import { paintField } from '../render/paint2d';
import { useApp } from '../state/store';

/**
 * Thumbnail: the preset run for a few hundred steps. The simulation runs in
 * a worker once per session, only when the card scrolls into view; a theme
 * switch just repaints the cached field.
 */
function Thumb({ preset, order }: { preset: Preset; order: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useApp((s) => s.theme);
  const [thumb, setThumb] = useState<ThumbReply | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    let cancel = () => {};
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        io.disconnect();
        cancel = requestThumb(preset.id, order, setThumb);
      },
      { rootMargin: '200px' },
    );
    io.observe(canvas);
    return () => {
      io.disconnect();
      cancel();
    };
  }, [preset, order]);

  useEffect(() => {
    if (thumb && ref.current) paintField(ref.current, thumb.field, thumb.material, thumb.rows, thumb.cols, thumb.peak * 0.6, theme);
  }, [thumb, theme]);

  return <canvas ref={ref} aria-hidden data-ready={thumb ? '1' : undefined} />;
}

export function Gallery() {
  return (
    <div className="content">
      <h1>Gallery</h1>
      <p className="lede">Ready-made experiments. Each one opens in the sandbox.</p>
      <div className="grid-cards">
        {PRESETS.map((p, k) => (
          <button key={p.id} className="preset-card" onClick={() => navigate(`/sandbox?preset=${p.id}`)} data-testid={`preset-${p.id}`}>
            <Thumb preset={p} order={k} />
            <div className="body">
              <strong>{p.title}</strong>
              <span>{p.blurb}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
