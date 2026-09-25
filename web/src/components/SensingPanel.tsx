import { useEffect, useRef, useState } from 'react';
import { colormapLut } from '../render/colormaps';
import { SkipModel } from '../sensing/model';
import { iou, maskFromScene, randomRoom, SensingSession } from '../sensing/session';
import { useApp } from '../state/store';

let modelPromise: Promise<SkipModel> | null = null;
const loadModel = () => (modelPromise ??= SkipModel.load(`${import.meta.env.BASE_URL}models/skip_v2`));

type MapKind = 'truth' | 'prior' | 'prob' | 'uncert';

function MapView({ kind, n, data, mask, poses, next, label }: { kind: MapKind; n: number; data: Float32Array | Uint8Array | null; mask: Uint8Array; poses: { driver: number[]; mics: number[][] }[]; next?: number[] | null; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const S = 3;
    cv.width = n * S;
    cv.height = n * S;
    const ctx = cv.getContext('2d')!;
    const img = ctx.createImageData(n, n);
    const lut = colormapLut(kind === 'uncert' ? 'magma' : 'viridis');
    for (let q = 0; q < n * n; q++) {
      let r = 20;
      let g = 24;
      let b = 32;
      if (kind === 'truth') {
        if (mask[q]) [r, g, b] = [210, 212, 220];
      } else if (data) {
        const v = Math.max(0, Math.min(1, data[q]));
        const c = Math.round(v * 255) * 4;
        [r, g, b] = [lut[c], lut[c + 1], lut[c + 2]];
      }
      img.data.set([r, g, b, 255], q * 4);
    }
    const tmp = document.createElement('canvas');
    tmp.width = n;
    tmp.height = n;
    tmp.getContext('2d')!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, n * S, n * S);
    if (kind !== 'truth') {
      // Truth outline for reference.
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          if (!mask[i * n + j]) continue;
          const edge = !mask[(i - 1) * n + j] || !mask[(i + 1) * n + j] || !mask[i * n + j - 1] || !mask[i * n + j + 1];
          if (edge) ctx.fillRect(j * S + 1, i * S + 1, 1, 1);
        }
    }
    for (const p of poses) {
      ctx.fillStyle = '#fb7185';
      ctx.fillRect(p.driver[1] * S - 2, p.driver[0] * S - 2, 5, 5);
      ctx.fillStyle = '#fbbf24';
      for (const m of p.mics) ctx.fillRect(m[1] * S - 1, m[0] * S - 1, 3, 3);
    }
    if (next) {
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(next[1] * S, next[0] * S, 12 * S, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }, [kind, n, data, mask, poses, next]);
  return (
    <figure style={{ margin: 0 }}>
      <canvas ref={ref} style={{ width: '100%', aspectRatio: '1', borderRadius: 6, imageRendering: 'pixelated' }} aria-label={label} />
      <figcaption className="muted" style={{ fontSize: 12 }}>
        {label}
      </figcaption>
    </figure>
  );
}

/**
 * Acoustic room sensing (plan 4.3.5, 6.7.1): the trained skip_v2 model runs in
 * the browser on poses simulated with the browser engine (v2 protocol, 64 x 64).
 * Scores are shown against the no-audio baseline (the training prior map).
 */
export function SensingPanel() {
  const scene = useApp((s) => s.scene);
  const notify = useApp((s) => s.notify);
  const [model, setModel] = useState<SkipModel | null>(null);
  const [loading, setLoading] = useState(false);
  const [session, setSession] = useState<SensingSession | null>(null);
  const [version, setVersion] = useState(0);
  const [busy, setBusy] = useState(false);
  const [k, setK] = useState(4);

  useEffect(() => {
    setLoading(true);
    loadModel()
      .then(setModel)
      .catch((e) => notify(`Could not load the sensing model: ${(e as Error).message}`, 'error'))
      .finally(() => setLoading(false));
  }, [notify]);

  const start = (mask: Uint8Array) => {
    if (!model) return;
    setSession(new SensingSession(model, mask));
    setVersion((v) => v + 1);
  };

  const run = async (poses: number, at?: number[]) => {
    if (!session) return;
    setBusy(true);
    try {
      for (let q = 0; q < poses; q++) {
        session.addPose(at);
        setVersion((v) => v + 1);
        await new Promise((r) => setTimeout(r, 0));
      }
    } catch (e) {
      notify(`Sensing failed: ${(e as Error).message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  if (!model) {
    return (
      <div>
        <h3>Acoustic room sensing</h3>
        <p className="hint">{loading ? 'Loading the sensing model (3 MB)…' : 'The sensing model is unavailable.'}</p>
      </div>
    );
  }
  const man = model.manifest;
  const n = man.protocol.grid;
  const prior = Float32Array.from(man.prior_map);
  const fused = session?.fused() ?? null;
  const next = session && fused ? session.suggestNext(fused) : null;
  const unc = fused ? fused.map((p) => 1 - Math.abs(2 * p - 1)) : null;
  const tau = man.calibration.threshold;
  const modelIou = session && fused ? iou(Array.from(fused, (p) => p > tau), session.mask) : null;
  const priorIou = session ? iou(Array.from(prior, (p) => p > man.prior_threshold), session.mask) : null;
  void version;

  return (
    <div data-testid="sensing-panel">
      <h3>Acoustic room sensing</h3>
      <p className="hint">
        A laptop-like device (one speaker, two mics 12 cells apart) plays a chirp from a few spots. The trained CNN turns each recording into an obstacle map, and the maps are
        fused. Everything runs in this browser: the physics is the same engine, and the network is the Python model's weights (epoch {man.epoch ?? '?'}).
      </p>
      <div className="btn-row">
        <button className="btn sm" onClick={() => start(maskFromScene(scene.materials, scene.params.shape[0], scene.params.shape[1], n))} disabled={busy || scene.params.dims !== 2} data-testid="sense-scene">
          Use this scene (as 64×64)
        </button>
        <button className="btn sm" onClick={() => start(randomRoom(n))} disabled={busy} data-testid="sense-random">
          Random room
        </button>
      </div>
      {session && (
        <>
          <div className="btn-row" style={{ alignItems: 'center' }}>
            <label className="row tight" style={{ gap: 6, fontSize: 13 }}>
              Poses
              <select className="input" value={k} onChange={(e) => setK(Number(e.target.value))} aria-label="Poses to add" style={{ width: 60 }}>
                {[1, 2, 4, 8].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <button className="btn sm primary" onClick={() => void run(k)} disabled={busy} data-testid="sense-run">
              {busy ? 'Sensing…' : `Add ${k} random pose${k > 1 ? 's' : ''}`}
            </button>
            {next && (
              <button className="btn sm" onClick={() => void run(1, next)} disabled={busy} data-testid="sense-next" title="Place the next pose where the map is least certain">
                Add a pose at the hint
              </button>
            )}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <MapView kind="truth" n={n} data={null} mask={session.mask} poses={session.poses} label="Room (truth) and poses" />
            <MapView kind="prior" n={n} data={prior} mask={session.mask} poses={[]} label="No-audio baseline (training prior)" />
            <MapView kind="prob" n={n} data={fused} mask={session.mask} poses={session.poses} label={`Fused obstacle probability (K = ${session.poses.length})`} />
            <MapView kind="uncert" n={n} data={unc} mask={session.mask} poses={[]} next={next} label="Uncertainty; blue ring = move here next" />
          </div>
          <div className="metric-row" data-testid="sense-scores">
            <div className="metric">
              <div className="v">{modelIou === null ? '—' : modelIou.toFixed(3)}</div>
              <div className="k">IoU, audio model (τ = {tau.toFixed(2)})</div>
            </div>
            <div className="metric">
              <div className="v">{priorIou === null ? '—' : priorIou.toFixed(3)}</div>
              <div className="k">IoU, no-audio baseline</div>
            </div>
          </div>
          <p className="hint dim" style={{ fontSize: 12 }}>
            This is the Phase 2 spectrogram CNN, retrained without data leaks. On 500 held-out rooms it does not beat the no-audio baseline (IoU 0.104 against 0.101 at 4 poses,
            not significant). The aligned models on the Research page do. The hint is a heuristic: the free spot whose neighbourhood has the most uncertainty.
          </p>
        </>
      )}
    </div>
  );
}
