import { useEffect, useState } from 'react';
import { defaultWaveform, nominalFrequency, WAVEFORM_LABELS, type WaveformSpec, type WaveformType } from '../engine/waveforms';

/** Number input that validates on blur/enter and only commits valid values. */
export function NumberField(props: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  integer?: boolean;
  hint?: string;
  testId?: string;
}) {
  const { label, value, onChange, min = -Infinity, max = Infinity, step, integer, hint, testId } = props;
  const [text, setText] = useState(String(value));
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const v = Number(text);
    if (text.trim() === '' || !Number.isFinite(v)) return setErr('Enter a number');
    if (integer && !Number.isInteger(v)) return setErr('Enter a whole number');
    if (v < min || v > max) return setErr(`Between ${min} and ${max}`);
    setErr(null);
    if (v !== value) onChange(v);
  };
  return (
    <div className="field">
      <label>
        <span>{label}</span>
      </label>
      <input
        className={`input mono${err ? ' invalid' : ''}`}
        inputMode="decimal"
        value={text}
        step={step}
        aria-label={label}
        aria-invalid={!!err}
        data-testid={testId}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit();
        }}
      />
      {err ? <span className="warn">{err}</span> : hint ? <span className="hint">{hint}</span> : null}
    </div>
  );
}

const PARAMS: Record<WaveformType, { key: string; label: string; min: number; max: number }[]> = {
  ricker: [
    { key: 'amplitude', label: 'Amplitude', min: 0, max: 1e6 },
    { key: 'frequency', label: 'Frequency', min: 1e-6, max: 1e6 },
    { key: 'delay', label: 'Delay', min: 0, max: 1e9 },
  ],
  gaussian: [
    { key: 'amplitude', label: 'Amplitude', min: 0, max: 1e6 },
    { key: 'center_time', label: 'Centre time', min: 0, max: 1e9 },
    { key: 'width', label: 'Width (σ)', min: 1e-9, max: 1e9 },
  ],
  cosine: [
    { key: 'amplitude', label: 'Amplitude', min: 0, max: 1e6 },
    { key: 'frequency', label: 'Frequency', min: 1e-6, max: 1e6 },
  ],
  chirp: [
    { key: 'amplitude', label: 'Amplitude', min: 0, max: 1e6 },
    { key: 'f0', label: 'Start frequency', min: 0, max: 1e6 },
    { key: 'f1', label: 'End frequency', min: 0, max: 1e6 },
    { key: 'duration', label: 'Duration', min: 1e-9, max: 1e9 },
    { key: 'delay', label: 'Delay', min: 0, max: 1e9 },
  ],
  burst: [
    { key: 'amplitude', label: 'Amplitude', min: 0, max: 1e6 },
    { key: 'frequency', label: 'Frequency', min: 1e-6, max: 1e6 },
    { key: 'cycles', label: 'Cycles', min: 1, max: 1000 },
    { key: 'delay', label: 'Delay', min: 0, max: 1e9 },
  ],
  noise: [
    { key: 'amplitude', label: 'Amplitude', min: 0, max: 1e6 },
    { key: 'duration', label: 'Duration', min: 0, max: 1e9 },
    { key: 'seed', label: 'Seed', min: 0, max: 1e9 },
  ],
  samples: [
    { key: 'amplitude', label: 'Amplitude', min: 0, max: 1e6 },
    { key: 'rate', label: 'Samples per time unit', min: 1e-6, max: 1e9 },
    { key: 'delay', label: 'Delay', min: 0, max: 1e9 },
  ],
};

/** Waveform editor with sampling/dispersion guidance for the current grid. */
export function WaveformEditor(props: { value: WaveformSpec; onChange: (w: WaveformSpec) => void; dt: number; dx: number; c: number }) {
  const { value, onChange, dt, dx, c } = props;
  const f = nominalFrequency(value);
  const cyclesPerStep = f * dt;
  const cellsPerWavelength = c / Math.max(f, 1e-12) / dx;
  let warning: string | null = null;
  if (cyclesPerStep >= 0.5) warning = `f·Δt = ${cyclesPerStep.toFixed(2)} ≥ 0.5: above the time-sampling Nyquist limit; the source will alias.`;
  else if (cellsPerWavelength < 2) warning = `λ = ${cellsPerWavelength.toFixed(1)} cells < 2: the grid cannot carry this frequency.`;
  else if (cellsPerWavelength < 8)
    warning = `λ = ${cellsPerWavelength.toFixed(1)} cells: expect visible numerical dispersion (aim for ≥ 8–10 cells per wavelength).`;
  return (
    <div>
      <div className="field">
        <label>Waveform</label>
        <select
          className="input"
          value={value.type}
          aria-label="Waveform type"
          onChange={(e) => onChange(defaultWaveform(e.target.value as WaveformType))}
        >
          {(Object.keys(WAVEFORM_LABELS) as WaveformType[])
            .filter((t) => t !== 'samples' || value.type === 'samples')
            .map((t) => (
              <option key={t} value={t}>
                {WAVEFORM_LABELS[t]}
              </option>
            ))}
        </select>
      </div>
      <div className="row wrap" style={{ alignItems: 'flex-start' }}>
        {PARAMS[value.type].map((p) => (
          <div key={p.key} style={{ flex: '1 1 45%' }}>
            <NumberField
              label={p.label}
              value={(value as unknown as Record<string, number>)[p.key]}
              min={p.min}
              max={p.max}
              onChange={(v) => onChange({ ...value, [p.key]: v } as WaveformSpec)}
            />
          </div>
        ))}
      </div>
      {warning ? (
        <p className="warn" role="status" style={{ color: 'var(--warn)', fontSize: 12 }}>
          {warning}
        </p>
      ) : (
        <p className="hint dim" style={{ fontSize: 12 }}>
          {`λ ≈ ${cellsPerWavelength.toFixed(0)} cells, ${(1 / Math.max(cyclesPerStep, 1e-12)).toFixed(0)} steps per period.`}
        </p>
      )}
    </div>
  );
}
