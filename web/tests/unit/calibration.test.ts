/** Device calibration: direct-path window and equalisation (plan 9.6). */
import { describe, expect, it } from 'vitest';
import { convolve, directPathWindow, equalize, findEchoes, roundTripLatencyMs } from '../../src/lab/measure';

const fs = 48000;

/** A ringing "laptop speaker": decaying 2.5 kHz resonance, ~1 ms long. */
function device(): Float32Array {
  const n = 64;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.exp(-i / 10) * Math.cos((2 * Math.PI * 2500 * i) / fs);
  return d;
}

describe('device calibration', () => {
  it('equalisation turns device-coloured echoes back into sharp impulses', () => {
    const room = new Float32Array(4000);
    const direct = 300;
    room[direct] = 1;
    const echoes = [direct + 290, direct + 700, direct + 1510];
    echoes.forEach((i, k) => (room[i] = 0.5 / (k + 1)));
    const measured = Float32Array.from(convolve(room, device()).subarray(0, room.length));
    // Calibrate on a direct-path-only recording of the same device.
    const dOnly = new Float32Array(4000);
    dOnly[direct] = 1;
    const calRec = Float32Array.from(convolve(dOnly, device()).subarray(0, dOnly.length));
    let pk = 0;
    for (let i = 0; i < calRec.length; i++) if (Math.abs(calRec[i]) > Math.abs(calRec[pk])) pk = i;
    const cal = directPathWindow(calRec, pk, fs, 2.5, 0.3);
    const eq = equalize(measured, cal, 1e-3);
    // Energy concentration: fraction of energy within +-2 samples of the true taps.
    const conc = (h: Float32Array) => {
      let tot = 0;
      let near = 0;
      for (let i = 0; i < h.length; i++) {
        tot += h[i] * h[i];
        if ([direct, ...echoes].some((j) => Math.abs(i - j) <= 2)) near += h[i] * h[i];
      }
      return near / tot;
    };
    expect(conc(eq)).toBeGreaterThan(0.9);
    expect(conc(eq)).toBeGreaterThan(conc(measured) + 0.3);
    const found = findEchoes(eq, fs, { thresholdDb: -12, minSeparationMs: 0.8 });
    expect(Math.abs(found.direct - direct)).toBeLessThanOrEqual(2);
    expect(found.echoes.map((e) => e.index).slice(0, 3).every((i, k) => Math.abs(i - echoes[k]) <= 2)).toBe(true);
  });

  it('round-trip latency subtracts the pre-roll and the flight time', () => {
    expect(roundTripLatencyMs(0.3 * fs + 480, 0.3 * fs, fs, 0)).toBeCloseTo(10, 6);
    expect(roundTripLatencyMs(0.3 * fs + 480, 0.3 * fs, fs, 0.343)).toBeCloseTo(9, 6);
  });
});
