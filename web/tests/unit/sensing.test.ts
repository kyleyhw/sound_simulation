/**
 * In-browser sensing (plan 4.3.5, 10.3) against the Python pipeline:
 * the browser engine must reproduce the archive's recordings, and the
 * TypeScript model must reproduce PyTorch's logits and fused map
 * (fixtures from scripts/export_web_model.py --fixtures).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { chirp, protocolSim, recordPose } from '../../src/sensing/acquire';
import { fuse, type ModelManifest, SkipModel } from '../../src/sensing/model';
import { tensor } from '../../src/sensing/nn';

const root = new URL('../../public/models/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('skip_v2.json', root), 'utf8')) as ModelManifest;
const buf = readFileSync(new URL('skip_v2.bin', root));
const model = new SkipModel(manifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const fx = JSON.parse(readFileSync(new URL('../fixtures/sensing_parity.json', import.meta.url), 'utf8')) as {
  rooms: { room: number; mask: number[]; drivers: number[][]; mics: number[][][]; sensor: number[][]; logits: number[][]; fused: number[]; front0: number[]; cross_mag0: number[] }[];
};

describe('in-browser sensing matches the Python pipeline', () => {
  const p = manifest.protocol;
  for (const room of fx.rooms) {
    it(`held-out room ${room.room}: simulation, front-end and CNN`, () => {
      const mask = Uint8Array.from(room.mask);
      const sim = protocolSim(p, mask);
      const src = chirp(p, sim.dt);
      const T = p.duration;
      // 1. The browser engine reproduces the archive's recordings.
      let simErr = 0;
      let simRef = 0;
      room.drivers.forEach((drv, k) => {
        const [a, b] = recordPose(sim, p, src, drv, room.mics[k]);
        const ref = room.sensor[k];
        for (let t = 0; t < T; t++) {
          simErr = Math.max(simErr, Math.abs(a[t] - ref[t]), Math.abs(b[t] - ref[T + t]));
          simRef = Math.max(simRef, Math.abs(ref[t]), Math.abs(ref[T + t]));
        }
      });
      // 2. The CNN on PyTorch's own front-end features reproduces its logits.
      const f = tensor(4, 33, 26, Float32Array.from(room.front0));
      const lg = model.forwardFeatures(f, src);
      let cnnErr = 0;
      for (let q = 0; q < lg.length; q++) cnnErr = Math.max(cnnErr, Math.abs(lg[q] - room.logits[0][q]));
      // 3. The front-end: magnitudes everywhere, phase wherever there is signal.
      const ours = model.features(Float32Array.from(room.sensor[0].slice(0, T)), Float32Array.from(room.sensor[0].slice(T)));
      const plane = 33 * 26;
      const cmax = Math.max(...room.cross_mag0);
      let magErr = 0;
      let phaseErr = 0;
      let noisy = 0;
      for (let q = 0; q < plane; q++) {
        for (const c of [0, 1]) magErr = Math.max(magErr, Math.abs(ours.d[c * plane + q] - room.front0[c * plane + q]));
        if (room.cross_mag0[q] > 1e-6 * cmax) {
          for (const c of [2, 3]) phaseErr = Math.max(phaseErr, Math.abs(ours.d[c * plane + q] - room.front0[c * plane + q]));
        } else noisy++;
      }
      console.log(
        `room ${room.room}: sim ${(simErr / simRef).toExponential(1)}, CNN ${cnnErr.toExponential(1)}, |X|^2 ${magErr.toExponential(1)}, phase ${phaseErr.toExponential(1)} (${noisy}/${plane} bins below 1e-6 of peak: phase is rounding noise)`,
      );
      expect(simErr / simRef).toBeLessThan(1e-3);
      expect(cnnErr).toBeLessThan(2e-3);
      expect(magErr).toBeLessThan(1e-3);
      expect(phaseErr).toBeLessThan(1e-3);
    }, 60_000);
  }

  it('end to end, the noise-phase bins flip at most 3 % of the thresholded map', () => {
    // Full browser pipeline (our STFT) vs PyTorch's fused map. The two differ
    // only through the phase of near-silent bins, which is float rounding
    // noise in either implementation. The trained model is sensitive to it,
    // so this is measured and bounded, not required to be exact.
    for (const room of fx.rooms) {
      const src = chirp(p, 0.5);
      const T = p.duration;
      const logits = room.sensor.map((s) => model.forward(Float32Array.from(s.slice(0, T)), Float32Array.from(s.slice(T)), src));
      const fused = fuse(logits, manifest.calibration);
      let agree = 0;
      let maxDiff = 0;
      for (let q = 0; q < fused.length; q++) {
        if (fused[q] > manifest.calibration.threshold === room.fused[q] > manifest.calibration.threshold) agree++;
        maxDiff = Math.max(maxDiff, Math.abs(fused[q] - room.fused[q]));
      }
      console.log(`room ${room.room}: thresholded maps agree on ${((100 * agree) / fused.length).toFixed(2)} % of cells, max |dp| ${maxDiff.toFixed(3)}`);
      expect(agree / fused.length).toBeGreaterThan(0.97);
    }
  }, 60_000);
});
