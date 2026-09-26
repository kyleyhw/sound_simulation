/**
 * Echo vision playback (#/echo): the lockstep sweep (room and empty room),
 * the scattered-field frames, the picture built ping by ping, and the
 * playback schedule.
 */
import { describe, expect, it } from 'vitest';
import { Simulation } from '../../src/engine/simulation';
import { migrationImages, pingRecordings } from '../../src/loop/closedLoop';
import { activeSpeakers, barDevice } from '../../src/echo/device';
import { EXAMPLES, N } from '../../src/echo/room';
import {
  cumulativeImages,
  decodeValue,
  echoSetup,
  echoWindow,
  encodeFrame,
  F0,
  frameCount,
  geometry,
  listenSweep,
  residuals,
  singleImages,
} from '../../src/echo/sense';
import { availableSeconds, displayScales, locate, schedule, stageOf, totalSeconds } from '../../src/echo/timeline';

const { params, array, steps } = echoSetup();
const device = barDevice();
const door = EXAMPLES.find((e) => e.id === 'door')!.build();
const room = geometry(door);
const emptyRoom = geometry(new Uint8Array(N * N));

describe('echo playback: the device', () => {
  it('the bar is the loop array, each speaker clicking alone with the loop ping', () => {
    expect(device.speakers).toEqual(array);
    expect(device.mics).toEqual(array);
    expect(device.emissions.length).toBe(array.length);
    for (let e = 0; e < array.length; e++) expect(activeSpeakers(device, e)).toEqual([e]);
    expect(device.f0).toBe(F0);
    expect(device.delay).toBeCloseTo(1.5 / F0, 12);
  });
});

describe('echo playback: the lockstep sweep', () => {
  // Two pings are enough for the frame checks; the full sweep is checked against the loop below.
  const two = { ...device, emissions: device.emissions.slice(0, 2) };
  const win = echoWindow(door, device, params, steps);
  const got: { frames: Int8Array; scales: Float32Array }[] = [];
  listenSweep(room, two, steps, win, (r) => got.push({ frames: r.frames, scales: r.scales }));

  it('the window runs from the click to past the farthest echo, within the recording', () => {
    const dt = new Simulation(params).dt;
    expect(win.from * dt).toBeLessThan(device.delay - 0.5 / F0);
    expect(win.to).toBeLessThan(steps);
    expect((win.to - win.from) % win.every).toBe(0);
    // The partition's top end (row 8) is the farthest cell: its echo must be inside the window.
    const far = Math.max(...array.map((a) => Math.hypot(8 - a[0], 48 - a[1]))) * 2;
    expect(win.to * dt).toBeGreaterThan(device.delay + far / params.c);
    expect(got[0].frames.length).toBe(frameCount(win) * N * N);
  });

  it('frames are the scattered field: room minus empty room, zero before the click reaches anything', () => {
    const f = Math.round((260 - win.from) / win.every); // a frame with echoes everywhere
    const k = win.from + f * win.every;
    const a = new Simulation(params);
    a.setMaterialMap(door);
    const b = new Simulation(params);
    for (const s of [a, b]) s.setDrivers([{ id: 'p', pos: array[0], waveform: device.emissions[0].drivers[0].waveform, enabled: true }]);
    let early = -1;
    for (let s = 0; s <= k; s++) {
      a.step();
      b.step();
      if (s === win.from) early = got[0].scales[0];
    }
    // The first frame: the click has not reached an obstacle, so the difference is exactly zero.
    expect(early).toBe(0);
    const S = got[0].scales[f];
    expect(S).toBeGreaterThan(0);
    let err = 0;
    let peak = 0;
    for (let q = 0; q < N * N; q++) {
      const want = a.p[q] - b.p[q];
      peak = Math.max(peak, Math.abs(want));
      err = Math.max(err, Math.abs(decodeValue(got[0].frames[f * N * N + q], S) - want));
    }
    expect(S).toBeCloseTo(peak, 6);
    // 8-bit gamma coding: worst error about one code step at the top of the range.
    expect(err).toBeLessThan(0.02 * S);
  });

  it('records exactly what the loop records, in the room and in the empty room', () => {
    const { recRoom, recEmpty } = listenSweep(room, device, steps, null);
    const loopRoom = pingRecordings(room, array, F0, steps);
    const loopEmpty = pingRecordings(emptyRoom, array, F0, steps);
    for (let s = 0; s < array.length; s++)
      for (let m = 0; m < array.length; m++) {
        expect(recRoom[s][m]).toEqual(loopRoom[s][m]);
        expect(recEmpty[s][m]).toEqual(loopEmpty[s][m]);
      }
    const r = residuals(recRoom[0], recEmpty[0]);
    expect(r.length).toBe(array.length * steps);
    expect(r[3 * steps + 400]).toBe(Math.fround(recRoom[0][3][400] - recEmpty[0][3][400]));
  }, 60_000);
});

describe('echo playback: frame coding', () => {
  it('round-trips within the 8-bit gamma step', () => {
    const v = Float32Array.from({ length: 2000 }, (_, i) => Math.sin(i * 0.37) * Math.exp(-i / 400));
    const out = new Int8Array(v.length + 5);
    const S = encodeFrame(v, out, 5);
    expect(S).toBeCloseTo(Math.max(...v.map(Math.abs)), 7);
    for (let q = 0; q < v.length; q++) {
      const a = Math.abs(v[q]) / S;
      // One code is 1/127 in sqrt(|v| / S): the error grows with sqrt(a).
      expect(Math.abs(decodeValue(out[5 + q], S) - v[q])).toBeLessThanOrEqual(S * ((2 * Math.sqrt(a)) / 254 + 1 / 254 ** 2) + 1e-7);
    }
    expect(encodeFrame(new Float32Array(10), out, 0)).toBe(0);
  });
});

describe('echo playback: the picture, ping by ping', () => {
  const recRoom = pingRecordings(room, array, F0, steps);
  const recEmpty = pingRecordings(emptyRoom, array, F0, steps);
  const full = migrationImages(recRoom, recEmpty, array, params, F0);

  it('the per-ping images sum to the full coherent image; the last cumulative image is the full one', () => {
    const sum = new Float64Array(N * N);
    for (let k = 0; k < array.length; k++) {
      const c = singleImages(recRoom, recEmpty, array, params, F0, k).coherent;
      for (let q = 0; q < sum.length; q++) sum[q] += c[q];
    }
    let peak = 0;
    let err = 0;
    for (let q = 0; q < sum.length; q++) {
      peak = Math.max(peak, Math.abs(full.coherent[q]));
      err = Math.max(err, Math.abs(sum[q] - full.coherent[q]));
    }
    expect(err).toBeLessThan(1e-5 * peak);
    const last = cumulativeImages(recRoom, recEmpty, array, params, F0, array.length - 1);
    expect(last.image).toEqual(full.image);
    expect(last.coherent).toEqual(full.coherent);
  }, 60_000);

  it('works while the sweep is still running (later pings not recorded yet)', () => {
    const partRoom = [recRoom[0], recRoom[1]];
    const partEmpty = [recEmpty[0], recEmpty[1]];
    const a = cumulativeImages(partRoom, partEmpty, array, params, F0, 1);
    const b = cumulativeImages(recRoom, recEmpty, array, params, F0, 1);
    expect(a.image).toEqual(b.image);
    // The picture grows as pings are added (the echoes add up where something is).
    const peak = (im: Float32Array) => im.reduce((m, v) => Math.max(m, v), 0);
    expect(peak(b.image)).toBeGreaterThan(peak(cumulativeImages(recRoom, recEmpty, array, params, F0, 0).image));
  }, 60_000);
});

describe('echo playback: the schedule', () => {
  const segs = schedule(8);

  it('plays the first ping slowly and the whole story in 8-12 s', () => {
    const T = totalSeconds(segs);
    expect(T).toBeGreaterThanOrEqual(8);
    expect(T).toBeLessThanOrEqual(12);
    expect(segs.map((s) => s.kind).join(',')).toBe(`${'ping,trace,'.repeat(8)}guess`);
    const pings = segs.filter((s) => s.kind === 'ping').map((s) => s.t1 - s.t0);
    expect(pings[0]).toBeGreaterThanOrEqual(3);
    for (let k = 1; k < pings.length; k++) expect(pings[k]).toBeLessThanOrEqual(pings[k - 1]);
    for (let i = 1; i < segs.length; i++) expect(segs[i].t0).toBe(segs[i - 1].t1);
    // A two-emission device plays a shorter story on the same rules.
    expect(totalSeconds(schedule(2))).toBeLessThan(T);
  });

  it('locates a time in its segment and stage', () => {
    expect(locate(segs, -1)).toMatchObject({ index: 0, f: 0 });
    const mid = locate(segs, segs[3].t0 + 0.5 * (segs[3].t1 - segs[3].t0));
    expect(mid.seg).toMatchObject({ kind: 'trace', ping: 1 });
    expect(mid.f).toBeCloseTo(0.5, 9);
    expect(locate(segs, 1e9)).toMatchObject({ index: segs.length - 1, f: 1 });
    expect([stageOf('ping'), stageOf('trace'), stageOf('guess')]).toEqual([1, 2, 3]);
  });

  it('holds playback just short of what is not computed yet', () => {
    expect(availableSeconds(segs, 0, false)).toBe(0);
    const a2 = availableSeconds(segs, 2, false);
    expect(a2).toBeLessThan(segs[4].t0);
    expect(a2).toBeGreaterThan(segs[4].t0 - 1e-3);
    expect(locate(segs, a2).seg).toMatchObject({ kind: 'trace', ping: 1 });
    // All pings in but no answer: wait before the guess.
    expect(locate(segs, availableSeconds(segs, 8, false)).seg.kind).toBe('trace');
    expect(availableSeconds(segs, 8, true)).toBe(totalSeconds(segs));
  });

  it('keeps faint echoes visible without blowing up the tail', () => {
    const s = Float32Array.from([0, 0.1, 1, 0.8, 0.3, 0.1, 0.01, 0.001, 0.2]);
    const d = displayScales(s, 2, 0.3);
    for (let f = 0; f < s.length; f++) {
      expect(d[f]).toBeGreaterThanOrEqual(s[f]);
      expect(d[f]).toBeGreaterThanOrEqual(0.3 - 1e-7);
    }
    expect(d[2]).toBe(1);
    // After the peak the scale decays (half-life 2 frames) down to the floor.
    expect(d[4]).toBeCloseTo(0.8 * 2 ** -0.5, 6);
    expect(d[6]).toBeCloseTo(0.3, 6);
  });
});
