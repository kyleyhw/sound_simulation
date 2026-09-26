/** Lab page logic: capture import, distance parsing, room geometry and CTC rate. */
import { describe, expect, it } from 'vitest';
import { designCtc, separationDb } from '../../src/control/ctc';
import { type Capture, mergeCaptures, parseDistance } from '../../src/lab/captures';
import { shoeboxFirstEchoes, shoeboxGeometryError } from '../../src/lab/measure';

const cap = (id: string, label = id): Capture => ({ id, label, createdAt: '2026-09-26', sampleRate: 48000, irs: [], estimates: {} });

describe('capture import', () => {
  it('re-ids imported captures that collide, so deleting one row keeps the other', () => {
    let n = 0;
    const merged = mergeCaptures([cap('a')], [cap('a', 'copy'), cap('b'), cap('b', 'b2')], () => `new${n++}`);
    expect(merged.map((c) => c.label)).toEqual(['a', 'copy', 'b', 'b2']);
    expect(new Set(merged.map((c) => c.id)).size).toBe(4);
    expect(merged[0].id).toBe('a');
    const afterDelete = merged.filter((c) => c.id !== merged[1].id);
    expect(afterDelete.map((c) => c.label)).toEqual(['a', 'b', 'b2']);
  });
});

describe('tape-measured distance', () => {
  it('keeps decimals and rejects garbage', () => {
    expect(parseDistance('')).toEqual({ value: undefined, error: null });
    expect(parseDistance('1.25').value).toBe(1.25);
    expect(parseDistance('1,5').value).toBe(1.5);
    expect(parseDistance('1.').value).toBe(1);
    expect(parseDistance('abc').error).not.toBeNull();
    expect(parseDistance('-2').error).not.toBeNull();
    expect(parseDistance('0').error).not.toBeNull();
  });
});

describe('room geometry', () => {
  const room = { lx: 5, ly: 4, lz: 2.7 };
  it('accepts a laptop inside the room', () => {
    expect(shoeboxGeometryError(room, [1.2, 2, 0.8])).toBeNull();
    expect(Math.min(...shoeboxFirstEchoes(room, [1.2, 2, 0.8]))).toBeGreaterThan(0);
  });
  it('rejects a laptop outside the room instead of predicting negative distances', () => {
    expect(shoeboxGeometryError(room, [10, 2, 0.8])).toMatch(/outside the room/);
    expect(shoeboxGeometryError(room, [1, 2, 2.7])).toMatch(/Laptop z/);
    expect(shoeboxFirstEchoes(room, [10, 2, 0.8])).toEqual([]);
    expect(shoeboxGeometryError({ ...room, ly: 0 }, [1, 0.5, 1])).toMatch(/width/);
  });
});

describe('virtual headphones at the device rate', () => {
  it('designs working CTC filters at 44.1 kHz (ConvolverNode buffers must match the context rate)', () => {
    const geo = { speakerSpan: 0.3, head: { x: 0, y: 0.5 } };
    const f44 = designCtc(geo, 44100, 1024, 0.005);
    const f48 = designCtc(geo, 48000, 1024, 0.005);
    expect(f44.sampleRate).toBe(44100);
    const freqs = [500, 1000, 2000, 4000];
    const s44 = separationDb(f44, geo, geo, freqs);
    const s48 = separationDb(f48, geo, geo, freqs);
    for (let k = 0; k < freqs.length; k++) expect(Math.abs(s44[k] - s48[k])).toBeLessThan(6);
    expect(Math.min(...s44)).toBeGreaterThan(15);
  });
});
