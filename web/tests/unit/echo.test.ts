/**
 * Echo vision (#/echo): room editing, the training-family check, and the
 * animated ping sweep, which must record exactly what the loop's
 * pingRecordings records (the U-Net was trained on that).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { pingRecordings } from '../../src/loop/closedLoop';
import { type LoopModelManifest, LoopUNet } from '../../src/loop/learnedSensing';
import { maskEdges } from '../../src/echo/draw';
import { AREA, dragRect, EXAMPLES, familyCheck, N, paintRect, pointToCell, randomRoom, RIGID, roomObjects } from '../../src/echo/room';
import { analyseEchoes, echoSetup, F0, geometry, recordPings } from '../../src/echo/sense';

describe('echo vision: drawing', () => {
  it('maps canvas points to cells and clamps to the grid', () => {
    expect(pointToCell(0, 0, 500, 500)).toEqual([0, 0]);
    expect(pointToCell(499, 250, 500, 500)).toEqual([50, 99]);
    expect(pointToCell(-20, 900, 500, 500)).toEqual([99, 0]);
  });

  it('a block covers the dragged box, clamped away from the speaker bar', () => {
    expect(dragRect('block', [30, 60], [20, 40])).toEqual({ r0: 20, r1: 30, c0: 40, c1: 60 });
    // Dragging onto the bar (row 86) or past the edges stops at the drawable area.
    expect(dragRect('block', [60, 2], [95, 99])).toEqual({ r0: 60, r1: AREA.r1, c0: AREA.c0, c1: AREA.c1 });
    expect(dragRect('erase', [0, 0], [5, 5])).toEqual({ r0: AREA.r0, r1: AREA.r0, c0: AREA.c0, c1: AREA.c0 });
    expect(dragRect('look', [10, 10], [20, 20])).toBeNull();
  });

  it('a wall snaps to the dominant axis and is two cells thick', () => {
    expect(dragRect('wall', [10, 50], [40, 53])).toEqual({ r0: 10, r1: 40, c0: 50, c1: 51 });
    expect(dragRect('wall', [30, 70], [33, 20])).toEqual({ r0: 30, r1: 31, c0: 20, c1: 70 });
    // At the edge of the area the wall stays inside it.
    expect(dragRect('wall', [20, 99], [60, 99])).toEqual({ r0: 20, r1: 60, c0: AREA.c1 - 1, c1: AREA.c1 });
    expect(dragRect('wall', [90, 20], [90, 50])).toEqual({ r0: AREA.r1 - 1, r1: AREA.r1, c0: 20, c1: 50 });
  });

  it('paints and erases without mutating the input', () => {
    const empty = new Uint8Array(N * N);
    const a = paintRect(empty, { r0: 20, r1: 29, c0: 30, c1: 34 }, RIGID);
    expect(empty.some((v) => v)).toBe(false);
    expect(a.reduce((s, v) => s + (v ? 1 : 0), 0)).toBe(50);
    const b = paintRect(a, { r0: 20, r1: 29, c0: 32, c1: 32 }, 0);
    const objs = roomObjects(b);
    expect(objs.length).toBe(2);
    expect(objs.every((o) => o.rectangular)).toBe(true);
  });
});

describe('echo vision: outlines', () => {
  it('traces a mask as merged cell-edge segments', () => {
    // A 2 x 3 block at rows 1-2, columns 1-3 of a 4 x 5 grid: four straight sides.
    const m = new Uint8Array(20);
    for (let i = 1; i <= 2; i++) for (let j = 1; j <= 3; j++) m[i * 5 + j] = 1;
    expect(maskEdges(m, 4, 5).sort()).toEqual(
      [
        [1, 1, 4, 1],
        [1, 3, 4, 3],
        [1, 1, 1, 3],
        [4, 1, 4, 3],
      ].sort(),
    );
    // Cells on the grid border are closed by the border.
    expect(maskEdges(Uint8Array.from([1, 1, 1, 1]), 2, 2).length).toBe(4);
    expect(maskEdges(new Uint8Array(9), 3, 3)).toEqual([]);
  });
});

describe('echo vision: training family', () => {
  it('random rooms and the demo rooms are in the family; the hard examples are not', () => {
    for (let s = 1; s <= 200; s++) expect(familyCheck(randomRoom(s)), `seed ${s}`).toEqual({ inFamily: true, reason: null });
    for (const ex of EXAMPLES) expect(familyCheck(ex.build()).inFamily, ex.id).toBe(!ex.hard);
  });

  it('flags non-rectangular, oversized and crowded rooms', () => {
    let m = paintRect(new Uint8Array(N * N), { r0: 20, r1: 40, c0: 20, c1: 23 }, RIGID);
    m = paintRect(m, { r0: 37, r1: 40, c0: 20, c1: 40 }, RIGID); // an L
    expect(familyCheck(m).reason).toBe('shape');
    expect(familyCheck(paintRect(new Uint8Array(N * N), { r0: 20, r1: 50, c0: 20, c1: 50 }, RIGID)).reason).toBe('size');
    let many = new Uint8Array(N * N);
    for (let k = 0; k < 7; k++) many = paintRect(many, { r0: 10 + 8 * k, r1: 13 + 8 * k, c0: 20, c1: 24 }, RIGID);
    expect(familyCheck(many).reason).toBe('count');
  });
});

describe('echo vision: the ping sweep', () => {
  const { params, array, steps } = echoSetup();
  const room = geometry(EXAMPLES.find((e) => e.id === 'door')!.build());

  it('records exactly what the loop records, and shows every step', () => {
    let frames = 0;
    const mine = recordPings(room, array, F0, steps, () => frames++);
    const loop = pingRecordings(room, array, F0, steps);
    expect(frames).toBe(array.length * steps);
    for (let s = 0; s < array.length; s++) for (let m = 0; m < array.length; m++) expect(mine[s][m]).toEqual(loop[s][m]);
    expect(params.shape).toEqual([N, N]);
  }, 60_000);

  it('the network beats back-projection on a training-family room', () => {
    const root = new URL('../../public/models/', import.meta.url);
    const man = JSON.parse(readFileSync(new URL('loop_unet.json', root), 'utf8')) as LoopModelManifest;
    const buf = readFileSync(new URL('loop_unet.bin', root));
    const model = new LoopUNet(man, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    const t0 = performance.now();
    const recRoom = recordPings(room, array, F0, steps);
    const t1 = performance.now();
    const recEmpty = pingRecordings(geometry(new Uint8Array(N * N)), array, F0, steps);
    const t2 = performance.now();
    const r = analyseEchoes(recRoom, recEmpty, array, room, model);
    const t3 = performance.now();
    console.log(`echo sweep: pings ${(t1 - t0).toFixed(0)} ms, empty ${(t2 - t1).toFixed(0)} ms, analysis ${(t3 - t2).toFixed(0)} ms; IoU learned ${r.iouLearned.toFixed(2)} bp ${r.iouBackprojection.toFixed(2)}`);
    expect(r.iouLearned).toBeGreaterThan(0.6);
    expect(r.iouLearned).toBeGreaterThan(r.iouBackprojection);
  }, 60_000);
});
