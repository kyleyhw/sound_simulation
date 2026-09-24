/** Closed loop in simulation (plan Phase 8): sense -> twin -> design -> act. */
import { describe, expect, it } from 'vitest';
import { runLoop, senseRoom } from '../../src/loop/closedLoop';
import { demoScenario } from '../../src/loop/scenarios';

describe('closed loop', () => {
  const sc = demoScenario(80);

  it('coherent back-projection finds the obstacle', () => {
    const ep = sc.epochs[0];
    const empty = { params: ep.truth.params, materials: new Uint8Array(ep.truth.materials.length), speed: null };
    const r = senseRoom(ep.truth, empty, sc.array, { truthObstacles: ep.truth.materials });
    expect(r.iou).not.toBeNull();
    expect(r.iou!).toBeGreaterThan(0.08);
    // Most of the estimate touches the true obstacle (within 2 cells).
    const [rows, cols] = ep.truth.params.shape;
    let near = 0;
    let tot = 0;
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) {
        if (!r.estimate[i * cols + j]) continue;
        tot++;
        let hit = false;
        for (let a = Math.max(0, i - 2); a <= Math.min(rows - 1, i + 2) && !hit; a++)
          for (let b = Math.max(0, j - 2); b <= Math.min(cols - 1, j + 2); b++) if (ep.truth.materials[a * cols + b]) hit = true;
        if (hit) near++;
      }
    expect(near / tot).toBeGreaterThan(0.5);
  }, 60_000);

  it('the guarded loop keeps >= 10 dB through every change and never loses to the static design', async () => {
    const res = await runLoop(sc.epochs, sc.array, sc.frequency);
    for (const r of res)
      console.log(`${r.label}: guarded ${r.guarded.toFixed(1)} (${r.guardChoice}) twin ${r.adaptive.toFixed(1)} static ${r.static.toFixed(1)} oracle ${r.oracle.toFixed(1)} empty ${r.naive.toFixed(1)}`);
    for (const r of res) {
      expect(r.guarded).toBeGreaterThanOrEqual(10);
      expect(r.guarded).toBeGreaterThanOrEqual(r.static - 1);
      expect(r.oracle).toBeGreaterThanOrEqual(r.guarded - 1);
      expect(Object.values(r.latencyMs).every((v) => v >= 0 && Number.isFinite(v))).toBe(true);
    }
  }, 300_000);
});
