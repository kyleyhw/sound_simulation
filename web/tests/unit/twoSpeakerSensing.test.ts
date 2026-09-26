/**
 * The exported two-speaker room estimate against PyTorch (study
 * tests/reports/two_speaker_2026_09_26.md): the browser re-simulates held-out
 * scenes with the device's emission schedule, separates, migrates, computes
 * the features and runs the U-Net, and must reproduce the training data's
 * images and the training script's features and logits (fixtures from
 * scripts/train_loop_sensing.py export --fixtures on the scheme's checkpoint).
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  loopFeatures,
  type LoopModelManifest,
  LoopUNet,
} from "../../src/loop/learnedSensing";
import { randomLoopScene } from "../../src/loop/scenarios";
import { TwoSpeakerSensor } from "../../src/twospeaker/sensing";

const root = new URL("../../public/models/", import.meta.url);

/** The exported schemes: the device in one place, and moved to K = 4 placements. */
for (const SCHEME of ["seq", "seq_k4"]) {
  const manifest = JSON.parse(
    readFileSync(new URL(`two_speaker_${SCHEME}.json`, root), "utf8"),
  ) as LoopModelManifest & { scheme: string; device: number[][] };
  const buf = readFileSync(new URL(`two_speaker_${SCHEME}.bin`, root));
  const model = new LoopUNet(
    manifest,
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  );
  const fx = JSON.parse(
    readFileSync(
      new URL(`../fixtures/two_speaker_${SCHEME}_parity.json`, import.meta.url),
      "utf8",
    ),
  ) as {
    stride: number;
    rooms: {
      split: string;
      index: number;
      seed: number;
      raw_sum: number[];
      raw_absmax: number[];
      feature_sample: number[];
      logits: number[];
    }[];
  };

  describe(`two-speaker sensing (${SCHEME}) matches PyTorch`, () => {
    const sensor = new TwoSpeakerSensor(model);
    const n = manifest.grid;

    it("loads the scheme and device from the manifest", () => {
      expect(sensor.scheme.name).toBe(manifest.scheme);
      expect(manifest.device.length).toBeGreaterThanOrEqual(4);
    });

    for (const room of fx.rooms) {
      it(`test scene ${room.seed}: emission, separation, images, features and logits`, () => {
        const truth = randomLoopScene(room.seed, n).truth;
        const r = sensor.sense(truth, { truth: truth.materials });
        const im = r.images;
        // 1. Images equal the training data's (same code, so to rounding).
        [im.coherent, im.left, im.right, im.image, im.incoherent].forEach(
          (a, c) => {
            let s = 0;
            let mx = 0;
            for (const v of a) {
              s += v;
              mx = Math.max(mx, Math.abs(v));
            }
            expect(Math.abs(mx - room.raw_absmax[c])).toBeLessThanOrEqual(
              1e-5 * room.raw_absmax[c],
            );
            expect(Math.abs(s - room.raw_sum[c])).toBeLessThanOrEqual(
              1e-4 * n * n * room.raw_absmax[c],
            );
          },
        );
        // 2. Features (the device's elements set the near field and centre).
        const x = loopFeatures(im, r.elements, n, manifest);
        let ferr = 0;
        room.feature_sample.forEach(
          (v, k) => (ferr = Math.max(ferr, Math.abs(x.d[k * fx.stride] - v))),
        );
        expect(ferr).toBeLessThan(2e-5);
        // 3. Logits against PyTorch.
        const logits = model.forward(x);
        let err = 0;
        let scale = 0;
        room.logits.forEach((v, q) => {
          err = Math.max(err, Math.abs(logits[q] - v));
          scale = Math.max(scale, Math.abs(v));
        });
        process.stderr.write(
          `${SCHEME} ${room.seed}: |feature err| ${ferr.toExponential(1)}, |logit err| ${err.toExponential(1)} (scale ${scale.toFixed(1)}), IoU ${r.iou?.learned?.toFixed(2)}\n`,
        );
        expect(err).toBeLessThan(2e-3 * Math.max(1, scale));
        expect(r.estimate).not.toBeNull();
      }, 120_000);
    }
  });
}
