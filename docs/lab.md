# Lab: real-hardware tooling (Phase 9)

The Lab page (`web/src/pages/Lab.tsx`, route `#/lab`) measures the room
the user is sitting in with the laptop's own speakers and microphones.
Everything runs in the browser, and no audio or video leaves the device.
The Python side (`scripts/eval_real_captures.py`,
`acoustic_system/utils/room_ir.py`, `acoustic_system/learning/augment.py`)
scores exported captures offline and trains the sensing models to be robust
to real devices.

## Measurement chain

| Step | Code | Method |
|---|---|---|
| Play and record (9.1) | `lab/audioio.ts` | AudioWorklet recorder; `getUserMedia` with `echoCancellation`, `noiseSuppression` and `autoGainControl` all **off**; 0.3 s pre-roll and 1 s tail. |
| Impulse response (9.2) | `lab/measure.ts` `essSweep`, `essInverse`, `deconvolve` | Exponential sine sweep (Farina 2000). Convolving the recording with the inverse sweep gives the linear IR; harmonic distortion lands at negative time and is discarded. |
| Device calibration (9.6) | `directPathWindow`, `equalize`, `roundTripLatencyMs` | Latency is the direct-peak delay minus the pre-roll. The device response is a short window around the direct speaker-to-mic path. Later IRs are equalised by a regularised inverse, `G = H·conj(D) / (\|D\|² + ε·max\|D\|²)`, which sharpens echo peaks. The calibration is stored per browser. |
| Echo distance (9.3) | `findEchoes` | Local maxima of \|h\| above −24 dB, at least 0.8 ms apart. Reflector distance = c·Δt/2. |
| Reverberation (9.4) | `decayMetrics`, `octaveBand` | Schroeder backward integral; T20 / T30 / EDT by least-squares fit. Octave bands 250 Hz to 4 kHz. |
| Room twin (9.5) | `lab/twin.ts` | Sabine and Eyring T60, plus a 3D FDTD shoebox with locally reacting walls. The wall admittance β comes from α by inverting the Paris random-incidence average ∫(1−\|R(θ)\|²) sin 2θ dθ. The page runs it in a Web Worker (`lab/twinWorker.ts`) and rejects a laptop position outside the room. |
| Capture set (9.7) | `lab/captures.ts` | Labelled IRs with the tape-measured distance, room size, pose, latency and phone orientation. Kept in localStorage and exported as JSON. |
| Head tracking (9.9) | `lab/headtrack.ts` | MediaPipe BlazeFace is loaded from a CDN on demand. It uses a pinhole model: distance = f·0.15 m / face width in pixels. |
| Virtual headphones (9.10) | `control/ctc.ts` | Kirkeby regularised 2×2 inverse with a modelling delay. Free-field plant, ears at ±8.75 cm. The filters are redesigned as the head moves, at the audio device's own sample rate (often 44.1 kHz). |
| Phone as sensor (9.11) | Lab §5 | `DeviceOrientationEvent` (with the iOS permission prompt). The orientation is stored with each capture. |

## Offline scoring

```
uv run python scripts/eval_real_captures.py captures.json --alpha 0.15 --json results.json
```

For each capture, the script recomputes the estimates with the NumPy port in
`utils/room_ir.py`. It reports:

- first-echo distance against the tape measurement: per-capture error, MAE, median, and how many are within 5 cm;
- T30 and T20, and the Eyring α they imply for the entered room;
- the browser-versus-offline deltas, as a cross-check of the two implementations.

## Sim-to-real randomisation (9.8)

`train.py --augment-device` applies a random device response to the
training view: a direct tap plus 1–3 damped resonances, normalised to unit
energy, with a small per-channel variation. It also adds a latency of 0–3
samples. This comes on top of the existing `--augment` gain and noise jitter.
The validation view stays clean.

The sim-trained models cannot yet be applied to a real capture.
The v2 protocol's physical scale is implausible: `units.plausibility`
reports a 1.07 m room, or a 94 cm mic baseline. A v3 protocol at laptop
scale is needed before real IRs can be fed to a model (Phase 6).

## Tests

- `web/tests/unit/lab.test.ts`: sweep and deconvolution, echoes and decay metrics on synthetic rooms.
- `web/tests/unit/calibration.test.ts`: equalisation restores sharp echoes; latency arithmetic.
- `web/tests/unit/twin.test.ts`: α↔β round trip; FDTD shoebox T30 in the Sabine/Eyring range.
- `web/tests/unit/ctc.test.ts`: at least 15 dB separation at the design point; degradation off-axis.
- `web/tests/e2e/lab.spec.ts`: the full pipeline in Chromium with a fake microphone and camera. It covers measure, calibrate, re-measure equalised, save a capture, run the twin, and the CTC plot.
- `tests/lab/test_real_captures.py`, `tests/learning/test_augment.py`.

## What needs a real room

The tools are verified on synthetic inputs and on Chromium's fake audio
device. The Phase 9 "done when" numbers need a person, a laptop and a tape
measure: distance within a few cm, T60 within 10 % of the twin, and a
working virtual-headphones demo. The workflow is:

1. Measure with equalisation off.
2. Click *Calibrate device from this*.
3. Measure again at a few spots, entering the tape distance each time.
4. Export the captures and run the script above.
