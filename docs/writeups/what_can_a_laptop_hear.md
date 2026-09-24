# What can a laptop hear?

*Write-up · room sensing (plan Phases 2, 3 and 6), negative results included*

Can a laptop map the room it sits in using only its own speaker and two
microphones? This project set out to answer that in simulation first: a
64 × 64-cell room with obstacles, one speaker playing a chirp, two mics
12 cells apart, recorded at several positions ("poses"). The honest
answer took two attempts.

## Attempt one: a neural network that learned the room statistics

The first approach trained convolutional networks to go straight from
spectrograms of the stereo recordings to an obstacle map. Several poses
were fused with a calibrated Bayes rule. After careful work (phase
channels, multi-scale skips, calibration, a validated operating point),
the best models reached **IoU 0.100** on 500 held-out rooms.

A 2026-09 audit asked a simple question: what does a model score if it
*never listens*? The obvious no-audio baseline takes the per-pixel
average of the training rooms, thresholded at a level chosen on
training rooms. It scores **0.1015 ± 0.003** on the same rooms. The
published networks were indistinguishable from it. What they had
learned was where obstacles usually are, not what the echoes said.

The audit found more:
- validation rooms leaking into training through sibling poses;
- the prior used for fusion computed from the held-out set;
- a headline number that the documented command could not reproduce.

All of it is fixed and listed in [the debug audit](../../tests/reports/debug_audit_2026_09_24.md).
The Phase 2 conclusion that 0.10 was the "information ceiling" of the
task was therefore unfounded.

## Attempt two: physics first, and always against the baseline

The second attempt started from imaging methods with no learning at all.
Each was fused with the same prior and scored with paired per-room
differences against the no-audio baseline. Nothing was tuned on
held-out rooms. The methods:

- **Deconvolution** of the known chirp, to recover impulse responses.
- **Echo ellipses and free-space carving.** An echo at delay τ places a
  reflector on an ellipse. The *absence* of an echo before τ proves the
  space inside is empty.
- **Synthetic-aperture back-projection.** It sums every recording at the
  delay a reflector at each pixel would produce, across all poses and
  both mics.
- **Time reversal.** The residuals are re-emitted backwards through the
  simulator.
- **Full-waveform inversion.** The obstacle map itself is optimised
  through the differentiable simulator.

Results on 500 held-out rooms, K = 4 poses (IoU; paired Δ against the
no-audio baseline, ± SE):

| method | IoU | Δ vs no-audio | z |
|---|---|---|---|
| no-audio baseline | 0.101 | — | — |
| published CNN (skip_v2) | 0.100 | ≈ 0 | — |
| echo ellipses | 0.128 | +0.027 ± 0.003 | 9.0 |
| back-projection | 0.168 | +0.066 ± 0.004 | 16.8 |
| **four physics images fused** | **0.189** | **+0.088 ± 0.005** | **19.3** |
| four fused, 8 poses | 0.244 | +0.143 ± 0.004 | 34.1 |
| four fused, 20 dB noise | 0.187 | +0.086 ± 0.005 | 18.5 |
| FWI (40 rooms, proof of concept) | 0.337 vs 0.087 | +0.251 ± 0.056 | 4.5 |

Sound does reveal room geometry, far beyond the baseline and robustly to
noise. The information was in the recordings. The first networks did not
extract it. Unlike the CNN fusion, the imagers keep improving with more
poses. Full-waveform inversion, which models the whole wavefield,
recovers thin walls almost exactly. On 40 rooms it triples the IoU.

A control shows where the gain comes from: telling the baseline that the
device's own cells are empty adds only 0.0004 IoU. The gain comes from
the audio.

## What limits a laptop

A Cramér–Rao analysis separates what is theoretically measurable from
what is resolvable. Range to a single wall is never the bottleneck:
millimetres at 20 dB SNR in any audible band. What limits a laptop is
**resolution**. Two reflectors closer than $c/2B$ merge (2 cm for a
0.3–8 kHz sweep), and features smaller than half the shortest wavelength
blur. A 20 cm mic pair gives bearings to about 1°, which is 5 cm
sideways at 3 m, with grating-lobe ambiguity at high frequencies. More
poses help as $1/\sqrt K$.

## Caveats

- **Simulated data.** Everything above uses simulated, noise-free data
  (plus one 20 dB noise test).
- **Idealised knowledge.** The exact device poses are known, and so is
  the empty room shell.
- **Inverse crime.** FWI inverts with the simulator that generated the
  data.

Real rooms add device colouration, unknown poses, clutter and noise. The
[Lab](#/lab) tools exist to measure exactly that gap.

## Try it

The **Sensing** tab in the sandbox runs the trained network in your
browser. It simulates poses in whatever room you draw, fuses them, shows
where it is uncertain, suggests where to move next, and scores itself
against the no-audio baseline. The **Loop** page uses coherent
back-projection to build a digital twin for sound-field control.

Full numbers: [physics imaging report](../../tests/reports/imaging_2026_09_24.md),
[debug audit](../../tests/reports/debug_audit_2026_09_24.md),
[plan audit](../plan_audit.md).
