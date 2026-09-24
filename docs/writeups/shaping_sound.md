# Shaping sound with speakers and a simulator

*Write-up · sound-field control (plan Phases 7 and 8)*

Once you can simulate a room, you can ask how to drive several speakers
so that sound goes where you want it, and stays out of where you don't.
This write-up summarises what the project achieved in simulated rooms
with realistic, partially absorbing walls. Every figure is measured by
playing the designed filters through the time-domain simulator, not just
predicted. Full numbers are in the [control report](../../tests/reports/control_2026_09_24.md).

## Loud here, quiet there

Eight speakers, 10 cm apart, face two zones 1 m apart in a furnished
3 × 2.4 m room. The walls absorb 71 % at normal incidence. Measured
bright/dark contrast, broadband over 300–1500 Hz:

| method | contrast |
|---|---|
| delay-and-sum (steer at the bright zone) | 8.8 dB |
| time-reversal focusing | 7.6 dB |
| pressure matching | 16.1 dB |
| **acoustic contrast control** | **25.5 dB** |

Acoustic contrast control chooses per-speaker filters that maximise the
ratio of bright-zone to dark-zone energy. The weights come from the top
generalised eigenvector of the two zones' correlation matrices. In a more
reverberant room it still reaches 17.2 dB. The target was 10 dB. The
sandbox's **Control** tab runs the same designs live in the browser, and
the **Quiet zone** gallery scene reaches 58 dB at a single frequency.

## Virtual headphones from two laptop speakers

Crosstalk cancellation drives both speakers so that each ear hears only
its own channel, so stereo from a laptop can sound like headphones.
- **Absorbing room:** filters designed from simulated transfer functions
  give 17.7 dB and 20.5 dB of broadband separation, meeting the 15 dB
  target.
- **Limits:** the target is not met at every frequency (84 % of the
  band; at room resonances the separation dips), and a live room reaches
  only 9.8 dB.
- **Head position:** it is very sensitive. Fixed filters hold 15 dB only
  within about ±2.5 cm of head movement. Filters re-designed for a
  tracked head keep 35–43 dB over ±15 cm. That is why the Lab pairs
  crosstalk cancellation with webcam head tracking.

## Quiet zones by noise cancellation

An adaptive (filtered-x LMS) canceller with one speaker and one error mic
removes a tone at the mic almost completely: 35–120 dB. What matters is
the size of the quiet region around it. It shrinks with frequency, at
roughly 1–2.5 × λ/10:

| frequency | ≥ 10 dB quiet zone |
|---|---|
| 150 Hz | 53 cm |
| 500 Hz | 11 cm |
| 1 kHz | 4 cm |

Broadband noise (100–500 Hz) is harder: 22 dB in the absorbing room,
11 dB in the live one.

## How well must the room be known?

This question links control back to sensing. Design the controller from
an *estimated* room, then run it in the *true* one:

| wall position error | absorbing room | live room |
|---|---|---|
| none | 27.5 dB | 23.3 dB |
| 2.5 cm | 18.0 dB | 9.2 dB |

To keep 15 dB, walls must be known to about **4 cm** in the absorbing
room and **1.5 cm** in the live one. Beyond about 5 cm of error, the
design is no better than one that ignores the room. Absorption can be
off by ±30 %, and furniture by about 5 cm.

That sets the bar for Phase 6. The physics imagers resolve features of a
few centimetres at best, so room-aware control is on the edge of what a
laptop can sense.

## Closing the loop

The [closed-loop demo](#/loop) runs the whole chain in the browser in
four steps:
1. The speaker bar pings the room.
2. Coherent back-projection of the echoes builds a digital twin.
3. The controller is designed on the twin.
4. The result is measured in the true room.

The scene changes during the run: a listener moves, an obstacle moves,
and a partition appears. The loop keeps 16–41 dB of contrast throughout,
while a controller designed once and never updated falls behind. The
remaining 10–15 dB gap to a controller that knows the true room is the
sensing gap. That is consistent with the wall-accuracy requirement above.

## Caveats

- **2D.** All scenes are 2D, and there is no head model in the crosstalk
  scenes.
- **Rate.** The filters run at the simulator's 27.4 kHz rate.
- **Noise reference.** The noise canceller uses the noise signal itself
  as its reference.
- **Real hardware.** Speaker responses, latency and head shadowing still
  have to be measured with the [Lab](#/lab) tools.
