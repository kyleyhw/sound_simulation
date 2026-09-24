# Measuring a real room with a laptop

*Write-up · real-hardware tooling (plan Phase 9)*

The simulations are only worth something if they match real rooms. The
[Lab](#/lab) page turns a laptop into a measuring instrument using only
its own speakers, microphones and webcam. Nothing is uploaded: all
processing happens in the browser. This write-up describes what the
tools do and how they were checked, and what still needs a person in a
real room.

## The measurement chain

1. **Impulse response.** The Lab plays an exponential sine sweep and
   records it, with the browser's echo cancellation, noise suppression and
   automatic gain control all switched off. Convolving the recording with
   the sweep's inverse filter (Farina 2000) gives the room's impulse
   response. Loudspeaker distortion lands at negative times and is
   discarded.
2. **Device calibration.** The laptop's own speaker-to-mic path arrives
   milliseconds before any room echo. A short window around it is the
   device's response. Later measurements are equalised by a regularised
   inverse. That makes echo peaks sharper and gives the round-trip
   latency.
3. **Echoes to distances.** Peaks of the impulse response give reflector
   distances, $d = c\,\Delta t / 2$.
4. **Reverberation.** The Schroeder backward integral gives the energy
   decay curve. Line fits give T20, T30 and EDT, broadband and in octave
   bands.
5. **Room twin.** The measured room is rebuilt as a 3D FDTD shoebox. Its
   walls use the admittance $\beta$ that reproduces the entered
   absorption coefficient under random incidence (Paris's average). The
   twin's T30 is compared with Sabine, Eyring and the measurement.
6. **Virtual headphones.** Crosstalk cancellation drives both speakers so
   that each ear hears only its own channel. A webcam head tracker
   (MediaPipe BlazeFace) keeps the filters pointed at the listener.

## How the tools were checked

- **Synthetic rooms with known answers:** the recovered echo delays and
  T60 match what was put in.
- **Device calibration:** equalisation turns device-coloured echoes back
  into sharp peaks (over 90 % of the energy within ±2 samples).
- **The FDTD twin:** its T30 lies near the Sabine/Eyring range. For small
  rooms at low frequency it runs longer, because the field is modal,
  not diffuse.
- **Crosstalk cancellation:** over 15 dB of modelled channel separation
  at the design head position.
- **In a browser:** the full pipeline runs end to end in Chromium with a
  fake microphone and camera: measure, calibrate, re-measure equalised,
  save, simulate the twin.
- **Offline:** `scripts/eval_real_captures.py` recomputes every estimate
  with an independent NumPy implementation. Exported captures can then be
  scored against tape measurements.

## What still needs a real room

The Phase 9 targets need a laptop, a room and a tape measure:
- the distance to a wall within a few centimetres;
- T60 within 10 % of the simulated twin;
- a working virtual-headphones demo.

The tools are ready and tested, but the measurements have not been made
yet. The procedure:
1. Measure once with equalisation off.
2. Calibrate the device.
3. Measure at a few spots, entering the tape distance each time.
4. Export the capture set and run the scoring script.

This write-up will be updated with those numbers.
