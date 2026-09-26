var e=`# Room reconstruction with two speakers instead of an array (2026-09-26)

**Question (project owner).** "Can we do [the Echo vision room
reconstruction] without an array, and just two speakers? Note that we can
play different frequencies from the speakers so that they can interfere and
get information out of that."


**How this squares with the information study**
(\`information_2026_09_26.md\`). That study found that one placement of the
pair resolves only 2–3 % of the room's pixels *on its own*: about 200
independent measurements, mostly range along echo arcs. Yet the network
reaches IoU 0.56. Both hold. The test rooms are 1–3 boxes or walls, a few
dozen numbers, well inside those ~200 measurable degrees of freedom. The
network's learned box prior fills in the rest. The same prior is why it
fails on shapes outside the training family, and why moving the device
(more independent measurements) helps most. Rigid walls helped in the
linear analysis, which models the reverberation exactly. They hurt here
(0.44), where the images only migrate direct and first-order paths.

**Answer.**

- **Yes, two speakers work, at about three quarters of the array's quality.**
  A laptop-like device with 2 speakers and 2 mics reaches IoU 0.56 on
  held-out rooms. The 8-element bar reaches 0.73 with the same training
  budget. Back-projection gets 0.15 and the no-audio prior 0.03.
- **Moving the device closes the gap.** Two placements give 0.67. Four
  placements give 0.80, which beats the bar (+0.06 ± 0.01, z = 5.6). The
  four placements span 48 cells, against the bar's 28.
- **Interference between different frequencies adds nothing.** The wave
  equation is linear. What simultaneous emission can buy is speed or
  signal-to-noise ratio, not information:
  - Disjoint frequency bands halve the measurement time, but each speaker
    covers only half the band. IoU drops to 0.44, no better than playing the
    same pulse from both speakers at once (0.44).
  - Separable broadband noise codes recover the in-turn result (0.58
    against 0.56, z = 1.6), but only when the shot is 1.5× longer than
    pinging in turn.
  - Their real gain is noise robustness. At 10 dB echo SNR they keep IoU
    0.48, against 0.33 for pinging in turn (+0.15, z = 8.5).
- **Known reflective walls did not help here.** In a rigid-walled room the
  same device reaches IoU 0.44, and adding image-source terms to the
  migration gives 0.43. Both are below the anechoic 0.56.

Artifacts are in \`two_speaker_2026_09_26_artifacts/\`: \`results.json\`,
\`iou_by_scheme.png\`, \`iou_vs_placements.png\` and \`examples.png\`.

## 1. Physics: what two speakers can and cannot measure

**The wave equation is linear, and so is the FDTD engine.** A mic records

  y_m(t) = Σ_s (e_s * h_sm)(t),   so   Y_m(f) = Σ_s E_s(f) H_sm(f),

that is, each speaker's emission e_s convolved with its speaker-to-mic
impulse response h_sm. Two speakers playing at once record exactly the sum
of the two single-speaker recordings. In the engine this holds to float32
round-off: the relative error on the room recordings is 2.7e-7
(\`web/tests/unit/twoSpeaker.test.ts\`).

**Interference between different frequencies carries no extra
information.** Suppose speaker A plays f1 and speaker B plays f2.

- The recording has a line at f1, with value E_A(f1) H_Am(f1), and a line
  at f2, with value E_B(f2) H_Bm(f2).
- The audible beat (the |y|² cross term at f1 − f2) has amplitude and phase
  H_Am(f1) H_Bm(f2)*. That is a product of two numbers the two lines
  already give.
- Nothing couples f1 and f2 in a linear medium. A new difference frequency
  would need a nonlinear medium, such as a parametric array at very high
  sound levels.

**Same-frequency phase control (beams) is a linear combination.** A beam
shot with gains and delays (a, b) records a E_A H_A + b E_B H_B. Any set of
beam shots is a linear map of the pair (H_A, H_B). It can therefore recover
at most what measuring each speaker separately gives. With at least two
independent shots it recovers exactly that: 96 dB separation here, and an
identical IoU (§3.3). A beam can only improve the signal-to-noise ratio.

**The information ceiling for one placement** is the set of speaker × mic
transfer functions H_sm(f) over the band:

- 2 × 2 = 4 of them for the two-speaker device;
- 8 × 8 = 64 for the bar, of which 36 are distinct by reciprocity.

The aperture (device size) and the bandwidth set the resolution. Moving the
device adds aperture, because every placement adds its own four transfer
functions.

**What simultaneous emission can buy.**

1. **Speed**, if the mics can separate the emissions. Count samples per
   mic:
   - Each unknown response has useful length L_u (the room's echo window).
     In a band B it has about 2 B L_u degrees of freedom.
   - A recording of total length T gives 2 B T samples.
   - With S speakers sharing the band, separation therefore needs
     T ≳ S L_u.

   Pinging in turn already takes S (L + pulse length), which is about S L.
   So a full-band simultaneous emission cannot be much faster than pinging
   in turn. It can be faster only in two ways:
   - by giving each speaker part of the band (frequency division);
   - by leaning on a prior, such as sparse echoes or the U-Net's room
     prior.

   The multiple-exponential-sweep literature (§5) saves time for the same
   reason. There the sweeps are much longer than the impulse responses (for
   SNR), and overlapping them removes only the waiting.
2. **Signal-to-noise ratio at a fixed peak level.** A long code puts much
   more energy into the room than a short ping of the same peak amplitude.
   Here the 1244-step noise code carries about 12 dB more energy per
   speaker than one Ricker ping, and the two band pulses carry about
   31 % of a ping's energy between them (measured by the information
   study). After deconvolution this is the classic
   pulse-compression gain, as with Farina's sweeps or MLS.

Both predictions are confirmed below:

- the codes need T_code + L ≥ 2 L_u to separate (§3.2);
- the codes' advantage shows up in noise (§3.5).

## 2. Setup

**Room and scenes.** These are the loop study's
(\`loop_sensing_2026_09_25.md\`):

- the grid is 100², with a 12-cell CPML;
- scenes come from \`scenarios.randomLoopScene\`, with 1–3 rigid blocks or
  partitions with a door;
- the seeds are identical: train 1e6 + i, validation 2e6 + i, and test
  3e6 + i, for the same 100 held-out rooms;
- the pulse is a Ricker with f0 = 0.08 (37.5-step delay);
- the listening window is L = 622 steps, with dt = 0.5 dx / c.

**Device geometry.** Positions are (row, column) and all elements sit on
row 86, the bar's row.

| device | speakers | mics |
|---|---|---|
| narrow ("laptop") | (86, 39), (86, 51): 12 cells apart | (86, 41), (86, 49): 2 cells inboard |
| wide | (86, 31), (86, 59): 28 apart, the bar's ends | (86, 33), (86, 57) |
| K = 2 placements | narrow device centred at columns 39 and 51 | its own 2 mics per placement |
| K = 4 placements | narrow device centred at columns 27, 39, 51 and 63 | its own 2 mics per placement |
| bar (baseline) | 8 elements at (86, 31 + 4k), k = 0…7 | the same 8 positions |

- The moved device slides by its speaker spacing, so neighbouring
  placements share a speaker position.
- K = 4 spans speakers from column 21 to column 69, which is 48 cells.
- Each placement records only with its own mics.

**Differences from the information study.** The Born-Jacobian study
(\`information_2026_09_26.md\`) uses the same narrow and wide pairs and the
same bar. It differs in three ways:

- **Placement centres.** It took this study's first draft: 38/52 for K = 2
  and 24/38/52/66 for K = 4. The trained models here use 39/51 and
  27/39/51/63, which have the same aperture within ±3 cells.
- **Reflective rooms.** It fills the outer 12 cells with a wall material,
  so the walls sit at the CPML boundary. Here the whole 100² grid is air,
  with rigid walls at its edge.
- **Emission energy.** It scales every emission to the Ricker's energy.
  Here all emissions have peak amplitude 1.

**Physical scale.** "Seconds of sound" below assume 5 cm cells:

- the grid is 5 m and c = 343 m/s;
- one step is 72.9 µs;
- f0 is 549 Hz;
- the listening window is 45 ms.

In a real room, reverberation sets the listening time (hundreds of ms), but
the ratios between schemes carry over.

**Emission schemes** (\`web/src/twospeaker/device.ts\`):

| # | scheme | emission | separation | steps (ms at 5 cm) |
|---|---|---|---|---|
| 1 | \`bar8\` | 8 Ricker pings in turn, 8 mics (64 traces) | none needed | 4976 (363) |
| 2 | \`seq\` | 2 Ricker pings in turn, 2 mics (4 traces) | none needed | 1244 (91) |
| 2w | \`wide_seq\` | as 2, with the wide device | none needed | 1244 (91) |
| 3 | \`sum\` | both speakers at once, same Ricker | not separable: the sum trace is migrated under both source hypotheses | 622 (45) |
| 4 | \`band\` | at once: A emits the Ricker's low band (< 0.075), B its high band (> 0.085), with a guard gap | zero-phase low/high split at 0.08 | 705 (51) |
| 5 | \`code\` | at once: two independent pseudo-noise codes with the Ricker's spectrum, 1244 steps, peak 1 | joint least squares (CGLS, 200 iterations) for both responses, mapped to Ricker-equivalent traces | 1866 (136) |
| 5' | \`code622\`, \`chirp\`, \`code_mf\` | 622-step codes; up/down chirps (1244 steps); the naive per-speaker inverse filter | as named | 1244 / 1866 / 1866 |
| 6 | \`beams\` | 4 same-pulse shots: in phase, anti-phase, B delayed 12 steps, A delayed 12 steps | per-frequency least squares over the shots | 2536 (185) |
| 7 | \`seq_k2\`, \`seq_k4\` | scheme 2 at K = 2 and 4 placements | none needed | 2488 (181), 4976 (363) |
| W | \`seq_rigid\`, \`seq_rigid_img\` | scheme 2 in the same grid with **rigid** outer walls | direct-path migration; with first-order image sources | 1244 (91) |

**Imaging and model.** These are the loop's, unchanged:

- The residual is the room recording minus the empty-room recording. The
  reference is simulated in the same room (CPML or rigid) with the same
  emission.
- The separated traces feed \`pairMigration\`, a generalisation of
  \`closedLoop.migrationImages\` that reproduces it bit for bit for the bar.
  It produces five grid-aligned images:
  - the coherent image;
  - the left and right half-source images (with image sources: the direct
    paths and the wall-image paths);
  - the smoothed coherent energy;
  - the smoothed incoherent energy.
- The compact U-Net, \`scripts/train_loop_sensing.py --scheme\`, has 11
  input channels. The device's elements set the near-field and centre
  channels.

**Training budget.** This is identical for every scheme, including the
bar, so the comparison is fair. It is smaller than the loop study's:

- 1000 training rooms (the loop used 2400) and 200 validation rooms;
- 20 epochs (the loop used 40);
- 2 CPU threads.

The epoch and the probability threshold are chosen on validation only.
The test split was scored once. For scale, the loop study's full-budget
bar model scores 0.785 on the same test rooms, and the reduced-budget bar
here scores 0.732.

## 3. Results (100 held-out rooms, mean ± SE; paired Δ with SE and z)

| scheme | steps (ms) | learned IoU | back-projection IoU | Δ vs 2 speakers in turn | Δ vs bar |
|---|---|---|---|---|---|
| no audio (training prior map) | 0 | 0.033 ± 0.003 | – | – | – |
| **8-element bar** | 4976 (363) | **0.732 ± 0.021** | 0.180 ± 0.009 | +0.169 ± 0.015 (z 11.1) | – |
| **2 speakers, in turn** | 1244 (91) | **0.564 ± 0.023** | 0.153 ± 0.007 | – | −0.169 ± 0.015 (z −11.1) |
| 2 speakers 28 apart, in turn | 1244 (91) | 0.625 ± 0.021 | 0.182 ± 0.010 | +0.061 ± 0.014 (z 4.3) | −0.108 ± 0.015 (z −7.1) |
| 2 at once, same pulse | 622 (45) | 0.436 ± 0.022 | 0.112 ± 0.008 | −0.127 ± 0.015 (z −8.2) | −0.296 ± 0.019 |
| 2 at once, disjoint bands | 705 (51) | 0.443 ± 0.021 | 0.138 ± 0.007 | −0.121 ± 0.014 (z −8.3) | −0.289 ± 0.018 |
| 2 at once, noise codes (LS) | 1866 (136) | 0.578 ± 0.024 | 0.153 ± 0.007 | +0.014 ± 0.009 (z 1.6) | −0.154 ± 0.014 |
| 2 speakers, K = 2 placements | 2488 (181) | 0.671 ± 0.021 | 0.181 ± 0.009 | +0.107 ± 0.014 (z 7.8) | −0.061 ± 0.012 (z −5.1) |
| **2 speakers, K = 4 placements** | 4976 (363) | **0.795 ± 0.018** | 0.198 ± 0.011 | +0.231 ± 0.017 (z 13.3) | **+0.063 ± 0.011 (z 5.6)** |
| 2 speakers, rigid walls | 1244 (91) | 0.441 ± 0.022 | 0.008 ± 0.002 | −0.123 ± 0.017 (z −7.3) | −0.292 ± 0.020 |
| 2 speakers, rigid walls + image sources | 1244 (91) | 0.426 ± 0.022 | 0.044 ± 0.011 | −0.138 ± 0.020 (z −6.9) | −0.306 ± 0.021 |

Every learned scheme beats its own back-projection (z ≥ 15) and the
no-audio prior (z ≥ 18). The two-speaker device in turn beats
back-projection in 97 % of rooms. The median IoU is 0.57 for two speakers
in turn, 0.77 for the bar and 0.83 for K = 4.

![IoU by scheme](two_speaker_2026_09_26_artifacts/iou_by_scheme.png)

### 3.1 Two speakers in one place: what is lost

- **The small device sees less.** Its 12-cell baseline sees objects mostly
  from straight ahead. It misses objects off to the side and far objects,
  such as the small block in row 1 of \`examples.png\`. It also blurs the
  lateral extent of what it sees. The bar's advantage (+0.17, z = 11,
  better in 93 % of rooms) is an aperture advantage.
- **A wider baseline helps.** With the speakers at the bar's ends (28
  apart) the device reaches 0.63 (+0.06, z = 4.3). That is still well below
  the bar, which also has 6 elements in between.

![examples](two_speaker_2026_09_26_artifacts/examples.png)

The examples figure shows four test rooms. The blue shading is the
learned probability, the orange outline is the true obstacle, and the
black squares are the device elements.

### 3.2 Simultaneous emission: what "different frequencies at once" buys

Separation quality is measured against the in-turn pings, as SDR on the
separated traces (100 test rooms):

| emission (one shot, both speakers) | total steps | separation SDR | IoU |
|---|---|---|---|
| same pulse (sum) | 622 | not separable | 0.436 |
| disjoint bands, filter split | 705 | 33–38 dB per band (probe, 2 rooms) | 0.443 (each speaker has only half the band) |
| noise codes, 622 steps, joint LS | 1244 | 9.5 dB | 0.524 (0.491 with the in-turn model) |
| **noise codes, 1244 steps, joint LS** | 1866 | **31.9 dB** | **0.578** (0.560 with the in-turn model) |
| up/down chirps, 1244 steps, joint LS | 1866 | 8.4 dB | 0.472 |
| noise codes, 1244 steps, naive per-speaker inverse filter | 1866 | 0.8 dB | 0.392 |
| beams (4 same-pulse shots, phase/delay steered), LS per frequency | 2536 | 96 dB | 0.564 (identical to in turn in 100 % of rooms) |

What the table shows:

- **Different frequencies at once (scheme 4).** The bands separate
  cleanly, but each speaker's echoes then carry half the bandwidth. The
  images lose resolution, and the result (0.443) is no better than playing
  the same pulse from both speakers (0.436; Δ +0.007 ± 0.013, z = 0.5).
  It is 0.12 below pinging in turn (z = 8.3).
  - Both one-shot schemes still beat back-projection by a wide margin:
    the U-Net's room prior does much of the work.
  - What they buy is time: 45–51 ms instead of 91 ms, at a cost of 0.12
    IoU.
- **Broadband codes (scheme 5).** They reach the in-turn quality only when
  the recording holds both full responses: T_code + L ≥ 2 L_u, as §1
  predicts.
  - At 1244 steps (total 1866 = 1.5× in turn) they separate at 32 dB and
    match in turn (0.578 against 0.564, z = 1.6). The in-turn model
    scores their images at 0.560, the same information.
  - At 622 steps (total 1244, the same time as in turn) the problem is
    underdetermined. Separation drops to 9.5 dB and IoU to 0.52 (z = −3.3
    against in turn).
  - The code design matters. Up/down chirps overlap in time-frequency near
    their crossing and separate poorly even at 1244 steps (8.4 dB; 15 dB
    at 2488 in the probe). Treating the other speaker as noise (the naive
    inverse filter) leaves 0 dB crosstalk (IoU 0.39).
  - **Codes do not beat pinging in turn on time.**
- **Beams (scheme 6).** Four phase- and delay-steered shots of the same
  pulse separate exactly into the two single-speaker responses (96 dB).
  They give the in-turn IoU exactly, in more time. That is the "at most
  scheme 2" ceiling, reached.

### 3.3 Moving the device (scheme 7)

| placements | steps (ms) | elements | learned IoU | Δ vs K = 1 | Δ vs bar |
|---|---|---|---|---|---|
| K = 1 | 1244 (91) | 4 | 0.564 ± 0.023 | – | −0.169 (z −11.1) |
| K = 2 | 2488 (181) | 7 | 0.671 ± 0.021 | +0.107 (z 7.8) | −0.061 (z −5.1) |
| K = 4 | 4976 (363) | 13 | 0.795 ± 0.018 | +0.231 (z 13.3) | +0.063 (z 5.6) |

At the bar's measurement time (8 pings), the moved two-speaker device beats
the bar. It has a larger aperture (48 cells against 28), though fewer
transfer functions (16 against 64). This is a synthetic aperture: the
aperture sets the resolution more than the element count does. It assumes
the placements are known, for example from inertial tracking as BatMapper
does, and that the room is static while the device moves.

![IoU vs placements](two_speaker_2026_09_26_artifacts/iou_vs_placements.png)

### 3.4 Known reflective walls: a virtual baseline?

The coordinator's follow-up asked about this. Known wall reflections act as
image sources, virtual speakers behind each wall, and could give a small
device a much larger baseline. We repeated scheme 2 with rigid outer walls:

- **Room.** The same grid and scenes, with the outer walls rigid (Neumann)
  instead of CPML. The walls sit half a cell outside the edge cells, so the
  image of row r across the bottom wall is 199 − r. The empty-room
  reference is simulated in the same rigid room, so the residual still
  isolates the obstacles, including their wall-reflected echoes.
- **Result, direct-path migration.** The learned IoU is 0.441 ± 0.022,
  back-projection 0.008.
- **Result, with image-source terms.** The images also migrate along the
  first-order paths via the four walls (source image or mic image; 9 paths
  per trace, in separate channels). The learned IoU is 0.426 ± 0.022,
  back-projection 0.044. Against direct-path migration the difference is
  −0.015 ± 0.014 (z = −1.0).
- **Both are clearly below the anechoic device:** −0.12 to −0.14,
  z ≈ −7.

So here the known walls hurt, and image-source migration did not recover
the loss. The likely reason is that the lossless room keeps the scattered
field reverberating through the whole 622-step window:

- object–wall–object multiples of every order;
- plus the source and mic images, which first-order migration maps onto
  ghosts.

Back-projection collapses (0.008). The U-Net still reaches 0.44 from its
prior and the early echoes, but with 1000 training rooms it does not
extract the extra aperture. In principle the virtual sources carry
information, and the Born-Jacobian analysis in \`information_2026_09_26.md\`
is the place to size it. Exploiting it would need more than first-order
imaging, for example:

- early-time gating;
- partially absorbing walls;
- higher-order image sources;
- or a network on the raw traces with more data.

That is a next step, not a result. **The migration images do need
image-source terms to use the walls at all, and first-order terms alone
are not enough.**

### 3.5 Noise (models trained without noise)

White noise is added to every room recording, with the same absolute
standard deviation for every scheme: σ = 10^(−SNR/20) × the peak residual
of the device's two in-turn pings. The speakers have the same peak level
throughout. The empty-room reference stays clean, as if averaged.

| scheme | steps | clean | 20 dB | 10 dB | Δ vs in turn at 20 dB | Δ vs in turn at 10 dB |
|---|---|---|---|---|---|---|
| 2 speakers, in turn | 1244 | 0.564 | 0.494 ± 0.024 | 0.329 ± 0.019 | – | – |
| disjoint bands | 705 | 0.443 | 0.372 ± 0.020 | 0.183 ± 0.015 | −0.122 (z −6.7) | −0.146 (z −7.4) |
| **noise codes (LS)** | 1866 | 0.578 | **0.557 ± 0.023** | **0.483 ± 0.023** | **+0.063 (z 4.6)** | **+0.154 (z 8.5)** |
| beams (4 shots) | 2536 | 0.564 | 0.546 ± 0.024 | – | +0.052 (z 5.0) | – |
| K = 4 placements | 4976 | 0.795 | 0.769 ± 0.018 | – | +0.275 (z 14.8) | – |
| 8-element bar | 4976 | 0.732 | 0.706 ± 0.022 | 0.544 ± 0.024 | +0.212 (z 12.2) | +0.215 (z 11.8) |

- **This is what simultaneous coded emission buys.**
  - At equal peak level, the 1244-step codes carry about 12 dB more energy
    per speaker than a ping.
  - After least-squares deconvolution they lose only 0.02 IoU at 20 dB and
    0.10 at 10 dB. Pinging in turn loses 0.07 and 0.24.
- **Beams gain from averaging.** Four shots average the noise, as
  repeating the pings would.
- **Disjoint bands suffer most.** They are the least robust scheme: each
  speaker radiates only half the Ricker's energy.
- **Bar noise numbers differ from the loop study's.** The bar's numbers
  here use the device's σ, the same absolute noise floor. They are not the
  loop study's per-scene bar σ, so they differ from its table.

## 4. Verdict

**Two speakers can do it:**

- In one place and pinging in turn, a laptop-like device (2 speakers, 2
  mics, 12 cells apart) reconstructs held-out rooms at IoU 0.56. That is
  about 3.7× back-projection (0.15) and far above the no-audio prior
  (0.03).
- It reaches three quarters of the 8-element bar (0.73), in a quarter of
  the measurement time (91 ms against 363 ms at 5 cm cells).
- It sees objects in front of it well and misses objects off to the side.

**Moving it is the way to array quality.** At four placements (same total
time as the bar) it reaches 0.80, significantly above the bar. Aperture,
real or synthetic, sets the quality.

**Playing different frequencies at once does not extract more
information.** The medium is linear, so the interference (beat) terms are
products of transfer functions the individual tones already measure.
Simultaneous emission trades rather than adds:

| option | effect |
|---|---|
| disjoint bands | 1.8× faster, but half the band per speaker: IoU −0.12 (z 8.3), and the worst noise robustness |
| broadband separable codes | the in-turn information, but only with ≥ 2 listening windows of code (1.5× slower than pings here); with a shorter code, separation fails (9.5 dB) and IoU drops |
| same-frequency beams | exactly the in-turn information (96 dB separation), more slowly |
| codes at equal peak level | the real gain is SNR: +0.06 IoU at 20 dB and +0.15 at 10 dB over pinging in turn |

**In a practical device:**

- ping in turn, or play long coded signals when the room is noisy;
- move the device when possible.

**Known reflective walls** did not give a virtual baseline with
first-order imaging (IoU 0.44 against 0.56 anechoic). That needs
higher-order modelling.

## 5. Literature

- **MIMO radar and sonar with orthogonal waveforms.** With M transmitters
  and N receivers emitting separable (orthogonal) waveforms, the receiver
  splits every transmitter's contribution and forms an M × N virtual
  array. That is the same set of transfer functions that sequential
  pinging measures; orthogonality lets it be measured in one dwell, with a
  waveform long enough to be separable.
  - J. Li, P. Stoica, "MIMO radar with colocated antennas", IEEE Signal
    Processing Magazine 24(5), 106–114 (2007),
    https://doi.org/10.1109/MSP.2007.904812
  - J. Li, P. Stoica (eds.), *MIMO Radar Signal Processing* (Wiley, 2009),
    https://download.e-bookshelf.de/download/0000/5720/01/L-G-0000572001-0002358740.pdf
- **Measuring several loudspeakers at once.** The multiple exponential
  sweep method overlaps and interleaves sweeps from several loudspeakers,
  then separates the impulse responses. It cut HRTF measurement time by
  about 4× at equal SNR, because the sweeps are long compared with the
  responses. The saving comes from overlapping long excitations, as §1
  predicts. The swept-sine method itself is the source of the energy and
  SNR gain of long excitations.
  - P. Majdak, P. Balazs, B. Laback, "Multiple exponential sweep method
    for fast measurement of head-related transfer functions", J. Audio
    Eng. Soc. 55(7/8), 623–637 (2007),
    https://www.aes.org/e-lib/download.cfm?ID=14190
  - P. Dietrich, B. Masiero, M. Vorländer, "On the optimization of the
    multiple exponential sweep method", J. Audio Eng. Soc. (2013),
    https://masiero.fee.unicamp.br/articles/Journal/Dietrich,%20Masiero,%20Vorl%C3%A4nder_2013_On%20the%20Optimization%20of%20the%20Multiple%20Exponential%20Sweep%20Method.pdf
  - A. Farina, "Simultaneous measurement of impulse response and
    distortion with a swept-sine technique", 108th AES Convention (2000),
    https://www.researchgate.net/publication/2456363_Simultaneous_Measurement_of_Impulse_Response_and_Distortion_With_a_Swept-Sine_Technique
- **Room shape from a few echoes, and echoes as virtual sources.**
  First-order echoes recorded by a few microphones determine a convex
  polyhedral room (Dokmanić et al.). Echoes can act as image sources, or
  virtual microphones and speakers, for beamforming (the rake receiver).
  Known walls can serve as virtual anchors for localization.
  - I. Dokmanić, R. Parhizkar, A. Walther, Y. M. Lu, M. Vetterli,
    "Acoustic echoes reveal room shape", PNAS 110(30), 12186–12191 (2013),
    https://doi.org/10.1073/pnas.1221464110
  - I. Dokmanić, R. Scheibler, M. Vetterli, "Raking the cocktail party",
    IEEE J. Sel. Topics Signal Process. (2015),
    https://arxiv.org/abs/1407.5514
  - Multipath-assisted indoor positioning, Leitinger, Witrisal et al.;
    overview and references in
    https://publications.lib.chalmers.se/records/fulltext/203646/local_203646.pdf
- **Active acoustic sensing on phones.** BatMapper builds floor plans with
  a phone's speaker and two mics by ranging to nearby surfaces. It relies
  on the user walking the phone around, adding aperture by motion with
  inertial tracking, as in §3.3.
  - B. Zhou, M. Elbadry, R. Gao, F. Ye, "BatMapper: acoustic sensing based
    indoor floor plan construction using smartphones", MobiSys 2017,
    https://dl.acm.org/doi/10.1145/3081333.3081363
- **Learned echo-to-depth with one emitter and two ears.** These systems
  rely on a strong learned prior. That matches our finding: with a small
  aperture, the prior carries much of the reconstruction.
  - J. H. Christensen, S. Hornauer, S. X. Yu, "BatVision: learning to see
    3D spatial layout with two ears", ICRA 2020,
    https://arxiv.org/abs/1912.07011
  - R. Gao, C. Chen, Z. Al-Halah, C. Schissler, K. Grauman, "VisualEchoes:
    spatial image representation learning through echolocation", ECCV
    2020, https://vision.cs.utexas.edu/projects/visualEchoes/

## 6. Limitations

- **Data and training.** These are noiseless 2D simulations of one room
  family: rectangles and partitions in an anechoic 100² grid. The training
  budget is reduced (1000 rooms, 20 epochs), so all absolute IoUs are
  lower than a full-budget model would give. The bar loses 0.05 against
  the loop study. The comparisons are paired, on the same rooms and with
  the same recipe.
- **Code length.** It was chosen from a two-room probe (SDR at 311, 622,
  1244 and 2488 steps; \`data/two_speaker/probe.json\`), not tuned on test
  data.
- **Speakers and mics** are ideal point sources and receivers. Real laptop
  speakers are band-limited and directional, and the direct path is much
  stronger. The residual assumes a clean empty-room reference.
- **Moved placements** are assumed exactly known, and the room is assumed
  static.
- **Rigid-wall rooms** use the same listening window as the anechoic room.
  Early-time gating and higher-order imaging were not explored.

## 7. Shipped (for a later "Two speakers" mode)

- **\`web/src/twospeaker/\`** covers the device geometry, emission schedules
  and codes (\`device.ts\`), the separation (\`separation.ts\`), and the
  sensing pipeline (\`sensing.ts\`). The pipeline runs recording with the
  loop's engine, then residual, separation, \`pairMigration\` and the U-Net.
  - \`new TwoSpeakerSensor(await LoopUNet.load('models/two_speaker_seq'))\`
    then \`.sense(room)\` returns the images, the estimate, the probability
    and the measurement time.
  - The empty-room reference is cached per room parameters.
- **Models.** \`web/public/models/two_speaker_seq.{json,bin}\` (one
  placement) and \`two_speaker_seq_k4.{json,bin}\` (four placements). They
  are in \`loop-unet-v1\` format, plus \`scheme\` and \`device\`, at 467 KiB
  each.
- **Parity.** \`web/tests/unit/twoSpeakerSensing.test.ts\` re-simulates
  three test rooms per model in TypeScript. Against PyTorch, the features
  agree to 1e-6 and the logits to ≤ 9e-5, at a logit scale of 13–24. The
  images match the training data.
- **Hook for the Echo vision page.** No edits were made to
  \`web/src/loop/*\`, \`web/src/echo/*\` or \`EchoVision.tsx\`. The page can
  offer "Two speakers" by constructing a \`TwoSpeakerSensor\` and drawing:
  - \`result.probability\` / \`result.estimate\`;
  - \`result.elements\` (4 or 13 cells on row 86) in place of the bar.

  \`LoopUNet.estimate(images, elements)\` from \`learnedSensing.ts\` is reused
  unchanged. The one-placement model needs 2 simulations of 622 steps per
  estimate; the K = 4 model needs 5 unique speaker positions.

## Reproduce

\`\`\`bash
cd web
npx vite build --ssr scripts/two_speaker_data.ts --outDir .render/ts-data --emptyOutDir --logLevel warn
J="node .render/ts-data/two_speaker_data.js"
MAIN=seq,wide_seq,sum,band,code,seq_k2,seq_k4,seq_rigid,seq_rigid_img
$J --split val --count 200 --schemes $MAIN
$J --split test --count 100 --schemes $MAIN,beams,code622,chirp,code_mf,bar8
$J --split train --count 1000 --schemes $MAIN
$J --split test --count 100 --snr 20 --name test_snr20 --schemes seq,band,code,seq_k4,beams,bar8
$J --split test --count 100 --snr 10 --name test_snr10 --schemes seq,band,code,bar8
# separation probe (SDR vs code length):
npx vite build --ssr scripts/two_speaker_probe.ts --outDir .render/ts-probe --emptyOutDir --logLevel warn
node .render/ts-probe/two_speaker_probe.js --count 2 --codes 311,622,1244,2488
cd ..
for s in bar8 seq code band sum seq_k4 seq_k2 wide_seq seq_rigid_img seq_rigid; do
  uv run python scripts/train_loop_sensing.py train --scheme $s --limit 1000 --val-limit 200 \\
    --epochs 20 --out checkpoints/two_speaker/$s
done
uv run python scripts/two_speaker_report.py
for s in seq seq_k4; do
  uv run python scripts/train_loop_sensing.py export checkpoints/two_speaker/$s/best.pt \\
    --out web/public/models/two_speaker_$s --fixtures
done
\`\`\`

Wall time on the shared 4-core container (at most 2 threads, one Node
process at a time):

- data: 4.6 s per room for the nine main schemes, about 2 h in total;
- training: 7.5 min per model at 2 threads (12 min at 1 thread);
- evaluation: 2 min.

The data is in \`data/two_speaker/\` (gitignored, 2.8 GB).
`;export{e as default};
//# sourceMappingURL=two_speaker_2026_09_26-PldMJD2K.js.map