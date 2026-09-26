var e=`# Is the room map attainable with two speakers? Information limits of small-device sensing (2026-09-26)

**Question (project owner).** "Can we do it without an array, and just two
speakers? ... can we have a virtual baseline? can we do the calculations to
see if the information is attainable?"

This study does the calculation. It does not train any networks. It
linearises the engine's forward map around the empty room and whitens it by
the measurement noise. It then counts how many independent features of the
room map each configuration can measure better than a weak prior, and at
what resolution. The numbers bound what *any* estimator can extract from
the recordings, a trained network included, without leaning on its shape
prior.

## Verdict

**From one fixed position in an anechoic room, two speakers do not deliver
a room map.** In the loop's band, the pair (in turn, 4 traces) measures
about 200 modes at 20 dB SNR and 260 at 30 dB. But almost all of them are
*range along ellipses*. Only 2–3 % of the 2 × 2-cell pixels are resolved
(diag R ≥ 0.5), against 36 % and 73 % for the 8-element bar. A point
object's image is an arc about 14 cells (35 cm) wide at 1 m. Raising the
SNR barely helps: the pair's Jacobian has a hard rank of about 320 of 1216
pixels, and even at 60 dB only 9 % of the pixels are resolved. Widening the
pair from 12 to 28 cells (30 → 70 cm) does not help either: two sparse
elements give a longer but gappy virtual aperture, with the same mode count
and larger sidelobes. Whatever a network draws beyond this comes from its
prior.

**A virtual baseline does make the information attainable.** In order of
strength:

1. **Motion (synthetic aperture).** The same device at K = 4 placements
   along the wall uses 8 pings, as the bar does. It matches or beats the
   bar: 622 / 814 DOF at 20 / 30 dB, with 45 % / 85 % of pixels resolved
   (bar: 592 / 770 and 36 % / 73 %), from 16 traces instead of 64.
   K = 8 resolves the whole region at 30 dB. At the same emitted energy,
   the pair at one position resolves only 3–5 %, so the gain is
   geometric. This needs the placements known to a small fraction of a
   wavelength (see \`pose_robust_2026_09_25.md\`).
2. **Bandwidth.** At equal energy, doubling the Ricker's peak frequency
   (f0 0.08 → 0.16) gives 2.5 × the DOF (500 / 584) and 23 % / 38 % of
   pixels resolved. A real laptop has far more bandwidth than the grid
   band.
3. **Reflective walls, if they are known and modelled.** With rigid walls
   the fixed pair gets 409 / 485 DOF, and 14 % / 26 % of pixels resolved.
   Plaster gives 16 % and wood 7 % at 30 dB. The walls act as image
   speakers far outside the room, and they also raise the echo energy by
   reverberant build-up (+17 dB for rigid walls). At matched energy, the
   anechoic pair gets 290 / 326 DOF and 5 % / 6 % of pixels, so about
   half of the gain is geometric. At 50 dB the rigid-room pair resolves
   93 % of the room. This holds only if the wall geometry, the materials
   and the reverberation are all in the model: the linear model explains
   83 % of a rigid pixel's echo in the rigid room, against 96 % in the
   anechoic one. It is the most model-dependent lever.

**Simultaneous multi-frequency emission buys time, not information.** A
simultaneous emission can at best reproduce the in-turn information, when
the codes are perfectly separable. It can never add a mode the speakers in
turn miss:

- **Same pulse at once (only the sum observed).** DOF drops by 40 %
  (119 vs 197 at 20 dB) and the rank halves (170 vs 316).
- **Steered beams.** Four steered shots restore the rank (309) but use 2 ×
  the time and 4 × the energy. At equal energy they give exactly the
  in-turn DOF (193 vs 197).
- **Disjoint half bands.** Each speaker covers only half the band, so the
  virtual array is only half populated in frequency. DOF falls by a third
  (132 vs 197). At equal time (repeat and average) the loss is still
  30 % (139).
- **Up/down chirps at once.** Measured against the same chirps played in
  turn, the cross-talk costs 9 % of the DOF for the 622-step codes
  (time-bandwidth product ≈ 65) and 33 % for 128-step codes (TB ≈ 13).
  The long code saves no time: 622 code steps plus 622 listening steps
  equal two Ricker pings. The short code saves 40 % of the time and
  loses a third of the information.

**Bottom line for the owner.** Two speakers plus two mics can map a room
only with a virtual baseline. The robust route is motion: 4 or more known
placements along the wall, or a laptop turned or carried, with pose
refinement. It works best with the widest band the hardware allows. Wall
echoes can double the information of a fixed device, but only once the
room's walls themselves are estimated and modelled. Emitting both speakers
at once with separable codes saves measurement time. It adds no
information.

## Summary table

Pixels are 2 × 2 cells over rows 12–75 and cols 12–87 (1216 pixels). The
prior sd is τ = 0.5 per pixel occupancy. SNR is the per-sample white-noise
level relative to the peak echo of a reference 8 × 11-cell rigid block
mid-room (0.0726 for the pair, the same σ for every configuration). Every
speaker emission has the loop Ricker's energy (f0 = 0.08, amplitude 1).

- **DOF**: modes measured better than the prior.
- **res.**: fraction of pixels with diag R ≥ 0.5.
- **conc.**: median fraction of a PSF's energy within 5 cells of its pixel.
- **PSF mid**: energy-weighted RMS extent (cross-range / range, in cells)
  of the PSF of a point 41 cells (1 m) in front of the device. It counts
  arcs and sidelobes; the floor is about 1 cell.

| config | shots | traces | rank | DOF 20 dB | DOF 30 dB | res. 20 | res. 30 | conc. 30 | PSF mid 20 dB | PSF mid 30 dB |
|---|---|---|---|---|---|---|---|---|---|---|
| (a) bar8, in turn | 8 | 64 | 848 | **592** | **770** | 0.36 | 0.73 | 0.94 | 3.4 / 1.7 | 3.6 / 1.3 |
| (b) pair 12 apart, in turn (\`seq\`) | 2 | 4 | 316 | **197** | **259** | 0.02 | 0.03 | 0.53 | 14.2 / 7.0 | 13.8 / 6.9 |
| (b) pair 28 apart, in turn (\`wide_seq\`) | 2 | 4 | 323 | 204 | 263 | 0.02 | 0.03 | 0.43 | 17.4 / 9.3 | 16.9 / 9.2 |
| (c) half bands at once (\`band\`) | 1 | 2 | 184 | 132 | 157 | 0.01 | 0.02 | 0.33 | 17.3 / 8.1 | 16.4 / 7.6 |
| (c) half bands, drives as trained (\`band_raw\`) | 1 | 2 | 184 | 112 | 147 | 0.01 | 0.01 | 0.32 | 18.1 / 8.5 | 16.7 / 7.8 |
| (c) up/down chirps at once, 622 (\`code\`) | 1 | 2 | 396 | 256 | 302 | 0.04 | 0.05 | 0.51 | 14.1 / 7.5 | 14.4 / 7.5 |
| (c) the same chirps in turn (\`chirp_seq\`) | 2 | 4 | 420 | 280 | 325 | 0.04 | 0.06 | 0.54 | 14.2 / 7.3 | 14.4 / 7.4 |
| (c) up/down chirps at once, 128 (\`code128\`) | 1 | 2 | 443 | 290 | 353 | 0.05 | 0.09 | 0.52 | 15.8 / 10.4 | 15.1 / 9.6 |
| (c) the same chirps in turn (\`chirp128_seq\`) | 2 | 4 | 637 | 430 | 515 | 0.11 | 0.23 | 0.71 | 13.6 / 6.2 | 12.5 / 5.8 |
| (d) same pulse at once (\`sum\`) | 1 | 2 | 170 | 119 | 147 | 0.01 | 0.01 | 0.34 | 15.1 / 7.4 | 14.9 / 7.5 |
| (d) 4 steered shots (\`beams\`) | 4 | 8 | 309 | 238 | 284 | 0.03 | 0.05 | 0.53 | 13.8 / 6.7 | 14.0 / 7.0 |
| (e) K = 2 placements | 4 | 8 | 585 | 357 | 479 | 0.07 | 0.16 | 0.75 | 10.8 / 6.0 | 10.9 / 5.6 |
| (e) K = 4 placements | 8 | 16 | 993 | **622** | **814** | 0.45 | 0.85 | 0.92 | 6.3 / 3.3 | 7.2 / 2.6 |
| (e) K = 8 placements | 16 | 32 | 1123 | 772 | 972 | 0.76 | 1.00 | 0.98 | 1.9 / 2.0 | 1.1 / 1.2 |
| (f) pair, wood walls (R ≈ 0.54) | 2 | 4 | 522 | 314 | 427 | 0.02 | 0.07 | 0.50 | 15.5 / 11.9 | 14.5 / 10.8 |
| (f) pair, plaster walls (R ≈ 0.82) | 2 | 4 | 514 | 366 | 465 | 0.06 | 0.16 | 0.50 | 15.5 / 11.7 | 14.5 / 10.6 |
| (f) pair, rigid walls | 2 | 4 | 465 | **409** | **485** | 0.14 | 0.26 | 0.52 | 15.2 / 11.3 | 14.3 / 10.4 |
| (f) wide pair, rigid walls | 2 | 4 | 475 | 413 | 486 | 0.16 | 0.27 | 0.52 | 17.0 / 12.1 | 16.1 / 11.4 |
| (f) bar8, rigid walls | 8 | 64 | 1205 | 1204 | 1216 | 1.00 | 1.00 | 1.00 | 1.0 / 0.9 | 0.4 / 0.4 |
| (g) pair, f0 = 0.04 | 2 | 4 | 141 | 98 | 119 | 0.00 | 0.01 | 0.39 | 13.2 / 6.8 | 14.0 / 7.3 |
| (g) pair, f0 = 0.12 | 2 | 4 | 509 | 396 | 436 | 0.08 | 0.11 | 0.62 | 14.2 / 6.6 | 13.9 / 6.4 |
| (g) pair, f0 = 0.16 | 2 | 4 | 674 | 500 | 584 | 0.23 | 0.38 | 0.77 | 12.2 / 5.8 | 10.7 / 5.1 |

"rank" counts singular values above 10⁻⁶ of the largest: the modes the
configuration could reach at unlimited SNR. Every number is in
\`information_2026_09_26_artifacts/results.json\`, which also holds the
40 and 10 dB results, the equal-time and equal-energy DOF, and the
probe PSFs.

## 1. Method

### 1.1 Forward model: discrete Born sensitivity

The engine advances

$$p^{n+1} = 2p^n - p^{n-1} + C\\,\\mathcal L p^n + q^n,$$

with $C = (c\\Delta t/\\Delta x)^2$ per cell, $\\mathcal L$ the 5-point
Laplacian over open faces, and $q^n$ the drive samples added to $p^{n+1}$.
A perturbation of the scheme at a cell acts as an extra injection there,
which propagates with the engine's Green's function $G$. The discrete
scheme is reciprocal, so one impulse simulation from each mic gives the
receiver side $G_{\\mathbf m}(\\mathbf x)$ for every cell. One simulation per
shot gives the incident field $P$ with all of that shot's drives playing
together. Two kernels come out:

- **monopole** (compressibility):
  $\\delta y_m[n] = \\sum_j G_{\\mathbf m}[n-j](\\mathbf x)\\,(C\\mathcal L P)[j-1](\\mathbf x)$
  per unit $\\delta C/C$;
- **dipole** (a closed face, the density term):
  $C\\,(G_{\\mathbf m}(\\mathbf a)-G_{\\mathbf m}(\\mathbf b)) * (P(\\mathbf a)-P(\\mathbf b))$
  per face.

A rigid 2 × 2 pixel closes its 8 boundary faces. Its echo is modelled as
$aK_M + bK_D$, where $K_M$ sums the monopole kernels of the pixel's cells
and $K_D$ sums the dipole kernels of its boundary faces. $a$ and $b$ are
fitted to exact engine runs with that pixel made rigid. Every trace is
sampled as the loop records it: every step, from the start of the shot, for
622 steps after the last drive sample. The unknown is pixel occupancy
(0 = air, 1 = rigid), so a column of $J$ is the echo of one occupied pixel.

**Region.** The loop places objects over rows 8–74 and cols 8–91. The
outer 4-cell rim of that region lies inside the 12-cell CPML, where the
operator is stretched and an object is half absorbed. The analysis
therefore covers the room interior inside the CPML, rows 12–75 and
cols 12–87, in 2 × 2-cell pixels (32 × 38 = 1216). The exact check (§1.3)
uses 4 × 4 pixels.

**Configurations.** The geometry and emissions mirror the training study
(\`web/src/twospeaker/device.ts\`):

- **Ricker.** f0 = 0.08 cycles per unit time, i.e. 0.04 per step (the
  brief's "0.05" is the loop's control tone; \`senseRoom\` pings at 0.08).
- **Bar.** Row 86, cols 31–59 every 4 cells.
- **Pair.** Speakers at cols 39 and 51 with mics at 41 and 49 (\`seq\`), or
  speakers at 31 and 59 with mics at 33 and 57 (\`wide_seq\`).
- **Placements.** K = 2 at centres 38 and 52. K = 4 at 24, 38, 52 and 66.
  K = 8 (added here) at 18, 26, …, 74. Each placement records only with
  its own mics.
- **Emission schemes.** Half-band pulses, up/down chirps over 0.01–0.22
  and beam delays exactly as in \`device.ts\`, except that each speaker
  emission is scaled to the Ricker's energy. \`band_raw\` keeps the trained
  drives, which carry 31 % of the energy.
- **Reflective rooms.** The outer 12 cells are filled with a material:
  rigid, plaster (β = 0.1) or wood (β = 0.3). The air interior and the
  device stay exactly as in the CPML room.

### 1.2 Noise and information measures

White noise has standard deviation σ = 10^(−SNR/20) × A_ref per sample.
A_ref = 0.0726 is the peak residual of an 8 × 11-cell rigid block at rows
37–44 and cols 45–55, as heard by the pair. The bar hears the same block
at 0.0726. The loop report uses the same convention, but per scene. σ is
the same for every configuration, so any extra echo energy (for example
from reverberation) counts as signal. As in the loop study, the empty-room
reference is taken as noise-free, as if averaged.

With a Gaussian prior of sd τ = 0.5 per pixel (weak: a Bernoulli(½)
variance), $F = J^\\top J/\\sigma^2$ and $x_i = \\lambda_i\\tau^2/\\sigma^2$:

- **Recoverable DOF**: $\\#\\{x_i \\ge 1\\}$, the modes the data pin down
  better than the prior.
- **Degrees of freedom for signal**: $\\mathrm{tr}R = \\sum x_i/(1+x_i)$
  (Rodgers 2000). It tracks the DOF closely and is in \`results.json\`.
- **Posterior sd** (CRB with prior): $\\sqrt{\\mathrm{diag}(F + I/\\tau^2)^{-1}}$.
- **Resolution matrix** $R = (F + I/\\tau^2)^{-1}F$. Its diagonal is the
  resolvability, and column $j$ is the PSF of pixel $j$ (Backus & Gilbert
  1968; Fichtner & Trampert 2011).

The PSF size is reported three ways. The **RMS extent** along the radial
and tangential directions, relative to the device centre, is weighted by
$R_{ij}^2$, so it counts arcs, sidelobes and ghosts. The **concentration**
is the share of PSF energy within 5 cells. A half-maximum width is also
computed, but it is not used. Arc-shaped PSFs change sign along the arc, so
the half-maximum width of the central lobe stays near one pixel (about
2 cells) everywhere and says nothing about the smearing.

### 1.3 Validation

- **Born monopole vs the engine.** A central difference of the engine with
  the speed map perturbed at one cell (±1 %) matches the monopole kernel
  to a relative error of 7 × 10⁻⁴ in the CPML room and 1.7 × 10⁻³ in the
  rigid room. The kernel is the engine's exact derivative. A fast version
  of this check is a unit test.
- **Rigid-pixel fit.** Over 24 random pixels, $aK_M + bK_D$ explains 96 %
  of the exact echo energy of an occupied 2 × 2 pixel in the CPML room,
  with a = 1.60 and b = 2.10. Fits on the wide pair (a = 1.62, b = 2.04)
  and the bar (a = 1.67, b = 2.00) agree, and the monopole alone explains
  91 %. The fit on the pair is used everywhere. In the rigid room the same
  form explains 83 % (a = 1.23, b = 2.54): the rest is multiple scattering
  between the pixel and the walls.
- **Exact Jacobian (4 × 4 pixels, 304).** One engine run per pixel and
  shot, with that pixel rigid, gives the exact single-pixel Jacobian.
  Figure \`exact_vs_born.png\` shows its spectra:

  | config | exact DOF 20 / 30 dB | Born, fitted at 4 × 4 | Born, 2 × 2 fit reused |
  |---|---|---|---|
  | pair | 171 / 195 | 175 / 200 | 175 / 197 |
  | wide pair | 181 / 208 | 185 / 212 | 188 / 211 |
  | bar | 298 / 302 | 294 / 302 | 298 / 302 |

  The Born DOF agrees with the exact one to within 3 %.

![exact vs Born](information_2026_09_26_artifacts/exact_vs_born.png)

## 2. Results

![spectra](information_2026_09_26_artifacts/spectra.png)

The spectra are normalised so that a mode is recoverable at SNR s where it
lies above the s dB line. The pair's spectrum falls much faster than the
bar's and hits a floor near mode 370, which is its rank. Placements, walls
and bandwidth all lift and extend the pair's tail.

![DOF and resolved fraction vs SNR](information_2026_09_26_artifacts/dof_vs_snr.png)

### (a) vs (b): the bar against the pair

The bar has 64 traces, 15 distinct virtual (pair-midpoint) elements and a
28-cell aperture. The pair has 4 traces and 4 virtual elements within
10 cells. The pair measures a third of the bar's modes, and they are the
wrong kind for a map.

Its PSF (\`psf_30dB.png\`) is the intersection of four near-identical
ellipses: sharp in range (about λ/4) but smeared along the arc. The RMS
extent at 1 m is 14 cells (35 cm) cross-range, against 3.6 for the bar.
The posterior sd stays at the prior almost everywhere (median 0.45 of 0.5).

The wide pair lengthens the baseline but leaves a gap in the middle of the
virtual aperture (elements at 32, 44–46 and 58). Its PSF core is narrower
on axis, but the sidelobes are larger (concentration 0.43 vs 0.53) and the
mode count is unchanged. **Baseline without fill is not aperture.**

![PSFs](information_2026_09_26_artifacts/psf_30dB.png)

![maps 30 dB](information_2026_09_26_artifacts/maps_30dB.png)

\`maps_20dB.png\` shows the same maps at 20 dB.

### (c) Separable simultaneous emission: the MIMO virtual array

Per frequency, a scheme's Fisher matrix is

$$F = \\sum_f \\sum_m K_m(f)^H M(f) K_m(f), \\qquad M(f) = \\sum_k \\bar w_k(f)\\, w_k(f)^\\top,$$

with $w_k(f)$ the speaker drive spectra of shot $k$ and $K_m(f)$ the
per-speaker sensitivities (\`information.emission_gram\`). Speakers in turn
give $M = |r|^2 I$: the full virtual array of every speaker–mic pair (Li &
Stoica 2007).

A single simultaneous shot has rank-1 $M(f)$ at every frequency. It
recovers the in-turn information only if the cross terms
$\\bar w_A w_B K_A^H K_B$ cancel when summed over frequency. That happens
exactly when the codes' cross-correlation vanishes at every lag in the
scene's delay spread; the unit test uses a time-multiplexed pair to show
the Gram matrices are then identical. With finite codes it is approximate:

- **Up/down chirps, 622 steps.** The Gram matrix differs from the same
  chirps played in turn by 51 % (Frobenius), yet the trace agrees to
  1 %. That costs 9 % of the DOF (256 vs 280 at 20 dB).
- **Up/down chirps, 128 steps.** The Frobenius difference is 77 %, and 33 %
  of the DOF is lost (290 vs 430).

The chirps themselves carry more information than the Ricker at equal
energy: they have a flat spectrum up to 0.22, which is a bandwidth effect,
see (g). The like-for-like comparison is therefore always code against
the same chirps in turn.

**Disjoint half bands** make $M(f)$ diagonal, with a single non-zero entry
per frequency: each speaker's virtual elements exist only in its half of
the band. That costs a third of the DOF and 40 % of the rank (184 vs 316).
The band scheme halves the measurement time. Spending that time on a
second averaged shot (+2.4 dB) still leaves 139 DOF against the in-turn
pair's 197.

### (d) Summed and steered emission cannot add information

A summed or steered emission's rows are linear combinations of the in-turn
rows (unit tests: the rank never exceeds the in-turn rank, and the summed
Fisher matrix is at most 2 × the in-turn one, the coherent array gain).

- **Same pulse at once.** One shot observes $K_A + K_B$ only, and the rank
  drops from 316 to 170.
- **Four steered shots.** Their emission Gram $M$ has eigenvalues
  $|r|^2(4 \\pm 2\\cos\\omega\\tau)$ (a unit test checks the analogous
  3-shot case). They restore the rank, 309 against 316, but spend 4 × the
  energy and 2 × the time. At equal emitted energy they deliver 193 DOF
  against 197 in turn, i.e. the same information.

### (e) Synthetic aperture: the virtual baseline from motion

Each placement adds four new virtual elements, so the virtual aperture
grows with K while each placement stays self-contained.

- **K = 4.** The virtual elements span cols 19–71. That beats the bar,
  whose single placement spans 31–59, even though no cross-placement
  pairs exist. It resolves 85 % of the pixels at 30 dB with a compact PSF
  (concentration 0.92).
- **K = 8.** It resolves the whole region at 30 dB, and 76 % at 20 dB.
- **Energy check.** The pair at one position with K = 4's total energy
  (+5.7 dB) gets only 238 / 285 DOF and resolves 3 % / 5 % of pixels. The
  gain comes from geometry.

The catch is pose: the travel times must be known to a fraction of a
period. The loop pulse at dx = 2.5 cm has a period of 0.9 ms, i.e. 31 cm
of path. At 20 kHz the period is 50 µs, i.e. 1.7 cm of path.
\`pose_robust_2026_09_25.md\` shows that a 2.5 cm / 2° rigid pose error costs
most of the gain unless poses are refined against the model.

### (f) Reflective walls: a virtual baseline from image sources

First-order images of the two speakers lie across the side walls (cols
−16 / −28 and 124 / 136) and the far wall (row −63): virtual speakers 1.5
to 3.7 m away. Later reflections add more (Allen & Berkley 1979), which
gives the "virtual baseline" of \`schematic.png\`. The fixed pair's Jacobian
gains rank (316 → 465) and DOF (197 → 409 at 20 dB). Its PSF core becomes
compact at every angle: the half-maximum width is one pixel, and R_jj at
1 m rises from 0.15 to 0.38. The PSF energy, however, spreads as a
room-wide speckle of weak sidelobes, so the concentration stays at 0.52.

Part of the gain is energy: a lossless room re-radiates each echo many
times within the window (Gram trace × 51, +17 dB). At matched energy:

| comparison (20 / 30 dB) | DOF | resolved | concentration (30 dB) |
|---|---|---|---|
| pair, anechoic, at the rigid room's echo energy | 290 / 326 | 0.05 / 0.06 | 0.55 |
| pair, rigid walls | 409 / 485 | 0.14 / 0.26 | 0.52 |
| pair, anechoic, at the plaster room's echo energy | 261 / 303 | 0.03 / 0.05 | 0.53 |
| pair, plaster walls | 366 / 465 | 0.06 / 0.16 | 0.50 |
| pair, anechoic, at the wood room's echo energy | 221 / 275 | 0.02 / 0.04 | 0.53 |
| pair, wood walls | 314 / 427 | 0.02 / 0.07 | 0.50 |
| bar, anechoic, at the rigid room's echo energy | 868 / 959 | 0.84 / 0.90 | 0.96 |
| bar, rigid walls | 1204 / 1216 | 1.00 / 1.00 | 1.00 |

So the walls add genuine geometric information on top of the energy.
**Three conditions must hold:**

- the walls must be known (Dokmanić et al. 2013 recover a convex room's
  walls from first-order echoes, and EchoSLAM does it with a co-located
  device);
- the model must include the reverberation (the linear model explains
  only 83 % of a rigid pixel's echo here);
- the walls must be reflective: with wood the DOF still rise, 314 against 221 at matched energy, but the resolved fraction barely moves (7 % against 4 % at 30 dB).

### (g) Bandwidth

All at equal energy per ping:

| f0 | DOF 20 / 30 dB | resolved at 30 dB |
|---|---|---|
| 0.04 | 98 / 119 | 1 % |
| 0.08 | 197 / 259 | 3 % |
| 0.12 | 396 / 436 | 11 % |
| 0.16 | 500 / 584 | 38 % |

The DOF grows roughly in proportion to f0, and the range PSF shrinks with
the wavelength. The pair at f0 = 0.16 is the best single-position anechoic
configuration: 23 % / 38 % of pixels resolved at 20 / 30 dB. f0 = 0.16 has
only about 6 cells per peak wavelength, past the grid's 8-cell rule, so in
the simulation this is qualitative. For real hardware it is the natural
lever (§3).

## 3. What the numbers mean for a real laptop

At \`LAPTOP_ROOM\` scale (dx = 2.5 cm, \`simulation/units.py\`):

- the Ricker peaks at 1.1 kHz, with a band of about 0.27–2.7 kHz;
- the narrow pair is 30 cm apart, a typical laptop speaker spacing, and
  the wide one 70 cm;
- the interior is 1.9 m square and the window lasts 23 ms.

The simulated pair is therefore only about one wavelength wide. A real
laptop playing near-ultrasound (17–23 kHz, λ ≈ 1.7 cm) or the full audible
band has the same 30 cm spacing at 15–20 λ. The single-reflector CRLB
(\`crlb.py\`, 2 m, 20 dB) gives:

| band | range resolution c/2B | bearing CRLB | lateral CRLB at 2 m |
|---|---|---|---|
| near-ultrasound, 17–23 kHz | 2.9 cm | 0.85° | 3 cm |
| audible, 0.3–20 kHz | 0.9 cm | 0.26° | 0.9 cm |
| sim band | 7 cm | 2.1° | 7 cm |

These bounds describe **localising one isolated echo**. This study's
point is different: **a map** has many more unknowns, and they all compete
for the same few views. Scaling the study to the real band:

- **Range resolution** scales with c/2B and improves by 10–40 ×.
- **The pair's arc problem does not go away.** Four bistatic pairs are
  still four look directions per placement. In the far field, their
  virtual elements span the 30 cm, 17 λ baseline sparsely, so the
  cross-range response carries grating-lobe ambiguity at spacing
  λR/d ≈ 11 cm at 2 m (a 30 cm pair at 20 kHz).
- **The number of unknowns grows** as (room area)/(λ/2)², by 250–1600 ×.
  The fraction of the map that 4 traces can pin down, already 2–3 % here,
  does not improve by itself.

What does transfer is the ranking of levers:

- motion, including a laptop turned or carried to a few known spots;
- band (the widest the speakers allow; laptop speakers roll off above
  about 16–18 kHz, and air absorption at 20–40 kHz is 0.3–1.3 dB/m);
- walls, once estimated;
- SNR.

A single ping's echo SNR in a room is typically 10–30 dB. N averaged pings
add 10 log₁₀ N dB, and a chirp of duration T adds its time-bandwidth gain.
Real-room numbers need the Lab page's measurements (plan phase 9).

## 4. Comparison with the trained networks

The parallel study's report (\`tests/reports/two_speaker_2026_09_26.md\`)
reads "DRAFT — results pending" as this report is written. Only its bar8
reference network was in training (\`checkpoints/two_speaker/bar8\`). There
are no trained two-speaker IoUs to compare yet.

The loop U-Net (\`loop_sensing_2026_09_25.md\`) tracks the bar's resolved
fraction, which is encouraging:

| echo SNR | U-Net IoU | bar pixels resolved |
|---|---|---|
| 30 dB | 0.78 | 73 % |
| 20 dB | 0.71 | 36 % |
| 10 dB | 0.36 | 16 % |

A strong shape prior therefore fills in a good deal beyond the linear
resolution, but it collapses once the resolved fraction falls to about
15 %. On that reading, these are the **predictions** for the parallel
study:

- \`seq\`, \`wide_seq\`, \`sum\`, \`band\`, \`code\` and \`beams\`, at 1–5 % of
  pixels resolved, should reach an IoU well below the bar's. What they do
  reach will be largely the prior. The ranking should be \`code\` ≈ \`beams\`
  ≈ \`seq\` ≈ \`wide_seq\` > \`band\` ≈ \`sum\`.
- \`seq_k4\` should match the bar within noise. \`seq_k2\` should sit in
  between.
- The rigid-wall \`seq\` should beat the anechoic \`seq\`, provided the
  network is trained in the same rigid room, i.e. it learns the known
  walls.

## 5. Caveats

- **Linearised about the empty room.** No shadowing and no multiple
  scattering between objects. The Born map is optimistic about faces
  hidden behind other objects and about the interior of large objects. It
  is pessimistic nowhere that matters here: a network cannot beat the data
  except through its prior.
- **2D, lossless air, and known poses and walls.** The empty-room
  reference is noise-free, and pixels are independent under the prior. A
  shape prior (what the U-Net learns) makes far fewer DOF necessary. The
  resolved fraction is therefore a conservative measure of "map
  attainable", while the DOF count is optimistic about how useful each
  mode is.
- **SNR is relative to one reference block's peak echo.** Small or distant
  objects have weaker echoes: read their numbers at a lower SNR on the
  curves.
- **2 × 2-cell pixels quantise the range PSF.** The range resolution of
  every configuration is at or below one pixel, about λ/4. The
  cross-range extent is the discriminating quantity.

## 6. Literature

- J. Li and P. Stoica, "MIMO radar with colocated antennas", *IEEE Signal
  Processing Magazine* 24(5), 106–114, 2007: virtual array =
  transmit ⊕ receive positions; waveform orthogonality.
  https://en.wikipedia.org/wiki/MIMO_radar (overview and references).
- O. M. Bucci and G. Franceschetti, "On the degrees of freedom of
  scattered fields", *IEEE Trans. Antennas Propag.* 37(7), 918–926, 1989:
  the DOF equal the Nyquist number of the field's spatial bandwidth over
  the observation domain.
  https://www.researchgate.net/publication/3008028_On_the_Degrees_of_Freedom_of_Scattered_Fields
- A. J. Devaney, "A filtered backpropagation algorithm for diffraction
  tomography", *Ultrasonic Imaging* 4, 336–350, 1982: Born inversion and
  k-space coverage.
  https://journals.sagepub.com/doi/abs/10.1177/016173468200400404 ; and
  Devaney & Beylkin 1984, arbitrary transmitter and receiver surfaces,
  https://doi.org/10.1177/016173468400600207
- G. Backus and F. Gilbert, "The resolving power of gross Earth data",
  *Geophys. J. R. Astr. Soc.* 16, 169–205, 1968: resolution (averaging)
  kernels. https://academic.oup.com/gji/article/16/2/169/623631
- A. Fichtner and J. Trampert, "Resolution analysis in full waveform
  inversion", *GJI* 187, 1604–1624, 2011: Hessian columns as
  point-spread functions. https://academic.oup.com/gji/article/187/3/1604/616815
- C. D. Rodgers, *Inverse Methods for Atmospheric Sounding*, World
  Scientific, 2000: degrees of freedom for signal and information content.
  https://searchworks.stanford.edu/view/10557747
- I. Dokmanić, R. Parhizkar, A. Walther, Y. M. Lu and M. Vetterli,
  "Acoustic echoes reveal room shape", *PNAS* 110(30), 12186–12191, 2013:
  first-order echoes determine a convex room.
  https://www.pnas.org/doi/abs/10.1073/pnas.1221464110
- M. Krekovic, I. Dokmanić and M. Vetterli, "EchoSLAM: simultaneous
  localization and mapping with acoustic echoes", ICASSP 2016: a
  co-located source and mic mapping a room while moving.
  https://infoscience.epfl.ch/record/215300
- J. B. Allen and D. A. Berkley, "Image method for efficiently simulating
  small-room acoustics", *JASA* 65(4), 943–950, 1979: image sources.
  https://pubs.aip.org/asa/jasa/article/65/4/943/765693/Image-method-for-efficiently-simulating-small-room
- A. Leshem, O. Naparstek and A. Nehorai, "Information theoretic adaptive
  radar waveform design for multiple extended targets", *IEEE JSTSP*, 2007:
  waveform design by mutual information.
  https://www.researchgate.net/publication/3481436_Information_Theoretic_Adaptive_Radar_Waveform_Design_for_Multiple_Extended_Targets
- The single-reflector range and bearing CRLBs are in \`imaging/crlb.py\`
  (plan 6.4, \`docs/imaging.md\` §4).

## Reproduce

\`\`\`bash
NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2 uv run python scripts/eval_information.py all
uv run pytest -q tests/imaging/test_information.py
\`\`\`

The stages are \`calibrate\`, \`grams\`, \`exact\`, \`analyse\` and \`figures\`.
Caches go to \`data/information/*.hdf5\` and the log to
\`data/logs/information.log\`. Wall time on the shared container (2 threads)
is about 3 min to calibrate, 4 min for the 21 Gram matrices, 11 min for
the exact 4 × 4 check, and 2 min for the analysis and figures.

Code: \`src/acoustic_system/imaging/information.py\`. Theory:
\`docs/imaging.md\` §11.
`;export{e as default};
//# sourceMappingURL=information_2026_09_26-B_JmcaXf.js.map