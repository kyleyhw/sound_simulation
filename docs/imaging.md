# Physics-based room imaging (Phase 6, no ML)

`src/acoustic_system/imaging/` asks the question the plan audit (A1)
left open: **does the recorded sound carry room geometry that a
method can extract and that beats a predictor which never hears it?**
It holds classical imagers (no training beyond a two-parameter fusion),
observable targets, accuracy bounds and room-parameter estimators.
`scripts/eval_imaging.py` scores everything on the v2 archives. The
report is `tests/reports/imaging_2026_09_24.md`.

```
NUMBA_NUM_THREADS=1 uv run python scripts/eval_imaging.py \
    --out-dir tests/reports/imaging_2026_09_24_artifacts      # ~19 min on 1 core
NUMBA_NUM_THREADS=1 uv run pytest tests/imaging               # 26 tests, ~5 s
```

| module | plan | what it does |
|---|---|---|
| `targets.py` | 6.1 | illuminated boundary, outline polygons, signed distance |
| `ir.py` | 6.2.1 | chirp deconvolution, free-field Green's function, empty-room background |
| `image_source.py` | 6.2.2 | echo ellipses, free-space carving |
| `backprojection.py` | 6.2.3 | synthetic-aperture delay-and-sum |
| `time_reversal.py` | 6.2.4 | reverse-time migration with the engine |
| `fwi.py` | 6.2.5 | full-waveform inversion with `TorchFDTD` (proof of concept) |
| `crlb.py` | 6.4 | Cramér-Rao bounds, laptop design chart |
| `room_params.py` | 6.5 | T60, DRR, Eyring absorption |
| `pipeline.py` | — | runs every imager on one archive room |

## 1. Measurement model

The archives store, per pose $k$, the pressure $y_{km}[n]$ at mic $m$
and the source audio. The engine adds $s_n = A\,u(n\Delta t)$ to
$p^{n+1}$ at the source cell after the wall scrub, and the recorder
stores $p^{n+1}$. The engine is linear and time invariant, so

$$ y_{km}[n] = \sum_{j=0}^{n} h_{km}[j]\, s_{n-j}, $$

with $h_{km}[j]$ the discrete impulse response at lag $j$ (delay
$j\Delta t$). `ir.source_drive` reproduces $s_n$ from the archive
exactly: re-simulating an archive room with it matches the stored
recording bit for bit.

**Background removal.** The outer $p = 0$ box is the same in every room,
so it is known background, not target. `ir.empty_room_response` runs the
engine in the empty box, and the residual

$$ r_{km} = y_{km} - y^{(0)}_{km} $$

is the field scattered by the interior obstacles. It removes the direct
path *and* the outer-wall reverberation. It is exact for the discrete
scheme, unlike the analytic 2D Green's function
$h[j] = (\Delta x^2/\Delta t^2)[F(j\Delta t) - F((j-1)\Delta t)]$ with
$F(t) = \operatorname{arccosh}(ct/r)/(2\pi c^2)$
(`ir.free_field_green_2d`, also provided). The analytic model ignores grid
dispersion, which is large at the top of the v2 band
($\lambda \approx 2.2$ cells). In noise-free data the residual is
exactly zero until the first scattered wave arrives, apart from a
numerical precursor of 3 to 6 steps.

**Deconvolution (6.2.1).** The v2 chirp plays for the whole 400-step
window, so a late echo of the high end of the sweep is never recorded.
`TikhonovDeconvolver` solves the *truncated* convolution exactly:

$$ \hat h = (S^\top S + \lambda' I)^{-1} S^\top y, $$

with $S$ the lower-triangular Toeplitz matrix of $s$. This is the Wiener
estimator for a white prior. The resolvent is shared by every recording
with the same source, so a whole archive costs one matrix product.
`wiener_deconvolve` is the spectral form
$\hat H = Y S^* / (|S|^2 + \lambda \max|S|^2)$, which is exact only when
the recording outlasts the chirp and the decay. On the imaging task the
two give the same results (training-room sweep, below).

## 2. Imagers

All travel times are the exact 2D bistatic times
$t_{km}(\mathbf{x}) = (\lVert\mathbf{x}-\mathbf{s}_k\rVert +
\lVert\mathbf{x}-\mathbf{m}_{km}\rVert)/c$. The v2 generator places the
source and the mic pair independently, so every pair is bistatic. A
co-located laptop is the special case where the ellipse becomes a circle
of radius $c\tau/2$.

**Echo ellipses (6.2.2, `image_source_maps`).** Pick up to 6 envelope
peaks $(\tau_e, a_e)$ per trace and add
$a_e \exp(-(t(\mathbf{x}) - \tau_e)^2/2\sigma^2)$. Stacking poses
intersects the ellipses.

**Free-space carving (6.2.2, `carve_free_space`).** The first scattered
arrival $\tau_1$ bounds the nearest scatterer, so every pixel with
$t(\mathbf{x}) < \tau_1$ is free: an obstacle there would have echoed
earlier. The image counts the (pose, mic) ellipses covering each pixel.
This is *negative* evidence. It clears whole regions, which a filled
occupancy mask rewards most. The onset is detected on the raw residual
(relative level $10^{-3}$), or on the matched-filter envelope when the
data are noisy. Which one is used is chosen on training rooms.

**Delay-and-sum (6.2.3, `backproject`).**

$$ I(\mathbf{x}) = \sum_{k}\sum_m a_{km}\bigl(t_{km}(\mathbf{x})\bigr),
   \qquad a = |\hat h + j\mathcal{H}\hat h| / \mathrm{rms}, $$

over all $K$ poses and both mics (the synthetic aperture), gated to the
first 200 lags. `kind="signed"` and `kind="kirchhoff"` (the
$(j\omega)^{1/2}$ 2D migration filter) are also provided. In the
training sweep, the envelope without spreading compensation was best:
$\sqrt{r_s r_m}$ weighting amplifies the late, multiply scattered tail.

**Time reversal (6.2.4, `time_reversal.py`).** Re-emit the reversed
residuals $\tilde r_m[k] = r_m[T-1-k]$ from the mics in the empty box and
record $B^k$. Correlate $Q^n = B^{T-1-n}$ with the incident field $P^n$:

$$ I(\mathbf{x}) = -\sum_n P^n(\mathbf{x})\,Q^n(\mathbf{x}) \Big/
   \Bigl(\sum_n (P^n)^2 + \epsilon\Bigr). $$

Linearising the scrub $p \leftarrow (1-o)p$ about $o = 0$ shows that the
numerator is exactly $-\partial J/\partial o$ for the misfit
$J = \tfrac12\lVert r - \delta y\rVert^2$: time reversal is the adjoint
of the Born operator. `test_time_reversal_is_the_fwi_adjoint` checks this
against `TorchFDTD` autograd to $10^{-5}$ (relative) away from the
injection cells.

**Full-waveform inversion (6.2.5, `fwi.py`).** Minimise the normalised
data misfit plus $\lambda_\text{TV}\,\mathrm{TV}(o)$ over
$o = \sigma(\theta)$ with Adam. Frequency continuation uses low-pass
cut-offs 0.12, 0.25 and 0.47, with 15, 15 and 20 iterations (the
defaults). A
fractional occupancy is a per-step absorber, so $\theta$ starts at
$-8$ rather than at the prior. Device cells are pinned to air. One
64² room with K = 4 poses and 50 iterations takes about 9 s on one CPU
core.

**Fusion.** Each image $z$ is smoothed ($\sigma \in \{0,1,2,3\}$ cells)
and standardised per room. It is combined with the no-audio prior as

$$ \operatorname{logit} q(\mathbf{x}) = \operatorname{logit}\hat\pi(\mathbf{x})
   + \sum_i a_i z_i(\mathbf{x}) + b, $$

with $(a, b)$ fitted by logistic regression on training rooms. The
smoothing (and the carving onset detector) is chosen there by log-loss.
"All physics" uses the four images together (five parameters).

## 3. Targets (6.1)

- **Illuminated boundary** (`illuminated_boundary`): surface cells that a
  pose's source reaches *and* one of its mics sees, by a straight ray,
  either directly or with one bounce off the outer walls (image method:
  unfold the ray and fold it back). The target is the union over poses.
  With K = 4 poses on the 500 held-out rooms it keeps 66 % of all surface
  cells (73 % per room on average) and no interior cell.
- **Outline polygons** (`obstacle_polygons`, `free_space_outline`):
  marching squares at level 0.5 plus Douglas-Peucker. The free-space
  outline traces the air region connected to the devices (the room as a
  mapper would draw it). `rasterize_polygons` maps polygons back to
  cells, and the round trip is exact on the toy rooms.
- **Signed distance** (`signed_distance`): $\phi = d(\text{air} \to
  \text{obstacle}) - \tfrac12$ outside and its negative inside. The zero
  level set sits on the cell faces. Truncation and outer walls are
  optional.

## 4. Cramér-Rao bounds (6.4)

For a known waveform in white noise,
$\operatorname{var}\hat\tau \ge 1/(8\pi^2\beta^2\,\mathrm{SNR})$, with
$\mathrm{SNR} = 2E/N_0$ the matched-filter output SNR and $\beta$ the RMS
frequency. $\beta^2 = B^2/12$ from the envelope alone, and
$\beta^2 = f_c^2 + B^2/12$ when the carrier phase is usable. So:

- range to a planar reflector: $\sigma_R = \tfrac{c}{2}\sigma_\tau/\sqrt N$,
  which scales as $1/(B\sqrt{\mathrm{SNR}})$ (unit-tested);
- bearing from TDOA with the range as a nuisance parameter:
  $\sigma_\theta = c\,\sigma_\tau / (\cos\theta\,\sqrt{\sum_m (x_m-\bar x)^2}\,\sqrt K)$,
  which is $\sqrt2\,c\sigma_\tau/(d\cos\theta)$ for a pair;
- two-reflector range resolution $c/2B$, and feature size $\lambda_{\min}/2$.

A Monte Carlo test confirms that the cross-correlation ML delay
estimator reaches the bound at high SNR (within 20 %).

## 5. Room parameters (6.5)

`estimate_room_params` performs spectral division (the recording must
hold the whole decay), band-limits the result to the excitation band,
fits a Schroeder EDC (T20, T30), computes the DRR, and inverts Eyring in
2D:
$\bar\alpha = 1 - \exp(-6\ln10\,\pi S/(c L T_{60}))$.

Two practical findings:

- A soft source integrates DC. Any excitation with
  $\sum s_n \ne 0$ or $\sum n\,s_n \ne 0$ pumps a quasi-static pressure
  that the room never sheds, and the EDC flattens. `dc_free_chirp` (the
  second difference of a tapered chirp) removes both moments.
- Without the band-pass, near-DC residue from the division doubles the
  broadband T60.

## 6. Development sweep (training rooms only)

Training rooms 0-199: the fusion was fitted on 0-99 and scored by AP on
100-199. The prior's AP on those rooms is 0.153. Values are the fused AP
with the raw and σ = 1.5-smoothed image.

| variant | fused AP |
|---|---|
| back-projection, Tikhonov λ ∈ {1e-3, 1e-2, 1e-1}, gate 200, offset −2 | 0.279 (all three) |
| same, gate 120 / 400 lags | 0.215 / 0.266 |
| same, spectral Wiener / matched filter | 0.275 / 0.258 |
| same, with √(r_s r_m) spreading weight (gate 400, first sweep) | 0.173 vs 0.264 without |
| carving, onset on the deconvolved envelope (0.2 of max) | 0.293 |
| carving, onset on the raw residual at 1e-3 of max | **0.304** |
| time reversal, full window / first 200 / first 120 lags | 0.215 / 0.201 / 0.165 |

FWI (3 training rooms): Adam step 0.3 and TV weight 2e-4. Doubling the
iterations did not help the filled-mask AP.

## 7. Results (held-out, see the report)

`tests/reports/imaging_2026_09_24.md`. 500 held-out rooms, K = 4 poses,
everything fitted on training rooms:

| method (fused with the prior) | IoU | ΔIoU vs prior (z) | ΔAP (z) |
|---|---|---|---|
| no-audio prior map | 0.101 | — | — |
| echo ellipses | 0.128 | +0.027 (9.0) | +0.078 (12.7) |
| time reversal | 0.128 | +0.026 (9.1) | +0.064 (11.1) |
| free-space carving | 0.163 | +0.061 (18.3) | +0.188 (22.4) |
| back-projection | 0.168 | +0.066 (16.8) | +0.149 (18.0) |
| all physics | **0.189** | **+0.088 (19.3)** | **+0.217 (23.4)** |
| all physics, K = 8 | 0.244 | +0.143 (34.1) | +0.331 (32.8) |
| all physics, 20 dB noise | 0.187 | +0.086 (18.5) | +0.203 (22.6) |
| FWI alone (40 rooms, prior 0.087) | 0.337 | +0.251 (4.5) | +0.335 (5.8) |

Illuminated-boundary target: the all-physics fusion gives IoU 0.056 vs
0.027 (z = 24.4) and boundary F 0.224 vs 0.156. FWI gives 0.306 vs 0.024
(z = 8.0). The Phase 6 "done when" criterion (beat the no-audio baseline
on held-out rooms with significance) is met by every imager.

Limitations: noise-free simulation (20 dB white noise tested), exact
poses, a known empty outer box for background subtraction, and an
inverse crime for FWI. Details are in the report.

## 8. Learned models on the physics images (6.3), pose error, passive and 3D sensing (6.6)

The report is `tests/reports/imaging_models_2026_09_24.md`. Modules added
(no existing module changed):

| module | plan | what it does |
|---|---|---|
| `pose_images.py` | 6.3, 6.6.1 | every imager of §2 kept *per pose*, pose geometry channels, pose perturbation |
| `models.py` | 6.3.1-6.3.4 | compact U-Net, impulse-response migration net, global IR encoder, pose-set net, D4 augmentation, temperature, reliability |
| `pose_refine.py` | 6.6.2 | pose refinement against the empty-box model (joint exhaustive least squares, alternating quiet-start search) |
| `passive.py` | 6.6.4 | GCC-PHAT interferometric imaging without the source signal |
| `room3d.py` | 6.6.5 | 3D rooms, 4-mic arrays, 3D back-projection and carving |

```
NUMBA_NUM_THREADS=1 uv run python scripts/eval_imaging_models.py \
    --out-dir tests/reports/imaging_models_2026_09_24_artifacts   # ~4 h on a shared core, cached
NUMBA_NUM_THREADS=1 uv run pytest tests/imaging                    # 57 tests, ~9 s
```

### 8.1 Per-pose images

Each imager of §2 is a sum of per-pose terms (back-projection normalises
each trace, ellipses and carving counts add per pair, time reversal scales
each pose by its own RMS), so

$$ I(\mathbf{x}) = \sum_{k=1}^{K} I_k(\mathbf{x}) $$

and `pose_images.aggregate(compute_pose_images(room).images)` equals
`compute_room_images(room)` exactly (unit-tested). The pose geometry is
rendered on the grid as $\lVert\mathbf{x}-\mathbf{s}_k\rVert/N$ and the
bistatic lags $t_{km}(\mathbf{x})/(4N\Delta t)$.

### 8.2 Models (all a residual on the prior logit)

Every model outputs two maps, the filled mask and the illuminated boundary,

$$ \operatorname{logit} q_h(\mathbf{x}) = \operatorname{logit}\hat\pi_h(\mathbf{x})
   + f_\theta(\cdot)_h(\mathbf{x}), \qquad h \in \{\text{fill}, \text{illum}\}, $$

with the last layer zero-initialised, so training starts at the prior. The
loss is the summed binary cross-entropy of the two heads. Training uses
the eight symmetries of the square box (images, targets, prior and device
coordinates transformed together).

- **Aligned U-Net (6.3.1).** Three levels (widths 12, 24, 48; 119 k
  parameters). Inputs: the four pose-summed images, each standardised per
  room, and the two prior logits.
- **IR migration net (6.3.2).** A shared 1D filter bank (three
  convolutions, kernel 9) maps each (IR, envelope) trace to 8 channels
  $a_{kmc}[j]$, which are migrated by a differentiable delay-and-sum,

  $$ F_c(\mathbf{x}) = \frac{1}{KM}\sum_{k,m} a_{kmc}\bigl(t_{km}(\mathbf{x}) - 2\bigr), $$

  (`models.migrate`, linear interpolation, equal to `backproject` with
  the same filter; unit-tested) and fed to the same U-Net. It learns the
  detection filter that §2 fixes by hand.
- **Global IR encoder (6.3.2, control).** The Phase 2 layout with IRs in
  place of spectrograms: per-pose 1D encoder to a vector, device
  coordinates appended, mean over poses, decoder from $8\times8$. No
  spatial alignment.
- **Pose-set net (6.3.3).** A shared encoder $\phi$ maps each pose's
  $(4 + 3)$ channels (standardised images plus geometry) to features
  $F_k$, pooled as $[\sum_k \alpha_k F_k,\ \bar F,\ \max_k F_k]$ with
  per-pixel attention $\alpha_k = \operatorname{softmax}_k a^\top F_k$:
  invariant to pose order, defined for any $K$. Trained on random subsets
  of 1-4 poses.
- **Uncertainty (6.3.4).** Monte Carlo dropout (`Dropout2d`, $p = 0.1$,
  16 draws) and a two-member deep ensemble average probabilities; a scalar
  temperature $T$ minimising the validation log-loss of
  $\sigma(z/T)$ is then applied. Reliability uses 15 equal-mass bins.

### 8.3 Pose error (6.6.1) and refinement (6.6.2)

`perturb_poses` displaces every device independently by
$\operatorname{round}(\mathcal{N}(0, \sigma^2))$ per coordinate; the
recordings stay at the true cells, and both the empty-box background and
all travel times use the assumed cells. Refinement fits the empty-box model
$G_{\hat{\mathbf{s}}\to\hat{\mathbf{m}}}$ to the recordings within a window
of radius $\lceil 2\sigma\rceil$:

- *joint*: $\min \sum_m \lVert y_m - G_{\hat{\mathbf{s}}\to\hat{\mathbf{m}}_m}\rVert^2$,
  exhaustive over the source window (one simulation per candidate source;
  the mics then separate);
- *alternating*: exhaustive mic search in the incident field, and
  exhaustive source search via reciprocity
  $G_{\mathbf{a}\to\mathbf{b}} = G_{\mathbf{b}\to\mathbf{a}}$ (exact for the
  symmetric discrete Laplacian, tested to $10^{-6}$), with the quiet-start
  cost $\sum_n \log(\epsilon E + \sum_{j\le n} r_j^2)$.

Poses are not identifiable exactly from the pre-scatter data (the direct
wave only fixes source-mic distances, and the scattered onset trails the
direct arrival by a median of 4 lags). The fit instead finds cells whose
empty-box response matches the data, which is what the background
subtraction needs. With the *true* obstacle map in the model the joint fit
puts 93 % of devices on their exact cell, against 36 % with the empty box
(10 validation rooms, σ = 1).

### 8.4 Passive (6.6.4) and 3D (6.6.5)

`passive.passive_images` computes, per pose and mic pair, the GCC-PHAT
$g = \mathcal{F}^{-1}[Y_1Y_2^*/|Y_1Y_2^*|]$, which does not depend on the
unknown source spectrum, subtracts the envelope of the empty box's GCC
(from an impulse probe, not the source), and migrates it along the
direct-scattered delays $t(\mathbf{s}\to\mathbf{x}\to\mathbf{m}_1) -
t(\mathbf{s}\to\mathbf{m}_2)$ and its mirror. The speaker and mic
positions are assumed known; the waveform is not used.

`room3d.py` generates $32^3$ $p = 0$ boxes with 2-4 box obstacles, a
source with four mics on a $\pm3$-cell square, a known Ricker drive
($f_0 = 0.12$), and runs 3D envelope back-projection and first-arrival
carving (echo peak lag and onset lag read from the drive).

### 8.5 Results (held-out, 500 rooms, K = 4 unless stated)

| method | filled IoU | AP | info gain (bits) | ΔIoU vs logistic (z) | illum. IoU | illum. BF |
|---|---|---|---|---|---|---|
| no-audio prior | 0.101 | 0.128 | 0 | — | 0.027 | 0.155 |
| logistic all physics (§7) | 0.189 | 0.344 | 227 | — | 0.056 | 0.224 |
| global IR encoder (no alignment) | 0.099 | 0.157 | 65 | −0.090 (−21.0) | 0.026 | 0.116 |
| pose-set net (6 epochs) | 0.346 | 0.552 | 447 | +0.156 (21.3) | 0.133 | 0.435 |
| aligned U-Net | 0.362 | 0.578 | 480 | +0.173 (22.0) | 0.163 | 0.495 |
| IR migration net | **0.370** | **0.598** | **509** | **+0.180 (24.4)** | 0.170 | 0.483 |
| U-Net ensemble ×2 + T | 0.368 | 0.586 | 488 | +0.178 (22.1) | **0.172** | 0.486 |
| U-Net, K = 8 | 0.450 | 0.684 | 621 | +0.206 (22.8) | — | — |
| passive GCC-PHAT U-Net (no source) | 0.110 | 0.154 | 22 | vs prior +0.009 (4.1) | 0.030 | 0.169 |

Findings: spatial alignment is what matters. The same impulse responses
reach 0.370 through a migration layer and stay at the prior (0.099)
through a global encoder, which reproduces the Phase 2 plateau. The
learned models nearly double the logistic fusion. The pose-set net is not
better than summing the images first (budget-matched it is +0.008 ± 0.004
at K = 4 and −0.004 ± 0.005 at K = 8). The U-Net is already calibrated
($T \approx 1$). Pose error of half a cell costs a third of the IoU;
joint refinement recovers 65-78 % of the loss for σ = 0.5-3 cells.
Passive imaging barely beats the prior. In 3D, carving plus back-projection
fused with a 3D prior gives voxel IoU 0.139 against 0.070 (z = 14.3,
100 held-out rooms).

**Next-best pose (6.6.3, evaluated here).** From the first two of the eight
held-out poses, greedily adding two more with the web panel's heuristic
(most fused-map entropy within 12 cells of the pose centroid) or with a
sensitivity-weighted entropy $\sum_x H(q_x)\,v(d_c(x))$ ($v$ the mean
squared logit change against the distance to the new pose's nearest device,
learned on training rooms) beats random choice by only +0.006 IoU at K = 4
(z = 2.3 and 2.5); the hindsight-best choice would gain +0.076 (z = 34).

## 9. A generative model of whole maps (6.3.4)

The report is `tests/reports/imaging_generative_2026_09_25.md`. There is
one new module, `generative.py`, and one script,
`scripts/eval_generative.py`.

```
NUMBA_NUM_THREADS=1 uv run python scripts/eval_generative.py \
    --out-dir tests/reports/imaging_generative_2026_09_25_artifacts   # ~2 h on two shared cores, cached
NUMBA_NUM_THREADS=1 uv run pytest tests/imaging/test_generative.py   # 18 tests, ~3 s
```

### 9.1 Masked (absorbing-state) discrete diffusion

§8 gives calibrated marginals $q_i = P(m_i = 1 \mid \mathbf{c})$ for the
filled mask $\mathbf{m} \in \{0,1\}^{64\times64}$, where the conditioning
$\mathbf{c}$ is the four aligned images plus the prior logits (K = 4). The
generative model targets the joint $p(\mathbf{m} \mid \mathbf{c})$.

- *Forward process:* per room draw a visible fraction $a \sim \mathcal{U}(0,1)$,
  then make each pixel visible with probability $a$; hidden pixels are
  "absorbed".
- *Reverse model:* a U-Net sees $\mathbf{c}$, the visible values (as
  $\pm1$, 0 if hidden) and the visibility mask, and outputs

  $$ \operatorname{logit} P(m_i = 1 \mid \mathbf{m}_V, \mathbf{c})
     = \operatorname{logit}\hat\pi_i + f_\theta(\mathbf{c}, \mathbf{m}_V, V)_i, $$

  trained by the binary cross-entropy on the hidden pixels. It is
  warm-started from the §8 U-Net, with zero weights on the two state
  channels, so at initialisation it returns the U-Net's marginal for any
  state.
- *Sampling:* draw a uniformly random pixel order $\sigma$ and reveal it in
  $T$ blocks, with the cumulative count after step $s$ equal to
  $\lceil P(1 - \cos(\tfrac{\pi}{2} s/T))\rceil$. Each block is drawn from
  the current conditionals. With one pixel per block this is the chain rule
  $p(\mathbf{m}) = \prod_i p(m_{\sigma(i)} \mid m_{\sigma(<i)})$, exact for
  any order. With $T$ blocks, the pixels of one block are drawn independently
  given the earlier ones.
- *Mean map:* $\bar q_i = \frac1N\sum_n p_i^{(n)}$, where $p_i^{(n)}$ is the
  probability pixel $i$ was drawn with in sample $n$. Since
  $\mathbb{E}[p_i^{\text{reveal}}] = \mathbb{E}[m_i]$ (tower property), this
  Rao-Blackwellised mean is unbiased for the same marginal as the sample
  average, has lower variance, and never returns an exact 0 or 1.
- An optional sampling logit bias $b$ is added to every conditional,
  chosen on validation rooms.

### 9.2 Multi-sample scores

For a sample set $\{X_n\}_{n=1}^N$ and truth $y$, the fair estimator of a
kernel score with distance $d$ is

$$ S = \frac1N\sum_n d(X_n, y) - \frac{1}{2N(N-1)}\sum_{n\ne n'} d(X_n, X_{n'}) . $$

| score | $d$ | sees |
|---|---|---|
| energy score | $\lVert X - y\rVert_2 = \sqrt{\text{Hamming}}$ | marginals and dependence (strictly proper; with $\lVert\cdot\rVert_2^2$ = Hamming it would see the marginals only) |
| Jaccard kernel score | $1 - \lvert X\cap y\rvert / \lvert X\cup y\rvert$ (Tanimoto kernel, positive definite) | set overlap, as IoU does (proper) |
| pixel CRPS | $\sum_i \lvert X_i - y_i\rvert$ | marginals only: its expectation is the Brier score $\sum_i (q_i - y_i)^2$ |
| variogram score | $\sum_{(a,b)} w_{ab}\bigl(\lvert y_a - y_b\rvert - \mathbb{E}\lvert X_a - X_b\rvert\bigr)^2$ over the 12 neighbour offsets within $\sqrt8$, $w = 1/\text{distance}$ | neighbour dependence only (proper, not strictly) |

The *decorrelated* control permutes each pixel's $N$ values independently
across samples. It keeps every pixel's empirical marginal exactly, so the
pixel CRPS is unchanged, and it removes all dependence.

### 9.3 Results (held-out, 500 rooms, K = 4, N = 32, T = 32)

| | IoU (mean map) | info gain (bits) | ECE | energy score | variogram | best-of-32 IoU |
|---|---|---|---|---|---|---|
| U-Net (§8) / its independent Bernoulli samples | **0.362** | **480** | **0.002** | **9.12** | 1210 | 0.272 |
| generative, sample mean / samples | 0.311 | 368 | 0.022 | 9.53 | **785** | **0.404** |
| generative, decorrelated | — | — | — | 9.60 | 1456 | 0.237 |

Findings. The samples are coherent: solid objects rather than
salt-and-pepper noise. Holding the marginals fixed, the coherence improves
every dependence-sensitive score: energy −0.072 (z = −20), variogram −670
(z = −39), best-of-32 +0.17. But fine-tuning for completion cost marginal
accuracy (0.302 IoU with nothing visible, against 0.362), and parallel
decoding over-fills by 29 %. So on the strictly proper energy score,
independent samples from the U-Net win: +0.41 ± 0.03 (z = +12.5) in their
favour. Sample diversity correlates with error across rooms (Spearman 0.58)
but no better than the U-Net's entropy (0.54-0.61). A validation-chosen
logit bias (−0.5) fixes nothing that matters. The calibrated U-Net remains
the better uncertainty model. The generative sampler is the tool 6.6.3
lacked for simulating poses on posterior samples, once its marginals match.

## 10. Pose-robust sensing

The report is `tests/reports/pose_robust_2026_09_25.md`. There is one new
module, `pose_robust.py`, and one script, `scripts/eval_pose_robust.py`.

```
NUMBA_NUM_THREADS=1 uv run python scripts/eval_pose_robust.py --stages dev,augdata,augtrain,held,polish,score \
    --out-dir tests/reports/pose_robust_2026_09_25_artifacts   # ~5.5 h of one core, cached; split with --chunks / --conditions
NUMBA_NUM_THREADS=1 uv run pytest tests/imaging/test_pose_robust.py   # 7 tests, ~1 s
```

### 10.1 A rigid error model

§8.3 displaces every device independently. A laptop (or any device with a
fixed speaker-mic layout) is misplaced as a whole: one rigid transform per
placement. With the pose's devices $\mathbf{d}_i$ and centroid $\mathbf{c}$,

$$ \hat{\mathbf{d}}_i = \operatorname{round}\bigl(R(\theta)(\mathbf{d}_i-\mathbf{c})
   + \mathbf{c} + \mathbf{t}\bigr), \qquad
   \mathbf{t}\sim\mathcal{N}(0,\sigma_t^2 I),\ \theta\sim\mathcal{N}(0,\sigma_\theta^2) $$

(`perturb_poses_rigid`). Rounding keeps the inter-device distances only to
within a cell. In the v2 archives the source and the mic pair of a pose are
placed independently (median source-mic distance 29 cells), so the RMS lever
arm about the centroid is 16.3 cells and a rotation of $\theta$ moves a
device by about $0.28\,\theta/{\rm deg}$ cells, more than for a real 20-30 cm
laptop. The magnitudes used, at 2.5 cm per cell: $(\sigma_t, \sigma_\theta) =
(0.5, 1°)$ (placement on measured marks, or phone VIO at 1.5-4 cm absolute
error), $(1, 2°)$ (careful hand placement) and $(2, 5°)$ (casual placement),
plus translation only $(1, 0°)$ and rotation only $(0, 3°)$.

### 10.2 Corrections (all search the same candidate set)

Per pose, the candidates are the assumed pose moved by every integer
translation within $R = \lceil 2\sigma_t\rceil$ and rotated by
$\{-2,-1,0,1,2\}\,\sigma_\theta$ about its centroid, rounded and
de-duplicated (`rigid_candidates`; 22-350 per pose). Each candidate is
scored once (`evaluate_candidates`): the empty-box recording $y^{(0)}$ at its
cells (one engine run per distinct source cell), three data misfits
(least squares, Huber with $\delta = 0.05\,\mathrm{rms}(y)$, and the
quiet-start cost of §8.3), and two focus images of its residual
$y - y^{(0)}$ (back-projection and first-arrival carving).

- **Rigid least squares** (`select_misfit`): the candidate with the least
  misfit. It is §8.3's joint fit with 3 unknowns per pose instead of
  $2(1 + M)$.
- **Autofocus** (`autofocus_select`): no model of the obstacles. With
  every per-pose focus feature $\tilde I_k$ smoothed, zero-mean and of unit
  norm,

  $$ \max_{c_1..c_K}\ \Bigl\lVert \sum_k \tilde I_k(c_k) \Bigr\rVert^2
     - \lambda \sum_k \rho(c_k),
     \qquad \Bigl\lVert \sum_k \tilde I_k \Bigr\rVert^2 = K + 2\sum_{k<l}
     \langle \tilde I_k, \tilde I_l\rangle, $$

  the quadratic sharpness metric of SAR autofocus, i.e. cross-pose
  coherence, with $\rho$ the negative log Gaussian prior of the correction.
  `metric="entropy"` minimises the Shannon entropy of
  $(\sum_k\tilde I_k)^2$ instead. Coordinate ascent over poses from the
  identity; every update is monotone.
- **Autofocus + rigid**: the same ascent with the relative data misfit
  $\beta\log(m_c/m_0)$ subtracted.
- **Rigid + polish** (`refine_room_rigid_polish`): rigid least squares,
  then §8.3's per-device joint fit within one cell of each refined device
  (absorbs the rotation grid step and the rounding).
- **Pose-jitter augmentation**: the §8.2 U-Net retrained (6 epochs) on
  the exact images of training rooms 500-3999 plus four copies imaged at
  poses perturbed by a random member of {independent σ = 0.5, 1, 2; rigid
  (0.5, 1°), (1, 2°), (2, 5°)}; its threshold is taken on validation rooms
  perturbed by the same mixture. It needs no test-time search.

Settings were chosen on validation rooms 0-39 (three conditions; the grid
is `dev_selections`): least squares for the rigid fit; carving features,
coherence and prior weight 0.2 for autofocus; both channels, data weight 3
for autofocus + rigid.

### 10.3 Results (held-out, 500 rooms, K = 4; U-Net exact 0.362, prior 0.101)

| condition | no correction | jitter-trained U-Net | autofocus | rigid LS | joint LS (§8.3) | rigid + polish |
|---|---|---|---|---|---|---|
| rigid (0.5 cell, 1°) | 0.280 | 0.290 | 0.284 | 0.319 | 0.318 | **0.329** |
| rigid (1 cell, 2°) | 0.203 | 0.247 | 0.220 | 0.295 | 0.308 | **0.324** |
| rigid (2 cells, 5°) | 0.143 | 0.184 | 0.160 | 0.245 | 0.281 | **0.290** |
| rigid (1 cell, 0°) | 0.272 | 0.284 | 0.287 | **0.350** | 0.320 | 0.323 |
| independent σ = 1 | 0.144 | 0.224 | 0.146 | 0.156 | **0.314** | 0.267 |

Findings. The independent model overstates the damage. At similar device
error, a rigid error costs 0.159 and a pure translation 0.090, against 0.218
for independent jitter, because only changes of the pose's internal
geometry break the direct-wave background. Rigid refinement with a one-cell
per-device polish recovers 60-76 % of the rigid loss (on par with the §8.3
joint fit, +0.008 to +0.015). A pure translation is recovered to within
0.012 of exact poses. Autofocus (cross-pose coherence) adds at most +0.017
and does not move the poses: a pose error corrupts the background
subtraction rather than shifting a sharp image. Jitter training gains
+0.01 to +0.08 without refinement. It loses 0.034 at exact poses and
0.015-0.025 after refinement, so a pipeline that refines should keep the
exact-pose network.
