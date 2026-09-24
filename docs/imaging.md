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
- two-reflector range resolution $c/2B$, and feature size $\lambda_\min/2$.

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
