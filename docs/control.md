# Sound-field control (Phase 7)

`src/acoustic_system/control/` designs loudspeaker controllers from transfer
functions measured in the FDTD engine, then checks each design by playing it
back through the engine. It covers:

- sound zones: delay-and-sum, pressure matching, acoustic contrast control,
  time reversal and broadband FIR design;
- laptop crosstalk cancellation ("virtual headphones") with head tracking;
- FxLMS noise cancellation and the size of the quiet zone;
- the sensing accuracy that control needs;
- drive signals optimised through the differentiable engine.

```
NUMBA_NUM_THREADS=1 uv run pytest tests/control            # 27 tests, ~25 s
NUMBA_NUM_THREADS=1 uv run python scripts/control_report.py # all experiments, ~7 min
NUMBA_NUM_THREADS=1 uv run python scripts/control_report.py --only ctc
```

The report is `tests/reports/control_2026_09_24.md`. Its figures and a
`results.json` holding every number are in
`tests/reports/control_2026_09_24_artifacts/`. Run with one numba thread:
numba's parallel loops collapse when the cores are shared.

| module | plan task | contents |
|---|---|---|
| `transfer.py` | 7.1.1 | `Room`/`Box` scenes in SI units, `simulate_drives`, `measure_transfer` → `TransferSet` |
| `metrics.py` | 7.1.2 | acoustic contrast, reproduction error, array effort, band energy |
| `beamforming.py` | 7.2 | `delay_and_sum`, `pressure_matching`, `acoustic_contrast_control`, `time_reversal`, `fir_from_weights`, `design_broadband`, `verify_fir` |
| `ctc.py` | 7.3 | `kirkeby_inverse`, `design_ctc`, `separation_db`, `TrackedCtc`, `displacement_sweep`, `verify_ctc` |
| `anc.py` | 7.4 | `fxlms` (runs sample by sample against the engine), `quiet_zone` |
| `requirements.py` | 7.5 | `SensingError`, `estimated_room`, `SensingStudy` |
| `differentiable.py` | 7.6 | `DiffScene`, `optimise_drives`, `gradient_check` (torch) |

## Scenes and units

A `Room` is 2D. Cell `(i, j)` sits at world position `origin + (i, j)·dx`, with
x along axis 0. The defaults come from `units.LAPTOP_ROOM`:

- cell size dx = 2.5 cm;
- c = 343 m/s;
- Courant number 0.5, so Δt = 36.4 µs (27.4 kHz).

The outer walls use any `Simulate` boundary. `"absorb"` is a locally reacting
impedance wall with normalised admittance β. Its normal-incidence absorption is
$1-((1-\beta)/(1+\beta))^2$: 0.33 at β = 0.1, 0.71 at β = 0.3 and 1 at
β = 1. For `"cpml"` the grid is padded by the layer, so the room keeps its
size. Furniture and wall treatment are `Box` blocks of a `physics.MATERIALS` id.

The furnished test room is 3.0 × 2.4 m, with a rigid sofa and an absorber panel:

| name | walls | notes |
|---|---|---|
| absorbing | β = 0.3 | the "absorbing room" of the plan targets |
| live | β = 0.1 | |
| treated | β = 1 | CTC only |
| anechoic | CPML | reference |

## Transfer functions (7.1.1)

Speaker $s$ plays a band-limited pulse $w[n]$: a Blackman-windowed band-pass
FIR over 100–2000 Hz, made exactly zero-mean. Every point $m$ is recorded, and
the recording is deconvolved:

$$H_{ms}(f) = \frac{Y_{ms}(f)}{W(f)} .$$

The engine's source is soft. Drive sample $u[n]$ is added to $p^{n+1}$ after
the wall update, and the recording is $y[n] = p^{n+1}$ at the probe. The
scheme is linear and time-invariant in $u$, so $y = h * u$ exactly, and
$H(f)$ is the DTFT of $h$. Two consequences:

1. Any drive designed from $H$ reproduces its prediction when played in the
   engine. A random 8-speaker drive matches to a relative error of 1e-6. The
   test pins 1e-4.
2. The pulse must be DC-free. A closed 2D room integrates DC. The -40 dB DC
   leak of a plain windowed design leaves a record tail at -18 dB; the
   zero-mean pulse brings it to -100 dB.

`TransferSet.at(freqs)` evaluates the DTFT exactly. It uses the cached record
FFT, a zero-padded FFT or a direct DTFT, whichever fits the frequency grid.

`impulse_responses(n)` returns band-limited IRs (used as the FxLMS secondary
path). They are masked to the excitation band with raised-cosine edges. Without
the mask, the inverse amplifies the room's near-DC response into a tail that
wraps around the FFT.

Two engines can run the measurement:

- `engine="numba"` (the reference) runs the speakers one after another.
- `engine="torch"` runs them as one `TorchFDTD` batch, which is the GPU route.
  It has no CPML or Mur boundary. On this CPU it is about 6× slower and agrees
  to 2e-6.

## Metrics (7.1.2)

With bright-zone and dark-zone transfer matrices $H_B$, $H_D$ and weights $q$:

$$C = 10\log_{10}\frac{\|H_B q\|^2/M_B}{\|H_D q\|^2/M_D},\qquad
\epsilon = \frac{\|p-p_T\|^2}{\|p_T\|^2},\qquad
E = 10\log_{10}\frac{\|q\|^2}{|q_r|^2},\quad |q_r|^2 = \frac{\|H_B q\|^2}{\|H_B e_r\|^2}.$$

- $C$ is the acoustic contrast, $\epsilon$ the normalised reproduction error
  and $E$ the array effort.
- The effort is measured against a single reference speaker $r$ that gives the
  same bright-zone level.
- The *band* contrast sums both zone energies over frequency before taking the
  ratio. That is what a flat-spectrum programme would get.
- `energy_contrast_db` and `band_energy` score engine recordings.

## Controllers (7.2)

**Delay-and-sum.** No room model. Each speaker gets the delay $r_s/c$ to the
focus (or a plane-wave delay along a steering direction), and all gains are
equal.

**Pressure matching.** Regularised least squares, targeting the reference
speaker's own field in the bright zone and silence in the dark zone:

$$q = (H_B^H H_B + \kappa H_D^H H_D + \lambda I)^{-1} H_B^H p_T .$$

**Acoustic contrast control.** The principal generalised eigenvector of
$R_B q = \mu (R_D + \delta I) q$, where $R = H^H H / M$. The Tikhonov term
$\delta$ is set relative to $\operatorname{tr}R_D/S$.

**Time reversal.** The matched filter $q = H_{f,:}^*$, which maximises the
focus pressure for a given drive energy (pinned by a test). In the time domain
it is the reversed IR from the focus, `time_reversal_fir`.

**Broadband FIR** (frequency sampling):

$$c_s[n] = w[n]\,\mathcal{F}^{-1}_N\bigl[g_k\,q_s(f_k)\,e^{-j2\pi f_k\tau}\bigr][n],\quad n < L.$$

- $g$ tapers the band edges.
- $\tau$ is the modelling delay.
- With $N > L$ (oversampling) the long room inverse is truncated rather than
  time-aliased onto the filter's start.

Narrowband weights have an arbitrary phase at each frequency, so every design
is first normalised by `normalise_to_reference`. The array must match the
reference speaker's level and phase at the bright-zone centre. This leaves the
contrast unchanged and gives the FIR a smooth, delay-like phase.

The defaults are L = 2048 (75 ms), τ = L/4, a Tukey window and 2× oversampling.
They were chosen on the test rooms. The room inverse is mostly causal with a
reverberant tail. The Hann + L/2 convention of the browser CTC throws away
half of that tail. Against the defaults, 2048-tap Hann + L/2 loses 2.6 dB in the absorbing room and 5.4 dB in the live room
(report table "FIR design choices").

`verify_fir` plays a programme through the filters in the engine and compares
the measured in-band contrast with the frequency-domain prediction from $H$
and the realised FIR response.

## Crosstalk cancellation (7.3)

The same conventions as `web/src/control/ctc.ts`:

- a Kirkeby regularised inverse
  $C = H^H(HH^H + \beta_f I)^{-1}e^{-j2\pi f\tau}$, with $\beta_f$ relative to
  the mean $|H|^2$;
- `taps[speaker, programme]`;
- plain stereo outside the design band;
- ears as two free-field points 17.5 cm apart, with no head.

$H$ is the FDTD plant in the room instead of the free-field model. Separation
for programme $p$ is $20\log_{10}|(HC)_{pp}|/|(HC)_{\bar pp}|$.

`TrackedCtc` measures the plant to a grid of points around the head once. It
can then evaluate fixed filters at a displaced head (7.3.2), or re-design them
at the tracked position (7.3.3), without re-simulating. Its defaults are
4096 taps, τ = L/4, Tukey, 4× oversampling and β = 0.001. `design_ctc` itself
keeps the browser defaults (L/2 delay, Hann).

## FxLMS (7.4)

A feedforward filtered-x NLMS runs sample by sample against the engine. Two
`Simulate` objects step together, one with control on and one with the primary
only.

$$y[n] = \sum_k w_k x[n-kD],\qquad
w_k \leftarrow w_k - \mu\frac{e[n]\,x'[n-kD]}{\epsilon+\sum_j x'[n-jD]^2},\qquad x' = \hat s * x.$$

The secondary-path estimate $\hat s$ is `TransferSet.impulse_responses`. Its
phase error is under 30° in 150–1500 Hz (a test pins this; FxLMS needs under
90°). Three details were needed for stable adaptation at 27.4 kHz:

- **Tap stride D.** The taps are spaced about a quarter period apart at the top
  of the band, so few weights span a useful window.
- **Updates every D samples.** Per-sample updates multiply the effective step
  by D. Near a room mode the plant answers a weight change only over the
  reverberation time, and the loop goes unstable (it did at 200 Hz).
- **A 30 Hz DC blocker on the error mic.** The mic is AC-coupled, and the
  filtered reference goes through the same blocker. Any tone onset injects net
  pressure, which a closed 2D room turns into a slowly decaying quasi-static
  wake. It is inaudible, but without the blocker it dominates the error at low
  frequencies. The steady-state optimum then only reached 10 dB.

`quiet_zone` takes the connected ≥ 10 dB region of the attenuation map
around the mic. The map is computed over a frozen-weight window, with fields
blocked the same way.

## Sensing requirements (7.5)

`SensingStudy` designs ACC from $\hat H$, simulated in an `estimated_room`, and
scores it with the true $H$. The estimated room can have:

- walls moved by ±δ;
- the admittance scaled;
- boxes shifted or missing.

The speakers and zones keep their world positions. Wall errors are quantised to
the grid (2.5 cm).

## Differentiable control (7.6)

`DiffScene` wraps a `TorchFDTD` run. The drives are $u_s = z_s * x$, with a
band-pass programme $x$ and per-speaker FIR filters $z_s$. The loss is the
negative bright/dark energy ratio (dB) over the simulated window.
`optimise_drives` runs Adam, from an ACC FIR or from random filters.
`gradient_check` compares autograd with central differences: the error is
8e-11 in float64. The optimised drives are replayed in the numba engine
(`verify_numba`).

## Results (from the report)

| target | result |
|---|---|
| ≥ 10 dB contrast, array, absorbing room | **25.5 dB** broadband ACC (8 speakers, 300–1500 Hz, 2048-tap FIRs), measured in the engine; prediction agrees to < 0.001 dB. Live room 17.2 dB, anechoic 28.9 dB. |
| ≥ 15 dB CTC, 2 laptop speakers, absorbing room | **17.7 / 20.5 dB** broadband (L / R programme, engine). Per frequency ≥ 15 dB over 84 % of 300–1500 Hz; dips to 2.5 dB at room modes. With β = 1 walls: ≥ 43.5 dB at every frequency. Live room: 9.8 dB, **not met**. |

The other results:

- **CTC head movement.** Fixed CTC filters stay above 15 dB only within about
  ±2.5 cm of lateral head movement. Filters re-designed for the tracked head
  keep 35–43 dB over ±15 cm.
- **FxLMS on tones.** Tones from 150 Hz to 1 kHz are cancelled by 35–120 dB at
  the mic. The ≥ 10 dB zone scales with wavelength, at about 1–2.5 × λ/10:
  53 cm at 150 Hz, 11 cm at 500 Hz and 4 cm at 1 kHz.
- **FxLMS on noise.** Band-limited noise at 100–500 Hz is reduced by 22 dB in
  the absorbing room and 11 dB in the live room.
- **Sensing requirements.** Exact-model ACC gives 27.5 dB in the absorbing
  room and 23.3 dB in the live room. A 2.5 cm wall error costs 9.5 dB
  (absorbing) and 14 dB (live).
  - For 15 dB, the walls must be known to about 4 cm (absorbing) and 1.5 cm
    (live).
  - Beyond about 5 cm of error, the room-aware design is no better than a
    free-field design.
  - Absorption within ±30 % and furniture within about 5 cm keep 15 dB.
- **Differentiable control.** On a 61 × 49 grid at 5 cm, 200–800 Hz, with
  256-tap filters, the ACC FIR reaches 14.6 dB in the 66 ms window. Adam
  through `TorchFDTD` raises it to 20.7 dB, or to 18.5 dB from random filters.
  The numba replay gives the same numbers.

## Known limits

- 2D only. The ears have no head, so there is no head shadowing, which would
  add natural separation. Speakers are monopole soft sources.
- The FIR filters run at the engine rate (27.4 kHz), so 2048–4096 taps
  correspond to 75–150 ms. A real system would design at a lower rate.
- Wall-position errors are resolved only to the 2.5 cm grid. The live-room
  requirement lies below one cell.
- FxLMS uses a perfect reference (the primary signal itself). Its step size is
  halved when a run does not converge (live room, 1 kHz).
- The differentiable objective sees only the simulated window.
- Plan 7.7 (UI) is not part of this module.
