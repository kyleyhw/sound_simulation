# Project-plan audit — 2026-09-24

A full review of `PROJECT_PLAN.md` against the code, the reports in
`tests/reports/`, and the stated vision. The revised plan acts on every
finding below. The finding ids (A1, B3, …) are used in the plan.

Severity: **critical** = changes a published conclusion or blocks a
later phase; **major** = the plan's structure or assumptions are wrong;
**minor** = hygiene.

## Summary

The engineering is strong. The engine is fast, its regressions are
gated, the UI works, and the reports are unusually careful. The
problems are in what the results *mean* and in what the plan assumes:

1. The Phase 2 headline (held-out IoU ≈ 0.10) is **matched by
   predictors that never hear the audio**. The "information ceiling of
   the acquisition" reading is not supported yet.
2. The boundary condition documented as a "rigid wall" is physically a
   **pressure-release (soft)** boundary. The rooms are lossless.
3. Sensing outputs an occupancy mask, but Phase 4 expects an RIR.
   Laptop-only hardware also conflicts with a speaker-array Phase 3.
   The plan has no sim-to-real phase and no showcase phase.

---

## A. Scientific validity

### A1 — No-audio baseline matches the sensing headline (critical)

`scripts/eval_no_audio_baseline.py` scores predictors that ignore the
recording. It draws masks from the same generators and settings as the
v1/v2 archives: mean occupancy is 7.09 % for mixed rooms (v2 reports
7.0 %) and 5.78 % for rect rooms (v1 reports 5.8 %). Held-out n = 500,
± is one standard error.

| predictor (no audio) | v2 mixed rooms | v1 rect rooms |
| --- | --- | --- |
| predict-all | 0.071 ± 0.002 | 0.058 ± 0.001 |
| interior box, best border k | 0.103 ± 0.003 | 0.091 ± 0.002 |
| **prior map, best τ** | **0.104 ± 0.003** | **0.090 ± 0.002** |

Here are the published sensing numbers on the same metric:

| published recipe | IoU |
| --- | --- |
| v2 best (skip_v2, calibrated Bayes, K=4) | 0.100 |
| v2 ablation (joint_v2, K=8) | 0.101 |
| v1 best (joint + Bayes, K=8, @0.5) | 0.0924 |
| v1 single-pose baseline | 0.037 |

A fixed square that excludes an 11-cell border scores the same as the
calibrated K=4 fusion. Every v1 number is at or below the v1 prior map.

What this does and does not show:

- The baseline and the models were scored on different draws from the
  same generator, so the comparison is statistical. The SE is about
  0.003. To confirm on the exact masks, run
  `--train-archive/--heldout-archive` on the archives. They are not in
  this checkout.
- The demo figures (for example `data/plots/demo_room_mapping.png`,
  room 20) do show room-specific structure. That room was hand-picked
  from a 24-room sweep. The audio may carry *some* information. The
  point is that **mean IoU on filled masks cannot detect it**, because
  IoU rewards large central blobs when objects are small and their
  positions are uncertain. The low calibrated threshold (τ = 0.12) is
  exactly the setting that turns the output into such a blob.
- So "two architectures converge on 0.10, so this is the information
  ceiling" has a simpler explanation: both converge to the prior. The
  same metric artefact that inflated the 2.4× fusion gain (found in
  Task 2.3) also covers the residual gain.

Required changes (plan 0.1): report every result against the prior-map
baseline. Add threshold-free and information-theoretic metrics:

- information gain over the prior, in bits per room: $\Delta = \mathrm{NLL}_\text{prior} - \mathrm{NLL}_\text{model}$
- average precision (AP)
- boundary F-score and Chamfer distance
- skill $(\mathrm{IoU}-\mathrm{IoU}_\text{prior})/(1-\mathrm{IoU}_\text{prior})$

Then re-score every checkpoint.

### A2 — The target is not fully observable (major)

The mask labels every obstacle cell as occupied, including interiors.
With $p=0$ held inside obstacles, interior cells never interact with
the field. Only surfaces that some pose illuminates scatter energy, and
obstacles in the acoustic shadow of others are largely invisible. A
filled-mask target asks the network to hallucinate unobservable cells.
The only safe answer to that is the prior. Replace the target with
observable quantities (plan 2.4): illuminated boundaries, the room
polygon, distance fields, or object detections.

### A3 — Known-pose assumption (major)

Bayes fusion uses the *true* driver and mic positions of every pose. A
hand-carried laptop does not know its pose to centimetre accuracy.
Without a pose-noise study, the multi-pose recipe cannot be deployed.
Plan 2.9 covers joint pose and map estimation. Phones have IMUs, so
plan 5.6 covers them too.

### A4 — Boundary physics mislabelled; rooms are lossless (critical)

The code, `docs/simulate.md` and `CLAUDE.md` call the $p=0$ condition a
"rigid Dirichlet wall". The two conditions are different:

- $p = 0$ is a **pressure-release** (acoustically soft) boundary. Its
  pressure reflection coefficient is Γ = −1, as in a water–air
  interface.
- A **rigid** wall has zero normal particle velocity, i.e.
  $\partial p/\partial n = 0$ (Neumann). Its reflection coefficient is
  Γ = +1.

Real walls and furniture are close to rigid, with partial and
frequency-dependent absorption. The two conditions give different mode
shapes and eigenfrequencies. Only Neumann has a $k=0$ mode. They also
give opposite echo polarity.

The domain is also a closed, lossless box. It has no PML and no
absorbing materials, so reverberation is infinite. For sensing, late
reverberation is unphysically strong. For Phase 3, control inside a
lossless cavity is purely modal and says little about real rooms.
Absorbing boundaries and materials therefore come **before**
beamforming (plan 0.2, 1.6, 1.7).

### A5 — No physical units (major)

The simulation is dimensionless ($c = \Delta x = 1$). The 64-cell room
has a 12-cell mic baseline, and the chirp reaches $f = 0.45$ (λ ≈ 2.2
cells). Map a laptop's ~20 cm mic spacing onto 12 cells and the room is
about 1.1 m wide, sounded up to about 9 kHz. Map a 5 m room onto 64
cells instead and the mic spacing becomes 0.94 m, with a band of only
about 2 kHz. Neither is a laptop in a room.

The 2D-versus-3D gap is also undocumented. In 2D, spreading goes as
$1/\sqrt{r}$ and the Green's function has a wake tail. In 3D, spreading
goes as $1/r$ and Huygens' principle holds sharply. Plan 0.2.2 adds an
SI scene layer and a physical-plausibility check for datasets.

### A6 — Regression-only verification (major)

`check_simulate.py` proves the kernel reproduces `reference.npz`. It
does not prove that the reference solves the wave equation correctly.
There are no analytic checks, such as:

- cavity eigenfrequencies
- second-order grid convergence
- discrete-energy conservation
- the numerical-dispersion curve
- point-source Green's functions

Plan 0.2.3 adds them. Adding Neumann, PML, and impedance boundary
conditions makes them mandatory.

## B. Plan structure and consistency

- **B1 (major):** Phase 2 is marked "completed" even though its
  conclusions have since changed. The 2.1.4 text still claims "2.4×",
  which Task 2.3 already overturned. The "information ceiling" claim in
  2.3 is contested by A1. → Phase 2 is reopened as a research programme
  (2R), and the superseded claims are annotated in place.
- **B2 (minor):** Task 3.1.1 ("support multiple, independently
  controlled drivers") is already done. The engine and UI support N
  drivers with independent waveforms. → Marked completed.
- **B3 (critical for Phase 4):** Task 4.2.2 feeds "the model's output
  (inferred RIR)" to the beamformer. The model outputs an occupancy
  mask, not an RIR. Controllers need acoustic transfer functions (ATFs)
  from each speaker to each control point. → Add the missing bridge:
  estimate geometry and materials, re-simulate ATFs with a digital
  twin, then design the controller. The alternative is to estimate
  ATFs or room parameters directly.
- **B4 (major):** The vision (directed audio, directed noise
  cancellation with speaker arrays) conflicts with the laptop-only
  constraint (2 speakers, 2–4 mics). → The plan defines two hardware
  tiers. On a laptop, "virtual headphones" is *crosstalk cancellation*
  (transaural audio) with head tracking, which is physically achievable
  with 2 speakers. Arrays and personal sound zones are the research
  tier.
- **B5 (major):** No phase ever touches real audio. The consumer
  real-time claim is never measured. → New Phase 5 (sim-to-real)
  starts with a browser acoustic lab built on Web Audio. This is cheap
  and matches the browser-first principle.
- **B6 (major):** Phases 3–4 have no success criteria or decision
  gates. → Gates G0–G5 are added.
- **B7 (major):** The plan has no plan for showcasing the work. → New
  Phase 6: a zero-install WebGPU sandbox, a scene gallery, interactive
  explainers, a media pipeline, write-ups, and an open benchmark.
- **B8 (minor):** The UI has no probes or oscilloscope, no dB or
  time-averaged view, no materials, and no scene save or share. These
  are the features that make a sandbox useful and demoable. → New
  Phase 1C.

## C. Engineering hygiene

- **C1:** No CI. The gates exist only as scripts, and pytest is not a
  dev dependency. → Plan 0.3.
- **C2:** `README.md` is stale:
  - it presents the matplotlib editor as the main UI
  - it lists GPU acceleration as future work
  - it calls `learning/` a placeholder
  - every image link points to `simulation/plots/…`, which no longer
    exists (the images are in `data/plots/`)
- **C3:** `.DS_Store` and `.idea/` are tracked.
- **C4:** `frontend/package.json` has no `build` or type-check script,
  and `App.tsx` is a single ~1.1k-line component.
- **C5:** `CURRENT_STATE.md` has two "§2" sections, and its Phase 2
  table still presents the superseded 2.4× / "unsaturated in K" claim.
- **C6:** Datasets and checkpoints exist only on one machine and are
  gitignored. They have already been lost once (2026-05-14). → Publish
  them to a remote artefact store with checksums.

## Runtime

Reading the repo and reports took about 10 minutes. The baseline script
runs in 5.8 s for mixed rooms and 0.9 s for rect rooms (10k training
masks plus 500 held-out masks, CPU).
