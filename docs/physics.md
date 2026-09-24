# Physics model and verification (Phase 5)

This page covers what the FDTD engine solves, which boundary and material
models it offers, and how each one is checked against analytic results.
The numbers below come from `scripts/verify_physics.py`, which writes
`tests/reports/physics_artifacts/verification.json` and the figures next to
it.

```
uv run python scripts/verify_physics.py                 # all stages, ~1 min
uv run python scripts/verify_physics.py --only edges    # one stage
```

## Model

The engine solves $p_{tt} = c(x)^2 \nabla^2 p$ with the leap-frog scheme
$p^{n+1} = 2p^n - p^{n-1} + (c\Delta t/\Delta x)^2 \mathcal{L}p^n$. It uses
the 5-point (2D) or 7-point (3D) Laplacian, or optionally the compact
9-point 2D scheme. There are two code paths:

- **Fast path** (`calculate.py`): pressure-release walls ($p = 0$,
  reflection −1), uniform $c$. It is bit-compatible with `reference.npz`.
- **General path** (`physics.py`, `cpml.py`): used for everything else.
  The browser engine (`web/src/engine/`) implements the same equations,
  and parity fixtures hold it to 1e-4.

| Feature | Model | API |
|---|---|---|
| Rigid wall | Neumann, the ghost cell mirrors the cell, so the face drops out of the Laplacian | material id 2, `boundary="rigid"` |
| Impedance wall | locally reacting spring-damper $Rv + K_s x = p$; $\beta = \rho c/R$; high-pass when $K_s > 0$ | ids 3–6, `boundary="absorb"`, `boundary_beta` |
| Mur edge | first-order Engquist–Majda | `boundary="mur"` |
| Sponge | graded damping $p_{tt} + 2\sigma p_t$, cubic profile | `boundary="sponge"`, `sponge_cells` |
| **CPML** | convolutional PML for the second-order equation (Pasalic & McGarry 2010), with a CFS frequency shift α | `boundary="cpml"`, `cpml_cells` |
| Heterogeneous medium | relative speed map $c(x)/c_0$, CFL-checked | `set_speed_map` |
| Per-face boundaries | any mix, ordered (axis 0 low, axis 0 high, …) | `boundary=(...)` |
| SI units | `units.PhysicalScale`: metres, seconds, Hz, $c = 343$ m/s | `units.py` |
| Sources | sub-cell (bilinear) and band-limited Gaussian injection, dipole, cardioid | `sources.py` |
| Mics | sub-cell sampling, directivity, noise, gain | `sources.MicModel` |

## Verification results

### 1. Cavity eigenfrequencies (5.3.1)

The field is projected onto each exact discrete eigenvector, so
near-degenerate modes stay separate. An example is (3,1) and (2,2) in a
41 × 31 box, which lie 1.4 % apart. Each projection is a single sinusoid.

| Box | Modes checked | Error vs discrete dispersion | Error vs continuum |
|---|---|---|---|
| pressure-release 41 × 31 | (1,1) (2,1) (1,2) (2,2) (3,1) | ≤ 4e-6 % | 0.02–0.13 % |
| rigid 41 × 31 | (1,0) (0,1) (1,1) (2,0) (2,1) (3,1) | ≤ 2e-5 % | 0.02–0.13 % |

The continuum error is the scheme's numerical dispersion. It is within
the Phase 5 target of 0.5 % for every mode.

### 2. Grid convergence (5.3.2)

The test uses the exact standing mode $\sin \pi x \sin \pi y \cos \sqrt2 \pi t$
on the unit square, Courant 0.5, with the maximum error at t = 1.

| n | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| max error | 3.34e-3 | 8.47e-4 | 2.15e-4 | 5.25e-5 |
| observed order | – | 1.98 | 1.98 | 2.03 |

### 3. Energy conservation (5.3.3)

The discrete leap-frog energy
$E = \tfrac12\sum((p-p_{prev})/\Delta t)^2 + \tfrac12\sum_{faces}(Dp)(Dp_{prev})$
is summed over fluid cells and fluid–fluid faces. Rigid faces carry no
flux. Over 10,000 steps of a 128² box with an interior obstacle, the
maximum relative drift is **1.5e-7** (pressure-release) and **1.0e-7**
(rigid). This is float32 round-off.

### 4. Numerical dispersion (5.3.4)

The measured modal phase speed matches the von Neumann prediction
$\sin^2(\omega\Delta t/2) = \sigma^2\sum_a \sin^2(k_a\Delta x/2)$ to five
digits, along the axis and along the diagonal
(`physics_artifacts/dispersion.png`). Some useful figures:

- At 10 cells per wavelength the axis phase error is 1.5 %.
- At 6 cells per wavelength it is 3.7 %.
- The diagonal error is about half of that.

The compact 9-point scheme (`scheme="compact"`) cuts the error at 6 cells
per wavelength to 0.84 %, against 3.4 % for the standard scheme
(`tests/physics/test_compact_scheme.py`).

### 5. Green's functions (5.3.5)

A point source with a Ricker wavelet is compared against the analytic
free-field response:

- **2D:** $\partial_t[H(t-r)/(2\pi\sqrt{t^2-r^2})] * w$. At r = 40 cells,
  correlation is **0.995** and the amplitude ratio is 1.02.
- **3D:** $w(t-r)/(4\pi r)$. At r = 24 cells, correlation is **0.991** and
  the amplitude ratio is 1.00. The window ends before the first wall
  reflection arrives.

### 6. Absorbing boundaries (5.4.3)

Reflection is measured with the image-source method: the reflected trace
is (domain − free field), divided spectrally by the free-field wave at the
image distance. The band is 0.03–0.15 cycles per unit time (7–33 cells
per wavelength). Every layer is 24 cells thick. Values are the mean over
the band, with the worst frequency in brackets.

| incidence | Mur | sponge | **CPML** |
|---|---|---|---|
| 0° | −35 dB (−27) | −59 dB (−20) | **−102 dB (−94)** |
| 21° | −36 dB (−29) | −57 dB (−17) | **−101 dB (−97)** |
| 37° | −20 dB (−19) | −40 dB (−12) | **−89 dB (−87)** |
| 51° | −13 dB (−13) | −21 dB (−8) | **−69 dB (−68)** |
| 62° | −9 dB (−8) | −12 dB (−5) | **−51 dB (−49)** |

Only the CPML meets the Phase 5 target of −40 dB, and it does so at every
angle measured. It is the default for the gallery's anechoic scenes. The
sponge is kept for continuity and for the tensor engine, which does not
implement the CPML.

The CPML is stable in long runs: after 8,000 steps only the slowly decaying
2D wake remains. It also stays stable with rigid, impedance and soft cells
painted into the layer. Its differences across no-flux faces are masked in
the same way as the kernel's Laplacian (`tests/physics/test_cpml.py`).

Cost, on a 256² grid with a 16-cell layer:
- Python engine: no measurable difference from the sponge.
- Browser engine: about 60 % of the sponge's steps/s.
- 3D Python engine: about 3× per step at 96³. The layer update is
  vectorised NumPy, not numba.

### 7. Reverberation time vs Sabine and Eyring (5.5.3)

The test room is 160 × 110 cells (2D) with impedance outer walls, 14 rigid
scatterers to diffuse the field, and four probes. Each probe's T60 comes
from a Schroeder T20 fit. The absorption uses the 2D random-incidence
average $\alpha_d = \tfrac12\int(1-|R(\theta)|^2)\cos\theta\,d\theta$.
The 2D diffuse-field theory is $T_{60} = 6\ln10\,\pi S/(cL\,a)$.

| β | α (2D diffuse) | FDTD T60 | Sabine | Eyring |
|---|---|---|---|---|
| 0.05 | 0.25 | 5512 | 5653 | 4912 |
| 0.10 | 0.43 | 2996 | 3314 | 2542 |
| 0.20 | 0.66 | 1629 | 2147 | 1315 |

In every case the FDTD value lies between Eyring and Sabine. It sits close
to Sabine at low absorption, where Sabine is accurate, and moves towards
Eyring as absorption rises, as theory expects. The 3D browser twin
(`web/src/lab/twin.ts`) is checked the same way in
`web/tests/unit/twin.test.ts`.

## Batched simulation (5.7.3)

`tests/perf/bench_batched.py` runs B rooms of 128² for 300 steps, single
threaded, on the 4-core CI-class container:

| B | numba, one room after another | `TorchFDTD` batch (CPU) |
|---|---|---|
| 1 | 24,600 room-steps/s | 11,300 |
| 8 | 25,000 | 7,500 |
| 32 | 24,000 | 5,800 |

On a CPU, the fused numba kernel beats a batched tensor launch per room.
The tensor engine exists for gradients and for GPUs. On CPU, dataset
generation therefore scales across processes instead:
`generate_active_sensing.py --workers N` makes every random draw in the
main process, in the original order. It simulates rooms in N
single-thread workers and writes them in index order, so a seeded archive
is identical for any N (`tests/simulation/test_generate_workers.py`).
A single numba process with `prange` loses badly when cores are
oversubscribed. On this machine the same 128² step took 8 ms with four
threads while training shared the CPU, against 0.03 ms on one thread.

## Known limits

- Staircase geometry: curved walls are voxelised, so there are no
  cut-cell corrections.
- Impedance walls are locally reacting only, with one spring-damper
  branch per cell. Frequency dependence is a single high-pass corner.
- The tensor engine (`torch_engine.py`) covers p = 0, rigid, impedance
  and sponge. It rejects Mur and CPML explicitly.
- The GPU general kernel (`calculate_gpu.py`) takes the CPML term, and the
  layer update runs as CuPy array code. It has not been run: that needs
  an NVIDIA GPU (`tests/perf/check_simulate_gpu.py`, `docs/gpu.md`).
