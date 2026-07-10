# GPU backend gate + benchmark — 2026-07-10

Verification and measurement of the CuPy CUDA backend (Task 1.5,
commit `b5ad4e5`). Hardware: NVIDIA GeForce RTX 2070 SUPER (8 GB,
CC 7.5, driver 581.80, WDDM), 16-logical-core desktop CPU, CuPy
14.1.1 / CUDA 12 wheels.

## What

1. **Correctness gate** (`tests/perf/check_simulate_gpu.py`): identical
   scenarios stepped on `Simulate(backend="cpu")` and
   `Simulate(backend="gpu")`, requiring end-state field agreement
   within `atol=1e-4` / `rtol=1e-3`.
2. **Benchmark** (`tests/perf/bench_simulate_gpu.py`): median of 5
   runs of 500 steps per backend per grid, centred Ricker driver.
   GPU trials synchronise the device before and after the timed
   region (kernel launches are asynchronous; without the sync only
   enqueue time would be measured) and absorb NVRTC compilation in an
   untimed warm-up step.

## Why

The GPU backend re-implements the fused leap-frog kernels in CUDA; any
re-implementation of the numerics needs (a) proof of equivalence to
the gated CPU truth chain (`reference.npz` → CPU backend → GPU
backend) and (b) a measured, not assumed, speedup profile so users
know when the backend is worth selecting.

## Gate results

Scenarios exercise interior + on-wall drivers (Dirichlet-overwrite
ordering), interior obstacles uploaded via both APIs (per-cell
`set_obstacle` on CPU vs bulk `set_obstacle_mask` on GPU — their
equivalence is certified simultaneously), and `reset()` reproduction.

```
CHECK_GPU_HOST     pass=true                                    (p_host() returns numpy)
CHECK_GPU_GUARD    pass=true                                    (1D + gpu rejected)
CHECK_GPU_2D       pass=true  max_abs=1.065e-06  l2_rel=1.246e-06  (128², 300 steps)
CHECK_GPU_2D_RESET pass=true  max_abs=1.065e-06  l2_rel=1.246e-06
CHECK_GPU_3D       pass=true  max_abs=1.304e-07  l2_rel=5.894e-07  (40³, 100 steps)
CHECK_GPU_3D_RESET pass=true  max_abs=1.304e-07  l2_rel=5.894e-07
```

The ~1e-6 relative disagreement is float32 rounding-order noise
(numba-fastmath x86 FMA vs NVRTC-default CUDA FMA), two orders of
magnitude inside the gate tolerance. CPU gates re-run green with
byte-identical error values (`CHECK max_abs=7.752e-07`,
`CHECK_3D max_abs=1.043e-07`), confirming the CPU path is untouched.

## Benchmark results

`--steps 500 --trials 5`; times are medians for the full 500 steps.

| dims | grid | CPU (ms) | GPU (ms) | speedup |
| --- | --- | --- | --- | --- |
| 2D | 256² | 13.3 | 17.5 | 0.76× |
| 2D | 512² | 39.3 | 20.9 | 1.9× |
| 2D | 1024² | 141.3 | 21.1 | 6.7× |
| 2D | 2048² | 877.3 | 66.5 | **13.2×** |
| 3D | 64³ | 66.4 | 22.0 | 3.0× |
| 3D | 128³ | 561.4 | 43.4 | 13.0× |
| 3D | 200³ | 2412.2 | 144.9 | **16.7×** |

Interpretation (how to read the table): CPU time grows linearly with
cell count as expected for a bandwidth-bound stencil; GPU time is
nearly flat until ~1024² because each step pays a fixed ~40 µs of
kernel-launch overhead (≈3 launches/step under Windows WDDM) that
dominates the sub-millisecond stencil work. The crossover sits between
256² and 512²; beyond it the speedup climbs toward the ~18× DRAM
bandwidth ratio (≈450 GB/s device vs ≈25 GB/s effective host) and
reaches 16.7× at 200³. Practical guidance is unchanged from
`docs/gpu.md`: CPU for small interactive grids, GPU for ≥512² 2D and
all production 3D.

Known optimisation headroom (not currently needed): folding driver
injection into the stencil kernel or capturing the step as a CUDA
graph would cut the per-step launch count and lower the small-grid
floor.

## Test-data rationale

Grid sizes bracket the documented CPU operating points (256²–512² UI
grids, 64³–200³ volumetric runs) plus 1024²/2048² to expose the
asymptotic regime; 500 steps is long enough to amortise warm-up while
keeping the full sweep under two minutes. The gate's 128²/40³ scenarios
are the smallest grids whose interiors still contain obstacle blocks,
wall drivers, and several wave round-trips within 300/100 steps.

## Environment caveat

An unrelated user process (`mercury_strategies` eval, ~2 cores) was
active during the sweep. Its impact is bounded and visible: measured
CPU per-step times (0.079 ms at 512², 4.8 ms at 200³) are actually
*faster* than the documented idle-machine values (0.11 ms, 7.0 ms),
so the reported speedups are not inflated by contention.

## Runtime

| stage | time |
| --- | --- |
| GPU correctness gate | 41 s |
| CPU gates (re-check) | 18 s |
| benchmark sweep (7 grids × 2 backends × 5 trials) | 94 s |
| **total** | **≈ 2.5 min** |
