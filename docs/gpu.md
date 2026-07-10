# `calculate_gpu.py` — CUDA (CuPy) Backend for the FDTD Engine

Task 1.5 of `PROJECT_PLAN.md`: GPU acceleration of the core simulation.
This document covers the design, the mathematics-preserving contracts,
the transfer strategy, and how to verify and benchmark the backend.

## 1. What it is

`Simulate(..., backend="gpu")` runs the identical leap-frog update

$$ p^{n+1} = 2p^n - p^{n-1} + \left(\frac{c\,\Delta t}{\Delta x}\right)^{2} \nabla_h^2\, p^n $$

(with the same 5-point / 7-point central Laplacian $\nabla_h^2$ as the
CPU kernels) on a CUDA device via CuPy `RawKernel`s. The default
`backend="cpu"` is byte-identical to the pre-GPU engine — the numba
fast path and its evolve-harness contract are untouched, and
`check_simulate.py` passes with unchanged error values.

## 2. Design

The controlling constraint was: **`Simulate.step()` must stay shared
between backends**, so every ordering contract the engine documents
(Dirichlet walls zeroed in-kernel → interior obstacle scrub → driver
injection, in that order) is enforced by one code path rather than
maintained twice. Three things make that possible:

1. **Identical kernel signatures.** `fused_leapfrog_step_{2d,3d}_gpu(p,
   p_prev, p_next, coeff)` mirror the numba kernels exactly; `__init__`
   binds one or the other to `self._kernel` once.
2. **CuPy mirrors the NumPy surface.** The obstacle scrub
   (`p_next[mask] = 0`) and driver injection (`p_next[idx] += value`)
   are the same source lines on device arrays.
3. **Backend-owned allocation.** `__init__` allocates `p`, `p_prev`,
   `_p_next`, and `obstacle_mask` through `xp ∈ {numpy, cupy}` chosen
   by the `backend` argument.

Inside the CUDA kernels:

- **One thread per cell**; the x block axis maps to the innermost
  (unit-stride) array axis so global-memory access coalesces — the GPU
  analogue of the CPU kernels' "outer axis parallel, inner axis
  unit-stride serial" loop nest. Block shapes are (32, 8) in 2D and
  (32, 8, 4) in 3D: warp-wide in x, 256/1024 threads total, the
  standard shape for a register-light, bandwidth-bound stencil [[1]](#ref-cuda-guide).
- **Boundary threads write the Dirichlet zero** instead of the stencil,
  fusing the wall enforcement into the same pass (as on CPU).
- **Flat 64-bit indexing** (`long long`) so 3D grids cannot overflow
  32-bit linear indices.
- **No `--use_fast_math`.** FMA contraction is NVRTC's default and is
  the only fast-math transform that affects this kernel (no reductions,
  no divisions) — matching what `fastmath=True` does for the numba
  kernels. Measured GPU-vs-CPU disagreement is ~1e-6 relative L2 over
  hundreds of steps, i.e. pure float32 rounding-order noise.
- **Lazy NVRTC compilation** (~100 ms, once per process, cached at
  module level) so importing the package on a GPU-less machine costs
  nothing and fails only if the GPU backend is actually requested.

## 3. Transfer strategy (Task 1.5.2)

The step path performs **zero host↔device transfers**: the three field
buffers rotate device pointers exactly as the CPU path rotates host
pointers, and the per-step host work is limited to kernel launches plus
the scalar waveform evaluation for each driver. Data crosses the bus
only at three explicit, user-controlled points:

| operation | direction | cost model |
| --- | --- | --- |
| `set_obstacle_mask(mask)` | host → device | one bulk transfer of the whole boolean mask; the efficient path for scene setup (per-cell `set_obstacle` still works but is one device write per cell) |
| `p_host()` | device → host | one transfer of the field; call at output cadence (sensor sampling, wire emission, archiving), never per step |
| construction / `reset()` | device-side only | `zeros` allocation / `fill(0)` — no transfer |

This is the standard FDTD-on-GPU discipline: the arithmetic intensity
of the stencil (~0.25 flop/byte) makes the kernel bandwidth-bound, so
any recurring PCIe transfer (~25 GB/s) against device DRAM bandwidth
(~450 GB/s on the RTX 2070 SUPER) would immediately dominate.

## 4. When to use which backend

- **Small grids (≲256² in 2D)**: CPU. Each GPU step costs a fixed
  ~10–20 µs of launch overhead (Windows WDDM); at 256² the numba kernel
  finishes in ~50 µs, so there is little to win.
- **Large 2D (≥512²) and all production 3D**: GPU. The stencil's
  byte-traffic scales with cell count while launch overhead stays
  fixed, so speedup grows with grid size (see the benchmark table in
  `tests/reports/gpu_backend_2026_07_10.md`).
- **The web UI stays on CPU** by default — its grids are small and the
  per-frame `p_host()` readback would serialise the device queue.
- The deployed sensing system targets stock laptops (no discrete GPU);
  the GPU backend exists to accelerate simulation-side dataset
  generation and future large-scene work, not deployment inference.

## 5. Verification and benchmarking

```bash
uv sync --extra dev --extra ml --extra gpu   # cupy-cuda12x; needs NVIDIA driver >= 525

uv run python tests/perf/check_simulate_gpu.py
# CHECK_GPU_{HOST,GUARD,2D,2D_RESET,3D,3D_RESET} — GPU vs CPU end-state
# equality within atol=1e-4 / rtol=1e-3 on scenarios exercising interior
# + wall drivers, obstacles (bulk & per-cell upload), and reset().

uv run python tests/perf/bench_simulate_gpu.py --steps 500 --trials 5
# BENCH_GPU dims=<d> grid=<n> cpu_ms=<..> gpu_ms=<..> speedup=<..>
```

The GPU gate chains to the same ground truth as the CPU gates: the CPU
backend is certified against `reference.npz` (2D) and the scipy
reference (3D), and the GPU backend is certified against the CPU
backend. Any change to the CUDA kernels must keep
`check_simulate_gpu.py` green and be measured with
`bench_simulate_gpu.py` medians, mirroring the evolve-harness rules for
the CPU kernels.

## References

<span id="ref-cuda-guide">[1]</span> NVIDIA. *CUDA C++ Programming Guide* — Memory coalescing and thread-block sizing. [Link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
