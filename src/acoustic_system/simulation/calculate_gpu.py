"""CUDA (CuPy) counterparts of the fused FDTD leap-frog kernels.

This module is the GPU half of Task 1.5. It provides drop-in
replacements for :func:`acoustic_system.simulation.calculate.fused_leapfrog_step_2d`
and ``fused_leapfrog_step_3d`` that operate on CuPy device arrays, so a
``Simulate`` constructed with ``backend="gpu"`` runs the identical
mathematical update

    p_next = 2 p - p_prev + coeff * Laplacian(p),   coeff = (c dt / dx)^2

with the same fused structure (stencil + leap-frog combine + Dirichlet
edge zeroing in one pass) as the numba CPU kernels.

Design
------
- **Same call signature** as the CPU kernels: ``kernel(p, p_prev,
  p_next, coeff)``. ``Simulate.step()`` is therefore backend-agnostic;
  the only difference is which kernel is bound and which array module
  owns the buffers.
- **One thread per cell.** The x block axis maps to the innermost
  (unit-stride) array axis so global-memory reads/writes coalesce; a
  boundary thread writes the Dirichlet zero instead of the stencil.
  This mirrors the CPU kernel's loop nest (parallel outer axis,
  unit-stride inner axis) in CUDA's grid geometry.
- **Flat 64-bit indexing** (``long long``): a 3D grid of 2048^3 cells
  overflows 32-bit indices; 64-bit costs nothing measurable on this
  bandwidth-bound kernel.
- **No fast-math flags.** FMA contraction is already NVRTC's default
  (matching the CPU kernel's ``fastmath=True`` FMA fusion, which is its
  only effective transform here — the update has no reductions or
  divisions). The GPU and CPU results therefore differ only by
  float32 rounding order, verified to ~1e-6 relative L2 by
  ``tests/perf/check_simulate_gpu.py``.
- **Kernels are compiled lazily** (NVRTC, on first use) and cached at
  module level, so importing this module on a machine without CUDA is
  harmless until a GPU kernel is actually requested.

Transfers (Task 1.5.2)
----------------------
No host<->device transfer happens in the step path. The field buffers
live on the GPU for the lifetime of the ``Simulate``; per-step host
work is limited to kernel launches and the (scalar) driver values.
Readback is explicit — ``Simulate.p_host()`` or ``cupy.asnumpy`` — and
bulk geometry upload goes through ``Simulate.set_obstacle_mask``, one
transfer for the whole mask.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:  # pragma: no cover - exercised only on GPU-less installs
    cp = None  # ty: ignore[invalid-assignment]
    HAS_CUPY = False


def gpu_available() -> bool:
    """True iff CuPy is installed AND a usable CUDA device is present.

    ``import cupy`` succeeds on driver-less machines; the runtime error
    only surfaces at first device use, so probe the device count.
    """
    if not HAS_CUPY:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


# One thread per cell. x -> innermost (unit-stride) axis for coalesced
# access; boundary threads write the Dirichlet zero. `coeff` and shape
# come in as kernel scalars.
_KERNEL_2D_SRC = r"""
extern "C" __global__
void fused_leapfrog_2d(const float* __restrict__ p,
                       const float* __restrict__ p_prev,
                       float* __restrict__ p_next,
                       const float coeff,
                       const int ni, const int nj)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ni || j >= nj) return;

    const long long idx = (long long)i * nj + j;

    /* Dirichlet hard wall: zero the outer edges. Drivers are injected
       by the caller AFTER this kernel, so a driver on the wall still
       emits — same ordering contract as the CPU kernels. */
    if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1) {
        p_next[idx] = 0.0f;
        return;
    }

    const float lap = p[idx + nj] + p[idx - nj]
                    + p[idx + 1]  + p[idx - 1]
                    - 4.0f * p[idx];
    p_next[idx] = 2.0f * p[idx] - p_prev[idx] + coeff * lap;
}
"""

_KERNEL_3D_SRC = r"""
extern "C" __global__
void fused_leapfrog_3d(const float* __restrict__ p,
                       const float* __restrict__ p_prev,
                       float* __restrict__ p_next,
                       const float coeff,
                       const int ni, const int nj, const int nk)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= ni || j >= nj || k >= nk) return;

    const long long sj = nk;                 /* j-stride */
    const long long si = (long long)nj * nk; /* i-stride */
    const long long idx = (long long)i * si + (long long)j * sj + k;

    if (i == 0 || i == ni - 1 ||
        j == 0 || j == nj - 1 ||
        k == 0 || k == nk - 1) {
        p_next[idx] = 0.0f;
        return;
    }

    const float lap = p[idx + si] + p[idx - si]
                    + p[idx + sj] + p[idx - sj]
                    + p[idx + 1]  + p[idx - 1]
                    - 6.0f * p[idx];
    p_next[idx] = 2.0f * p[idx] - p_prev[idx] + coeff * lap;
}
"""

# Lazily-compiled kernel cache: NVRTC compilation costs ~100 ms per
# kernel; do it once per process, on first use, never at import.
_kernels: dict[str, Any] = {}


def _kernel(name: str, src: str) -> Any:
    k = _kernels.get(name)
    if k is None:
        if not HAS_CUPY:
            raise RuntimeError(
                "GPU kernels requested but CuPy is not installed. "
                "Install with `uv sync --extra dev --extra ml --extra gpu`."
            )
        k = cp.RawKernel(src, name)
        _kernels[name] = k
    return k


# Block shapes: x is the coalesced axis so keep it warp-wide (32).
# (32, 8) = 256 threads in 2D and (32, 8, 4) = 1024 in 3D are the
# standard occupancy sweet spots for a register-light stencil like
# this on CC 7.x; the kernel is bandwidth-bound, so the exact shape
# matters little once x >= 32.
_BLOCK_2D = (32, 8)
_BLOCK_3D = (32, 8, 4)


def fused_leapfrog_step_2d_gpu(
    p: Any,
    p_prev: Any,
    p_next: Any,
    coeff: np.float32,
) -> None:
    """GPU twin of ``fused_leapfrog_step_2d`` (arguments are CuPy arrays)."""
    ni, nj = p.shape
    bx, by = _BLOCK_2D
    grid = ((nj + bx - 1) // bx, (ni + by - 1) // by)
    _kernel("fused_leapfrog_2d", _KERNEL_2D_SRC)(
        grid,
        _BLOCK_2D,
        (p, p_prev, p_next, np.float32(coeff), np.int32(ni), np.int32(nj)),
    )


def fused_leapfrog_step_3d_gpu(
    p: Any,
    p_prev: Any,
    p_next: Any,
    coeff: np.float32,
) -> None:
    """GPU twin of ``fused_leapfrog_step_3d`` (arguments are CuPy arrays)."""
    ni, nj, nk = p.shape
    bx, by, bz = _BLOCK_3D
    grid = ((nk + bx - 1) // bx, (nj + by - 1) // by, (ni + bz - 1) // bz)
    _kernel("fused_leapfrog_3d", _KERNEL_3D_SRC)(
        grid,
        _BLOCK_3D,
        (p, p_prev, p_next, np.float32(coeff), np.int32(ni), np.int32(nj), np.int32(nk)),
    )
