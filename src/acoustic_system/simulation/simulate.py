import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .calculate import Calculate, fused_leapfrog_step_2d, fused_leapfrog_step_3d
from .setup import Driver, Sensor
from .utils import set_edge_values

laplacian_operator = Calculate().laplacian_operator


class Simulate:
    """Stateful FDTD simulation advanced one step at a time.

    Solves the acoustic wave equation $\\partial_t^2 p = c^2 \\nabla^2 p$
    with the explicit second-order leap-frog update
        p_{n+1} = 2 p_n - p_{n-1} + (c \\Delta t)^2 \\nabla^2 p_n
    on a regular Cartesian grid of arbitrary dimension.

    Stability requires the Courant number to satisfy
        C = c \\Delta t / \\Delta x \\le 1 / \\sqrt{d}
    where d is the spatial dimensionality. A warning is raised when violated.

    The 2D path uses a numba ``@njit`` fused stencil kernel
    (``fused_leapfrog_step_2d``); 1D and 3D paths fall back to the legacy
    ``scipy.ndimage.laplace`` based code so the public API behaves identically
    for all supported dimensionalities.

    Backends (Task 1.5)
    -------------------
    ``backend="cpu"`` (default) is the numba/scipy engine described above,
    unchanged. ``backend="gpu"`` (2D/3D only, requires the ``gpu`` extra and
    a CUDA device) binds the CuPy ``RawKernel`` twins from
    ``calculate_gpu.py`` and allocates ``p``, ``p_prev``, ``_p_next`` and
    ``obstacle_mask`` as device arrays. ``step()`` is shared verbatim between
    backends — the kernels have identical signatures and CuPy mirrors the
    NumPy operations used for the obstacle scrub and driver injection — so
    the ordering contracts (walls, then obstacles, then drivers) hold on
    both. No transfer occurs in the step path; use ``p_host()`` for
    readback and ``set_obstacle_mask()`` for bulk geometry upload.

    Round-3 cand-c performance technique
    -------------------------------------
    Per-step Python overhead is minimized by:

    1. Pre-binding the @njit dispatcher and the cached coefficient as private
       attributes in ``__init__`` (``self._kernel``, ``self._coeff``), so the
       hot ``step()`` body uses local-variable bindings rather than chained
       attribute lookups. The CPython attribute-lookup path is ~50-100 ns per
       resolution; eliminating ~6 of them per step (kernel, p, p_prev,
       _p_next, coeff, drivers) saves on the order of 0.5 microsecond.

    2. Caching whether the simulation is 2D (``self._is_2d``) once at
       construction time. The hot 2D branch then takes a single boolean
       check rather than ``len(self.grid_shape)``-style introspection.

    3. Special-casing the single-driver path. The vast majority of FDTD
       runs use exactly one excitation, so ``len(self.drivers) == 1`` is
       hot. In that branch we avoid the for-loop, the per-driver
       ``zip``-bounds check is folded into precomputed integer indices,
       and the in-bounds check itself is precomputed once at construction
       (since driver positions are static for the lifetime of the run).
       Multi-driver and zero-driver branches fall back to the generic
       loop unchanged for behavioural compatibility.

    Numerics are unchanged: same kernel, same operand ordering, same
    leap-frog combine, same hard-wall-before-driver ordering, same buffer
    rotation. Only the surrounding Python plumbing is leaner.

    Interior obstacles
    ------------------
    A boolean ``obstacle_mask`` of the same shape as the field marks cells
    that act as rigid Dirichlet walls inside the domain. After the stencil
    pass and before driver injection, ``step()`` zeroes ``p_next`` at the
    masked cells; the cached ``_has_obstacles`` flag lets the hot loop
    skip this work entirely when the mask is empty, preserving bit-identical
    behaviour against the no-obstacle reference. Drivers placed on
    obstacle cells still emit (just like drivers placed on the outer wall),
    which matches the boundary semantics already encoded in the kernel.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...] = (200, 200),
        drivers: Optional[List[Driver]] = None,
        sensors: Optional[List[Sensor]] = None,
        wavespeed: float = 1.0,
        timestep: Optional[float] = None,
        gridstep: float = 1.0,
        courant: float = 0.5,
        backend: str = "cpu",
    ) -> None:
        self.grid_shape: Tuple[int, ...] = tuple(grid_shape)
        self.wavespeed: float = float(wavespeed)
        self.gridstep: float = float(gridstep)
        self.dims: int = len(self.grid_shape)

        # Backend selection (Task 1.5). "cpu" (default) is the numba fast
        # path / scipy fallback, byte-identical to the pre-GPU engine.
        # "gpu" keeps all three field buffers resident on the CUDA device
        # as CuPy arrays for the lifetime of the object — the step path
        # performs no host<->device transfer at all; readback is explicit
        # via p_host(). Restricted to 2D/3D because only the fused kernels
        # have GPU twins (the 1D scipy fallback has no GPU value).
        self.backend: str = str(backend)
        if self.backend not in ("cpu", "gpu"):
            raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")
        if self.backend == "gpu":
            if self.dims not in (2, 3):
                raise ValueError("backend='gpu' supports 2D and 3D grids only")
            from . import calculate_gpu

            if not calculate_gpu.gpu_available():
                raise RuntimeError(
                    "backend='gpu' requested but no usable CUDA device found. "
                    "Install the extra (`uv sync --extra dev --extra ml --extra gpu`) "
                    "and check `nvidia-smi`."
                )
            self._xp = calculate_gpu.cp
        else:
            self._xp = np
        xp = self._xp

        # If no timestep provided, derive one from the requested Courant number.
        # The CFL limit in d dimensions is C_max = 1/sqrt(d); we stay strictly under it.
        cfl_limit = 1.0 / np.sqrt(self.dims)
        if timestep is None:
            chosen_courant = min(courant, 0.95 * cfl_limit)
            self.timestep = chosen_courant * self.gridstep / self.wavespeed
        else:
            self.timestep = float(timestep)
            actual_courant = self.wavespeed * self.timestep / self.gridstep
            if actual_courant >= cfl_limit:
                warnings.warn(
                    f"CFL violated: Courant={actual_courant:.3f} >= 1/sqrt({self.dims})={cfl_limit:.3f}. "
                    f"Simulation will be unstable.",
                    RuntimeWarning,
                )

        self.drivers: List[Driver] = list(drivers) if drivers is not None else []
        self.sensors: List[Sensor] = list(sensors) if sensors is not None else []

        self.time: float = 0.0
        self.step_count: int = 0
        # xp is numpy on the CPU backend (allocations identical to the
        # pre-GPU engine) and cupy on the GPU backend (device-resident).
        self.p_prev: np.ndarray = xp.zeros(self.grid_shape, dtype=np.float32)
        self.p: np.ndarray = xp.zeros(self.grid_shape, dtype=np.float32)

        # Interior Dirichlet obstacles: a boolean mask the same shape as the
        # field. Cells flagged True are forced to p=0 each step before driver
        # injection, which gives them the same rigid-wall semantics as the
        # outer boundary. The mask defaults to all-False so behaviour is
        # bit-identical to the no-obstacle case (and check_simulate.py keeps
        # passing without regenerating the reference). The ``_has_obstacles``
        # flag is an O(1) hot-loop guard: ``mask.any()`` would re-scan the
        # whole grid every step, so we cache it and update it only on mutation.
        self.obstacle_mask: np.ndarray = xp.zeros(self.grid_shape, dtype=bool)
        self._has_obstacles: bool = False

        # Pre-allocated next-step buffer. Rotated each step rather than
        # allocated each step, eliminating per-step heap traffic. Kept private
        # to avoid expanding the public attribute surface tested by the gate.
        self._p_next: np.ndarray = xp.zeros(self.grid_shape, dtype=np.float32)

        # Cached scalar coefficient for the fused 2D kernel:
        #   coeff = (c * dt / dx) ** 2
        # Equivalent to (c * dt) ** 2 * (1 / dx ** 2) used by the legacy path,
        # but precomputed once so the inner kernel takes a plain float32.
        self._coeff: np.float32 = np.float32((self.wavespeed * self.timestep / self.gridstep) ** 2)

        # Pre-bind the active fused kernel as a plain attribute. Picked once
        # at construction by dimensionality:
        #   * 2D -> fused_leapfrog_step_2d (5-point stencil)
        #   * 3D -> fused_leapfrog_step_3d (7-point stencil)
        #   * other -> None (1D falls through to the legacy scipy path)
        # Assigning to a local at the top of step() collapses the chained
        # ``self._kernel(...)`` lookup into a single load-fast call.
        if self.backend == "gpu":
            # GPU twins share the CPU kernels' exact call signature
            # (p, p_prev, p_next, coeff), so step() below is backend-
            # agnostic: same buffer rotation, same obstacle scrub, same
            # driver-injection ordering — on device arrays.
            from . import calculate_gpu

            if self.dims == 2:
                self._kernel = calculate_gpu.fused_leapfrog_step_2d_gpu
            else:
                self._kernel = calculate_gpu.fused_leapfrog_step_3d_gpu
        elif self.dims == 2:
            self._kernel = fused_leapfrog_step_2d
        elif self.dims == 3:
            self._kernel = fused_leapfrog_step_3d
        else:
            self._kernel = None

        # Cached fast-path predicate. The 2D AND 3D paths share the same
        # surrounding plumbing — they only differ in which @njit kernel
        # they call, which is already encoded in self._kernel. The 1D path
        # still uses the scipy.ndimage.laplace fallback.
        self._fast_path: bool = self._kernel is not None
        # Kept for backwards compatibility with any external code that
        # referenced ``_is_2d`` directly. Value remains correct.
        self._is_2d: bool = self.dims == 2

        # Single-driver fast path bookkeeping. When exactly one driver exists
        # and its position is in-bounds, we precompute the integer tuple index
        # and skip the per-step Python ``zip`` + ``all`` predicate. The cache
        # is now refreshed on every driver mutation rather than only at
        # construction, so live add/remove during a UI session still hits the
        # fast path. Out-of-bounds single drivers fall through to the generic
        # loop, where the original guard runs unchanged.
        self._fast_driver: Optional[Driver] = None
        self._fast_driver_pos: Optional[Tuple[int, ...]] = None
        self._refresh_driver_cache()

    # ----- Driver mutation ---------------------------------------------- #

    def _refresh_driver_cache(self) -> None:
        """Recompute the single-driver fast-path cache.

        Called by ``__init__`` and by every driver-list mutation method.
        Cheap: one length check and at most one bounds check on the position.
        """
        self._fast_driver = None
        self._fast_driver_pos = None
        if len(self.drivers) == 1:
            d0 = self.drivers[0]
            if all(0 <= pos < size for pos, size in zip(d0.position, self.grid_shape)):
                self._fast_driver = d0
                self._fast_driver_pos = tuple(d0.position)

    def add_driver(self, driver: Driver) -> None:
        """Append a driver and refresh the fast-path cache."""
        self.drivers.append(driver)
        self._refresh_driver_cache()

    def remove_driver(self, index: int) -> None:
        """Remove the driver at ``index`` (raises IndexError if invalid)."""
        del self.drivers[index]
        self._refresh_driver_cache()

    def set_drivers(self, drivers: Sequence[Driver]) -> None:
        """Replace the entire driver list."""
        self.drivers = list(drivers)
        self._refresh_driver_cache()

    # ----- Obstacle mutation -------------------------------------------- #

    def set_obstacle(
        self,
        positions: Iterable[Tuple[int, ...]],
        value: bool = True,
    ) -> None:
        """Mark (``value=True``) or clear (``value=False``) obstacle cells.

        Out-of-bounds positions are silently ignored — the UI sends grid
        indices from a downsampled view, and rounding can put a stray
        coordinate one cell past the edge.

        When marking new obstacles, also zero the existing field at those
        cells in ``p``, ``p_prev`` and ``_p_next``. Otherwise stale pressure
        from before the cell was an obstacle would leak into one final
        stencil read before the next ``step()`` scrubs it.
        """
        shape = self.grid_shape
        v = bool(value)
        for pos in positions:
            tpos = tuple(int(c) for c in pos)
            if len(tpos) != len(shape):
                continue
            if not all(0 <= c < s for c, s in zip(tpos, shape)):
                continue
            self.obstacle_mask[tpos] = v
            if v:
                self.p[tpos] = 0.0
                self.p_prev[tpos] = 0.0
                self._p_next[tpos] = 0.0
        self._has_obstacles = bool(self.obstacle_mask.any())

    def set_obstacle_mask(self, mask: np.ndarray) -> None:
        """Replace the whole obstacle mask in one operation.

        Accepts any array-like of shape ``grid_shape``; it is coerced to
        the backend's array type, so on the GPU backend a NumPy mask is
        uploaded in a single host->device transfer (the efficient bulk
        path — per-cell ``set_obstacle`` calls would each be a device
        write). Field values at newly masked cells are zeroed for the
        same stale-pressure reason documented on ``set_obstacle``.
        """
        xp = self._xp
        m = xp.asarray(mask, dtype=bool)
        if m.shape != self.grid_shape:
            raise ValueError(f"mask shape {m.shape} != grid shape {self.grid_shape}")
        self.obstacle_mask = m
        zero = np.float32(0.0)
        self.p[m] = zero
        self.p_prev[m] = zero
        self._p_next[m] = zero
        self._has_obstacles = bool(m.any())

    def clear_obstacles(self) -> None:
        """Remove every obstacle. Field is left untouched."""
        self.obstacle_mask.fill(False)
        self._has_obstacles = False

    def p_host(self) -> np.ndarray:
        """Current pressure field as a NumPy array.

        CPU backend: returns the live ``self.p`` (no copy). GPU backend:
        one device->host transfer returning a fresh host copy. This is
        the intended readback point for backend-agnostic consumers —
        per-step device readbacks are exactly the transfer pattern the
        GPU backend exists to avoid, so call it only at output cadence
        (sensor sampling, wire emission, archiving).
        """
        if self.backend == "gpu":
            from . import calculate_gpu

            return calculate_gpu.cp.asnumpy(self.p)
        return self.p

    def reset(self) -> None:
        """Zero pressure fields and the clock; preserve geometry and drivers."""
        self.p_prev.fill(0.0)
        self.p.fill(0.0)
        # Also zero the rotation buffer so a stale slot cannot leak into the
        # next call after the three-way pointer rotation in step().
        self._p_next.fill(0.0)
        self.time = 0.0
        self.step_count = 0

    def step(self) -> None:
        """Advance the simulation by a single timestep.

        Hot loop: the body below is intentionally written to expose every
        repeated load to the CPython peephole optimiser as a STORE_FAST /
        LOAD_FAST pair against a function-local. ``self.X`` lookups (which
        cost a LOAD_ATTR + dict probe each) are pulled out once at the top.
        """
        # Bind every per-call value used more than once as a local.
        # This converts O(N_uses) attribute lookups to O(N_uses) of cheaper
        # local-variable loads after a fixed O(N_distinct) attribute snapshot.
        # Bind the kernel once. The ``is not None`` test both selects the
        # fast path AND narrows self._kernel's type for ty (the 1D
        # fallback below leaves self._kernel = None).
        kernel = self._kernel
        if kernel is not None:
            # 2D and 3D both go through a fused @njit kernel. Which one
            # was bound to self._kernel depends on dimensionality (set
            # once in __init__); the surrounding plumbing — buffer
            # rotation, obstacle scrub, driver injection — is identical.
            p = self.p
            p_prev = self.p_prev
            p_next = self._p_next
            coeff = self._coeff

            # Fused njit kernel: writes p_next, also zeroing the outer
            # faces to enforce the Dirichlet hard-wall BC. Operands are
            # passed positionally to skip any kwarg dict construction.
            kernel(p, p_prev, p_next, coeff)
        else:
            # Legacy 1D path: scipy.ndimage.laplace fallback. Kept for
            # behavioural compatibility with any caller that builds a 1D
            # Simulate; the fused kernels are 2D and 3D only.
            laplacian = laplacian_operator(grid=self.p, gridstep=self.gridstep)
            p_next = 2.0 * self.p - self.p_prev + (self.wavespeed * self.timestep) ** 2 * laplacian
            # Hard-wall (Dirichlet, p=0) boundaries first so interior drivers
            # are not erased.
            set_edge_values(arr=p_next, value=0)
            p = self.p
            p_prev = self.p_prev

        # Interior Dirichlet obstacle scrub. Mirrors the outer wall: zero
        # the masked cells in p_next AFTER the stencil pass but BEFORE driver
        # injection, so a driver placed on an obstacle cell still emits
        # (matches the documented boundary semantics where a driver at the
        # edge overwrites the wall). Guarded by the cached flag so the
        # no-obstacle path is bit-identical to the pre-obstacle code and
        # check_simulate.py keeps matching reference.npz.
        if self._has_obstacles:
            p_next[self.obstacle_mask] = np.float32(0.0)

        # Driver injection happens after the boundary zeroing in BOTH paths
        # (the 2D kernel zeros edges internally). This ordering matters and
        # must not be changed; drivers placed on a boundary are intentionally
        # allowed to overwrite the wall.
        time = self.time
        fast_driver = self._fast_driver
        if fast_driver is not None:
            # Single-driver fast path: position validated and tuple-cached at
            # construction time, so we skip the per-step ``zip``/``all``/
            # ``tuple(...)`` chain and go straight to the indexed write.
            p_next[self._fast_driver_pos] += fast_driver.get_value(time)
        else:
            # Generic path: zero or many drivers, or out-of-bounds single
            # driver. Behaviour is bit-identical to the original code.
            grid_shape = self.grid_shape
            for driver in self.drivers:
                value = driver.get_value(time)
                if all(0 <= pos < size for pos, size in zip(driver.position, grid_shape)):
                    p_next[tuple(driver.position)] += value

        # Three-way pointer rotation: p_prev <- p, p <- p_next, _p_next <- old p_prev.
        # The old p_prev buffer becomes the new scratch pad for the next step,
        # so no array is allocated in the hot path. We use the locals captured
        # above (rather than re-reading self) for the source half of the swap.
        self.p_prev = p
        self.p = p_next
        self._p_next = p_prev

        self.time = time + self.timestep
        self.step_count += 1
