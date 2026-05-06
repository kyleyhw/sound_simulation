import warnings
from typing import List, Optional, Tuple

import numpy as np

from .calculate import Calculate, fused_leapfrog_step_2d
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
    ) -> None:
        self.grid_shape: Tuple[int, ...] = tuple(grid_shape)
        self.wavespeed: float = float(wavespeed)
        self.gridstep: float = float(gridstep)
        self.dims: int = len(self.grid_shape)

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
        self.p_prev: np.ndarray = np.zeros(self.grid_shape, dtype=np.float32)
        self.p: np.ndarray = np.zeros(self.grid_shape, dtype=np.float32)

        # Pre-allocated next-step buffer. Rotated each step rather than
        # allocated each step, eliminating per-step heap traffic. Kept private
        # to avoid expanding the public attribute surface tested by the gate.
        self._p_next: np.ndarray = np.zeros(self.grid_shape, dtype=np.float32)

        # Cached scalar coefficient for the fused 2D kernel:
        #   coeff = (c * dt / dx) ** 2
        # Equivalent to (c * dt) ** 2 * (1 / dx ** 2) used by the legacy path,
        # but precomputed once so the inner kernel takes a plain float32.
        self._coeff: np.float32 = np.float32(
            (self.wavespeed * self.timestep / self.gridstep) ** 2
        )

        # Pre-bind hot-path callables as plain attributes. The @njit dispatcher
        # is hashable and can be stored on the instance; assigning it to a
        # local at the top of step() collapses the chained name resolution
        # ``self._kernel(...)`` into a single load-fast local-variable call.
        self._kernel = fused_leapfrog_step_2d

        # Cached 2D dispatch predicate. Avoids re-evaluating ``self.dims == 2``
        # on every step; in 2D this is overwhelmingly the common path.
        self._is_2d: bool = (self.dims == 2)

        # Single-driver fast path bookkeeping. When exactly one driver exists
        # and its position is in-bounds, we precompute the integer tuple index
        # and skip the per-step Python ``zip`` + ``all`` predicate. Driver
        # positions are immutable on Driver (a frozen-style dataclass field
        # is set once); the position check is therefore safe to cache at
        # construction time. Out-of-bounds single drivers fall through to
        # the generic loop, where the original guard runs unchanged.
        self._fast_driver: Optional[Driver] = None
        self._fast_driver_pos: Optional[Tuple[int, ...]] = None
        if len(self.drivers) == 1:
            d0 = self.drivers[0]
            if all(0 <= pos < size for pos, size in zip(d0.position, self.grid_shape)):
                self._fast_driver = d0
                self._fast_driver_pos = tuple(d0.position)

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
        if self._is_2d:
            kernel = self._kernel
            p = self.p
            p_prev = self.p_prev
            p_next = self._p_next
            coeff = self._coeff

            # Fused 2D njit kernel: writes into p_next, also zeroing the four
            # edges to enforce the Dirichlet hard-wall BC. Operands are
            # passed positionally to skip any kwarg dict construction.
            kernel(p, p_prev, p_next, coeff)
        else:
            # Legacy generic-dimension path for 1D / 3D simulations.
            laplacian = laplacian_operator(grid=self.p, gridstep=self.gridstep)
            p_next = (
                2.0 * self.p
                - self.p_prev
                + (self.wavespeed * self.timestep) ** 2 * laplacian
            )
            # Hard-wall (Dirichlet, p=0) boundaries first so interior drivers
            # are not erased.
            set_edge_values(arr=p_next, value=0)
            p = self.p
            p_prev = self.p_prev

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
