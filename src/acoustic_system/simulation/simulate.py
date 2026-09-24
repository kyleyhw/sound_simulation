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

    2D and 3D use numba ``@njit`` fused stencil kernels
    (``fused_leapfrog_step_2d`` / ``fused_leapfrog_step_3d``); other
    dimensionalities (1D, N-D) fall back to the legacy
    ``scipy.ndimage.laplace`` based code so the public API behaves
    identically for all supported dimensionalities.

    Source convention: a driver's value evaluated at time $t_n$ is *added*
    to $p^{n+1}$ (soft source), after the wall and obstacle zeroing.

    ``drivers`` is a read-only tuple; mutate it only through
    ``add_driver`` / ``remove_driver`` / ``set_drivers`` (or assign a new
    sequence to ``sim.drivers``) so the single-driver fast-path cache can
    never go stale. ``Driver`` is frozen for the same reason.

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
    that act as pressure-release (Dirichlet, p = 0) walls inside the domain. After the stencil
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
        boundary: str = "soft",
        boundary_beta: float = 1.0,
        sponge_cells: int = 24,
    ) -> None:
        self.grid_shape: Tuple[int, ...] = tuple(int(n) for n in grid_shape)
        self.wavespeed: float = float(wavespeed)
        self.gridstep: float = float(gridstep)
        self.dims: int = len(self.grid_shape)
        if self.dims == 0:
            raise ValueError("grid_shape must have at least one axis")
        if not (self.wavespeed > 0 and np.isfinite(self.wavespeed)):
            raise ValueError(f"wavespeed must be positive and finite, got {wavespeed!r}")
        if not (self.gridstep > 0 and np.isfinite(self.gridstep)):
            raise ValueError(f"gridstep must be positive and finite, got {gridstep!r}")
        if timestep is None and not (courant > 0):
            raise ValueError(f"courant must be positive, got {courant!r}")
        if timestep is not None and not (float(timestep) > 0):
            raise ValueError(f"timestep must be positive, got {timestep!r}")

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
            chosen_courant = min(float(courant), 0.95 * float(cfl_limit))
            # Plain Python float: a numpy float64 here would silently promote
            # the float32 field buffers on the non-fused (1D / N-D) path.
            self.timestep = float(chosen_courant * self.gridstep / self.wavespeed)
        else:
            self.timestep = float(timestep)
            actual_courant = self.wavespeed * self.timestep / self.gridstep
            # sigma <= 1/sqrt(d) is stable (the bound itself is marginally
            # stable), so warn strictly above it, with a float tolerance.
            if actual_courant > cfl_limit * (1.0 + 1e-9):
                warnings.warn(
                    f"CFL violated: Courant={actual_courant:.3f} > 1/sqrt({self.dims})={cfl_limit:.3f}. "
                    f"Simulation will be unstable.",
                    RuntimeWarning,
                )

        self._drivers: List[Driver] = []
        for d in drivers or []:
            self._validate_driver(d)
            self._drivers.append(d)
        # Sensors are a caller-owned container (e.g. main.py records into
        # them); step() does not sample them. Use dataset.run_with_sensors
        # for streaming sensor recording.
        self.sensors: List[Sensor] = list(sensors) if sensors is not None else []

        self.time: float = 0.0
        self.step_count: int = 0
        # xp is numpy on the CPU backend (allocations identical to the
        # pre-GPU engine) and cupy on the GPU backend (device-resident).
        self.p_prev: np.ndarray = xp.zeros(self.grid_shape, dtype=np.float32)
        self.p: np.ndarray = xp.zeros(self.grid_shape, dtype=np.float32)

        # Interior Dirichlet obstacles: a boolean mask the same shape as the
        # field. Cells flagged True are forced to p=0 each step before driver
        # injection, which gives them the same pressure-release (p = 0) semantics as the
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

        # General-path physics (Phase 5, see physics.py): rigid/impedance
        # walls via a per-cell material map, absorbing outer boundaries,
        # and a relative wave-speed map. Any non-default setting routes
        # step() through the general kernel; the default leaves the fast
        # path (and reference.npz) untouched.
        from .physics import OUTER_KINDS

        if boundary == "pml":  # accepted alias: the graded absorbing layer
            boundary = "sponge"
        if boundary not in OUTER_KINDS:
            raise ValueError(f"boundary must be one of {OUTER_KINDS}, got {boundary!r}")
        self.boundary: str = boundary
        self.boundary_beta: float = float(boundary_beta)
        self.sponge_cells: int = int(sponge_cells)
        self.material: np.ndarray = np.zeros(self.grid_shape, dtype=np.uint8)
        self.speed_map: Optional[np.ndarray] = None
        self._wall_v = np.zeros(self.grid_shape, dtype=np.float32)
        self._wall_x = np.zeros(self.grid_shape, dtype=np.float32)
        self._gcoef = None
        self._general: bool = False
        self._refresh_general()

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

    @property
    def drivers(self) -> Tuple[Driver, ...]:
        """Read-only view of the drivers. Mutate via add/remove/set_drivers."""
        return tuple(self._drivers)

    @drivers.setter
    def drivers(self, drivers: Sequence[Driver]) -> None:
        self.set_drivers(drivers)

    def _validate_driver(self, driver: Driver) -> None:
        if len(driver.position) != self.dims:
            raise ValueError(
                f"driver position {driver.position} has {len(driver.position)} "
                f"coordinates but the grid is {self.dims}D"
            )
        aliased = getattr(driver.waveform, "aliased_energy_fraction", None)
        if callable(aliased):
            frac = float(aliased(self.timestep))
            if frac > 0.01:
                warnings.warn(
                    f"{frac:.0%} of the driver waveform's energy lies above the "
                    f"simulation Nyquist 1/(2 dt) = {0.5 / self.timestep:.4g} and will "
                    f"alias. Use waveform.resampled_for(sim.timestep).",
                    RuntimeWarning,
                )

    def _refresh_driver_cache(self) -> None:
        """Recompute the single-driver fast-path cache.

        Called by ``__init__`` and by every driver-list mutation method.
        Cheap: one length check and at most one bounds check on the position.
        """
        self._fast_driver = None
        self._fast_driver_pos = None
        if len(self._drivers) == 1:
            d0 = self._drivers[0]
            if all(0 <= pos < size for pos, size in zip(d0.position, self.grid_shape)):
                self._fast_driver = d0
                self._fast_driver_pos = tuple(d0.position)

    def add_driver(self, driver: Driver) -> None:
        """Append a driver and refresh the fast-path cache."""
        self._validate_driver(driver)
        self._drivers.append(driver)
        self._refresh_driver_cache()

    def remove_driver(self, index: int) -> None:
        """Remove the driver at ``index`` (raises IndexError if invalid)."""
        del self._drivers[index]
        self._refresh_driver_cache()

    def set_drivers(self, drivers: Sequence[Driver]) -> None:
        """Replace the entire driver list."""
        new = list(drivers)
        for d in new:
            self._validate_driver(d)
        self._drivers = new
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
        self._gcoef = None

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
        # Always copy: asarray would alias a caller's bool array, so later
        # caller edits (or our clear_obstacles) would silently cross over and
        # desynchronise the cached _has_obstacles flag.
        m = xp.array(mask, dtype=bool, copy=True)
        if m.shape != self.grid_shape:
            raise ValueError(f"mask shape {m.shape} != grid shape {self.grid_shape}")
        self.obstacle_mask = m
        zero = np.float32(0.0)
        self.p[m] = zero
        self.p_prev[m] = zero
        self._p_next[m] = zero
        self._has_obstacles = bool(m.any())
        self._gcoef = None

    # ----- General-path geometry (Phase 5) ------------------------------ #

    def _refresh_general(self) -> None:
        general = (
            self.boundary != "soft" or bool((self.material > 1).any()) or self.speed_map is not None
        )
        if general and (self.dims not in (2, 3) or self.backend != "cpu"):
            raise NotImplementedError(
                "rigid/impedance walls, absorbing boundaries and c(x) need a 2D/3D CPU Simulate"
            )
        self._general = general
        self._gcoef = None  # rebuilt lazily

    def set_material(self, positions: Iterable[Tuple[int, ...]], material_id: int) -> None:
        """Assign a material (``physics.MATERIALS`` id; 0 = air) to cells.

        Id 1 (pressure-release) is equivalent to ``set_obstacle``. Ids >= 2
        (rigid, plaster, wood, absorber, curtain) enable the general path.
        """
        mid = int(material_id)
        cells = [tuple(int(c) for c in pos) for pos in positions]
        cells = [
            c
            for c in cells
            if len(c) == self.dims and all(0 <= v < n for v, n in zip(c, self.grid_shape))
        ]
        for c in cells:
            self.material[c] = mid
            if mid != 0:
                self.p[c] = 0.0
                self.p_prev[c] = 0.0
                self._p_next[c] = 0.0
        self._refresh_general()

    def set_material_map(self, material: np.ndarray) -> None:
        """Replace the whole material map (uint8 ids, same shape as the grid)."""
        m = np.array(material, dtype=np.uint8, copy=True)
        if m.shape != self.grid_shape:
            raise ValueError(f"material shape {m.shape} != grid shape {self.grid_shape}")
        self.material = m
        walls = m != 0
        self.p[walls] = 0.0
        self.p_prev[walls] = 0.0
        self._p_next[walls] = 0.0
        self._refresh_general()

    def set_speed_map(self, speed: Optional[np.ndarray]) -> None:
        """Relative wave speed c(x)/c (None = uniform). Checks the CFL bound."""
        if speed is None:
            self.speed_map = None
        else:
            r = np.array(speed, dtype=np.float32, copy=True)
            if r.shape != self.grid_shape or not np.all(r > 0):
                raise ValueError("speed map must match the grid and be positive")
            sigma = float(r.max()) * self.wavespeed * self.timestep / self.gridstep
            if sigma > 1.0 / np.sqrt(self.dims) * (1.0 + 1e-9):
                warnings.warn(
                    f"CFL violated by the speed map: max local Courant {sigma:.3f}",
                    RuntimeWarning,
                )
            self.speed_map = r
        self._refresh_general()

    def _effective_material(self) -> np.ndarray:
        m = self.material
        if self._has_obstacles:
            m = np.where((m == 0) & self.obstacle_mask, np.uint8(1), m)
        return m

    def _run_general(self, p: np.ndarray, p_prev: np.ndarray, p_next: np.ndarray) -> None:
        from .physics import build_coefficients, general_step_2d, general_step_3d, mur_edges

        if self._gcoef is None:
            self._gcoef = build_coefficients(
                self._effective_material(),
                self.speed_map,
                self.boundary,
                self.boundary_beta,
                self.sponge_cells,
                float(self._coeff),
                self.wavespeed,
                self.gridstep,
                self.timestep,
            )
        g = self._gcoef
        step = general_step_2d if self.dims == 2 else general_step_3d
        step(
            p, p_prev, p_next, g.active, g.k_air, g.c2, g.s, g.qq, g.qa, g.inv_a, g.ks,
            self._wall_v, self._wall_x, np.float32(self.timestep),
        )  # fmt: skip
        if g.mur_edges:
            mur_edges(p, p_next, float(np.sqrt(self._coeff)))

    def clear_obstacles(self) -> None:
        """Remove every obstacle. Field is left untouched."""
        self.obstacle_mask.fill(False)
        self._has_obstacles = False
        self._gcoef = None

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
        self._wall_v.fill(0.0)
        self._wall_x.fill(0.0)
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
        if self._general:
            # Phase 5 general path (physics.py): materials, rigid/impedance
            # walls, absorbing boundaries, c(x). Handles obstacles itself.
            p = self.p
            p_prev = self.p_prev
            p_next = self._p_next
            self._run_general(p, p_prev, p_next)
        elif kernel is not None:
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
        if self._has_obstacles and not self._general:
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
            for driver in self._drivers:
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
