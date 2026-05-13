# Documentation for `simulate.py`

## 1. Purpose

`simulate.py` defines the `Simulate` class, a stateful FDTD engine that
advances a discrete pressure field one timestep at a time. The class owns
the current and previous pressure grids, the wall clock, the driver list,
and exposes only two mutating methods: `step()` and `reset()`. This
step-at-a-time design is what makes the interactive web UI possible — the
backend can pause, resume, or reconfigure the simulation between steps,
something the original batch-mode loop did not allow.

## 2. Scientific Principles

### The acoustic wave equation and its leap-frog discretisation

The simulator solves the homogeneous acoustic wave equation,

$$ \frac{\partial^2 p}{\partial t^2} = c^2 \nabla^2 p, $$

with $p$ the acoustic pressure and $c$ the wavespeed. Using central
second-order finite differences in both time and space, the explicit
*leap-frog* update is

$$
p^{n+1}_{\mathbf{i}} = 2 p^{n}_{\mathbf{i}} - p^{n-1}_{\mathbf{i}}
+ (c \Delta t)^2 \, \big( \nabla^2 p^{n} \big)_{\mathbf{i}},
$$

where $\mathbf{i}$ indexes the spatial grid and the discrete Laplacian is
the standard 3-point stencil per axis (computed in `calculate.py` via
`scipy.ndimage.laplace` divided by $\Delta x^2$).

### CFL stability

The leap-frog scheme is stable iff the Courant number $\sigma$ satisfies

$$ \sigma \;\equiv\; \frac{c \, \Delta t}{\Delta x} \;\le\; \frac{1}{\sqrt{d}}, $$

with $d$ the spatial dimensionality. The constructor either chooses
$\Delta t = \kappa \cdot \Delta x / c$ for a target Courant number
$\kappa < 1/\sqrt{d}$ when no `timestep` is supplied, or — if the caller
supplies its own — emits a `RuntimeWarning` when the CFL bound is
violated. Operating just below the bound (e.g. $\kappa = 0.5$ in 2D) is
the usual sweet spot: stable, low numerical dispersion, and large enough
steps to make wall-clock progress.

### Source sampling and spurious DC

Because the source is added directly to $p^{n+1}$ at the driver cell,
the time-domain sampling of the waveform is what the simulation sees. A
continuous cosine $A\cos(2\pi f t)$ is acceptable only when
$f \, \Delta t \ll 0.5$ (Nyquist); ten or more samples per period
($f \, \Delta t \le 0.1$) is recommended. Pulses with non-zero mean
(e.g. `GaussianPulse`) deposit a DC component that, in a closed
hard-walled domain, leaves a quasi-static residual; the Ricker wavelet
in `waveforms.py` is mean-zero and is the recommended demo source.

## 3. Implementation Details

### `class Simulate`

Constructor arguments:

| name        | meaning |
| ----------- | ------- |
| `grid_shape` | tuple giving the spatial extent in cells, any dimension |
| `drivers`    | list of `Driver` objects, each with `position` and a callable `waveform` |
| `sensors`    | list of `Sensor` objects (passive, not used inside `step`) |
| `wavespeed`  | $c$ |
| `timestep`   | $\Delta t$; if `None`, derived from `courant` and the CFL bound |
| `gridstep`   | $\Delta x$ |
| `courant`    | target $\sigma$ used only when `timestep is None`; clipped to $0.95/\sqrt{d}$ |

State:

- `self.p`: current pressure $p^n$ as a `float32` array of shape `grid_shape`
- `self.p_prev`: $p^{n-1}$
- `self.time`: current simulated time
- `self.step_count`: number of `step()` calls executed

Methods:

- **`step()`** advances one timestep. The order is:
  1. compute the Laplacian of `self.p`,
  2. evaluate the leap-frog formula into a new buffer `p_next`,
  3. apply hard-wall boundary conditions (zero on every edge cell) so
     interior drivers are not erased afterwards,
  4. add each driver's `waveform(time)` value at its grid position,
  5. roll buffers (`p_prev <- p`, `p <- p_next`) and advance the clock.

- **`reset()`** zeros the pressure buffers, the clock, and the step
  counter. The geometry, drivers, and sensors are preserved. This is
  what the web UI's *Reset* button calls.

### Why this ordering matters

Applying the boundary condition *before* injecting the driver is a
deliberate fix for a bug discovered during the web UI work: when a
driver was placed close to an edge, zeroing edges *after* injection
silently cancelled its energy on every step. Applying boundaries first
preserves the driver's contribution at every interior cell — including
cells one step inside the boundary, where the edge zeroing nominally
would not touch them but the build-order made it ambiguous.

## 4. Interior obstacles

A boolean `obstacle_mask` attribute of shape `grid_shape` marks
interior cells that should behave as rigid Dirichlet walls. Conceptually
identical to the outer hard-wall boundary, just at arbitrary interior
positions specified by the user.

### Physical interpretation

Setting $p = 0$ at an obstacle cell every step is the simplest model of
an acoustically rigid scatterer: the pressure pinned at the wall cannot
do work on adjacent cells, so any incident wave is reflected with a
sign flip (closed-end reflection, $\Gamma = -1$). This is exactly the
behaviour the outer kernel already enforces on the four edge rows of
the 2D grid. Real materials have a finite specific acoustic impedance
$Z = \rho c$ that would produce partial reflection
($\Gamma = (Z_2 - Z_1)/(Z_2 + Z_1)$); modelling those would require
either a one-sided wave-equation update at the boundary cell or an
explicit impedance boundary condition, neither of which is currently
implemented.

### Ordering inside `step()`

```
kernel writes p_next over the interior, also zeroing the four outer edges
        ↓
if any obstacles exist: p_next[obstacle_mask] = 0      ← interior wall scrub
        ↓
for each driver: p_next[driver.position] += waveform(time)
```

Drivers placed on obstacle cells still emit. This is intentional and
matches the boundary semantics: a `Driver` at a corner cell overwrites
the wall zero in exactly the same way. If you want to silence drivers
inside walls, remove them from `simulation.drivers`.

### Hot-loop guard

`step()` only runs the masked write when `self._has_obstacles` is true.
That flag is updated by `set_obstacle` and `clear_obstacles`; a fresh
`Simulate` with no obstacles takes the same code path as before the
feature existed, so `check_simulate.py` keeps matching the existing
`reference.npz` with `max_abs ≈ 7.8e-7` and `l2_rel ≈ 1.0e-6` —
well inside the `atol=1e-5 / rtol=1e-4` gate.

### Mutation methods

| method | purpose |
| ------ | ------- |
| `set_obstacle(positions, value=True)` | mark/clear a batch of cells; out-of-bounds entries are silently dropped; marking also zeros the field at those cells so no stale pressure leaks through one final stencil read |
| `clear_obstacles()` | reset the mask without touching the field |
| `add_driver(driver)` | append a `Driver` and refresh the single-driver fast-path cache |
| `remove_driver(index)` | `del self.drivers[index]` and refresh the cache |
| `set_drivers(drivers)` | replace the entire list |

The single-driver fast path (`_fast_driver`, `_fast_driver_pos`) is now
refreshed on every driver mutation rather than only at construction, so
live add/remove during a UI session still hits the precomputed tuple-
index write.
