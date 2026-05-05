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
