# Is the simulator right?

*Write-up · physics verification (plan Phase 5)*

Every result in this project rests on one finite-difference solver of the
wave equation. The engine is a leap-frog update on a Cartesian grid, run
in Python (numba) and in the browser (TypeScript, and WebGPU). This
write-up checks it against analytic answers rather than against its own
earlier output. The full tables are in [the physics report](../physics.md).

## Five checks against exact answers

**Room resonances.** A rectangular box only rings at
$f_{mn} = \tfrac{c}{2}\sqrt{(m/L_x)^2 + (n/L_y)^2}$. Projecting the
simulated field onto each exact mode shape isolates one sinusoid per
mode. Every mode lands on the scheme's own discrete dispersion relation
to about $10^{-5}\,\%$, and within 0.02–0.13 % of the continuum
frequency. The target was 0.5 %.

**Convergence.** An exact standing wave, run on grids from 16² to 128²,
shows errors that fall by a factor of 4 for each halving of the cell
size. The observed order is 1.98, 1.98 and 2.03, as a second-order
scheme should give.

**Energy.** In a lossless box the discrete leap-frog energy is
conserved exactly. Over 10,000 steps it drifts by $1.5\times10^{-7}$
(pressure-release walls) and $1.0\times10^{-7}$ (rigid walls): float32
round-off.

**Dispersion.** Short waves travel slightly too slowly on a grid. The
measured phase speed matches the von Neumann prediction to five digits,
along the axes and along the diagonal. The practical rule: 10 cells per
wavelength costs a 1.5 % speed error.

**Point sources.** The pressure from a point source matches the analytic
2D and 3D Green's functions, with correlation 0.995 and 0.991.

## Walls that behave like walls

**Absorbing edges.** Rooms need walls that absorb, and simulations of
open space need edges that don't reflect. The first-order Mur edge and a
graded sponge both reflect badly at oblique incidence: −8 dB and −5 dB at
62°. A convolutional perfectly matched layer (CPML) fixes this. It
reflects −94 dB at normal incidence and −49 dB at 62°, meeting the
−40 dB target. It stays stable over long runs, including with walls
painted into the layer.

**Reverberation.** A 2D room with partially absorbing walls decays with a
reverberation time between the Sabine and Eyring predictions, for every
absorption tested. It sits near Sabine for live rooms and moves towards
Eyring as the walls absorb more, which is what theory predicts.

## Three engines, one answer

The browser engine is a port of the Python one. The WebGPU engine runs
the same update in compute shaders. The parity fixtures hold them together:
- Python ↔ TypeScript agree to $10^{-4}$ on every physics path (materials,
  impedance walls, sponge, CPML, Mur, c(x), 2D and 3D);
- TypeScript ↔ WebGPU agree to about $10^{-6}$.

## What this does not show

- Walls are staircased: curved surfaces are approximated cell by cell.
- Impedance walls are locally reacting, with one frequency corner.
- The GPU path has been checked on a software adapter, not timed on real
  graphics hardware.
- Nothing here compares the simulator with a *real* room. That is the job
  of the [Lab](#/lab) and [its write-up](real_rooms.md).
