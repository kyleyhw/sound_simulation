# Documentation for `simulate.py`

## 1. Purpose

`simulate.py` is the main driver of the simulation. It orchestrates the entire FDTD process, from initialization to the final time-stepping loop. It brings together the calculation kernel, boundary conditions, drivers, and sensors to evolve the pressure field over time and record the results.

## 2. Scientific Principles

The core of this script is the implementation of the explicit FDTD update equation, which was derived by discretizing the acoustic wave equation.

### The FDTD Update Equation

The pressure at the next time step, `p(t+Δt)`, is calculated based on the pressure at the current, `p(t)`, and previous, `p(t-Δt)`, time steps, and the Laplacian of the current pressure field, `∇²p(t)`.

The `_update` method implements this directly:

```python
pressure_next = 2 * pressure_curr - pressure_prev + (self.wavespeed * self.timestep)**2 * laplacian
```

This corresponds to the rearranged discretized wave equation:

$$ p(t+\Delta t) = 2p(t) - p(t-\Delta t) + (c \Delta t)^2 \nabla^2 p(t) $$ 

### Courant-Friedrichs-Lewy (CFL) Stability Condition

For an explicit finite difference scheme like this to be numerically stable, the time step `Δt` and spatial step `Δx` must satisfy the CFL condition. The `_is_stable` method checks this condition:

$$ \sigma = \frac{c \Delta t}{\Delta x} \le \frac{1}{\sqrt{N_{dims}}} $$ 

Where `N_dims` is the number of spatial dimensions. If this condition is violated, the simulation will produce physically meaningless, exponentially growing oscillations.

## 3. Implementation Details

### Class: `Simulate`

-   **`__init__(...)`**: Initializes the simulation parameters such as grid size, time steps, and duration. It also creates the `history` array, which will store the entire pressure field at every time step. This is memory-intensive but allows for full playback and analysis after the simulation is complete.

-   **`_initialize(...)`**: Creates and zeros out the three pressure field arrays required for the FDTD update: `pressure_prev`, `pressure_curr`, and `pressure_next`.

-   **`_update(...)`**: Implements the core FDTD update equation as described above.

-   **`_apply_boundary_conditions(...)`**: This method is responsible for enforcing what happens at the edges of the simulation grid. Currently, it calls `set_edge_values(arr=pressure, value=0)`, which implements a simple "hard wall" boundary where the pressure is fixed to zero. This is a Dirichlet boundary condition.

-   **`add_driver(...)` and `_apply_drivers(...)`**: These methods manage the sound sources. The `_apply_drivers` method adds the driver's current amplitude to the pressure field at its specified location at each time step, effectively injecting sound into the simulation.

-   **`_simulation_loop(...)`**: This method represents a single step in time. It performs the following sequence:
    1.  Calculates the Laplacian of the current pressure field using the function from `calculate.py`.
    2.  Calculates the next pressure field using the `_update` method.
    3.  Applies any drivers (sound sources).
    4.  Applies the boundary conditions.
    5.  Increments the time.

-   **`run(...)`**: This is the main public method that executes the entire simulation. It initializes the fields and then calls `_simulation_loop` repeatedly for the total number of iterations, storing the result of each step in the `history` array.

-   **`assign_sensors(...)`**: After the simulation is complete, this method extracts the time series data from the `history` array at the locations specified by the `Sensor` objects. This provides the simulated "recordings" from the virtual microphones.
