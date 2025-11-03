# Documentation for `boundary.py`

## 1. Purpose

`boundary.py` is intended to define and apply boundary conditions to the simulation grid. In wave simulations, boundary conditions are critical as they define how waves behave when they reach the edge of the simulation domain (e.g., reflect, get absorbed, or wrap around).

**Note:** The current implementation in `simulate.py` uses a more direct approach for simple hard-wall boundaries (`set_edge_values(arr=pressure, value=0)` from `utils.py`). The `Boundary` class in this file represents a more general, albeit currently unused, structure for handling more complex or specifiable boundary conditions.

## 2. Scientific Principles

Boundary conditions in differential equations specify the behavior of the solution at the boundary of its domain. Common types in wave simulations include:

-   **Dirichlet Boundary Conditions**: The value of the field is specified at the boundary. A value of `p=0` represents a "soft" boundary where pressure is released, causing a phase-inverted reflection. This is what is currently implemented in `simulate.py`.
-   **Neumann Boundary Conditions**: The derivative of the field (the pressure gradient) is specified at the boundary. A value of `∂p/∂n = 0` represents a "hard" boundary where the particle velocity is zero, causing a reflection with no phase inversion.
-   **Absorbing Boundary Conditions (e.g., PML)**: These are more complex conditions designed to absorb incident waves without reflecting them, simulating an open, infinite domain.

## 3. Implementation Details

### Class: `Boundary`

This class is designed as a generic container for a boundary condition.

-   **`__init__(self, gridsize)`**: Initializes with the grid size.
-   **`set_boundary(self, boundary)`**: This method is intended to store the *indices* of the grid points that constitute the boundary.
-   **`set_boundary_condition(self, boundary_coefficient)`**: This stores a coefficient that determines the effect of the boundary. For example, a coefficient of `0` would force the boundary to zero pressure, while a coefficient of `1` would mean the boundary values are unchanged (which is not a physically meaningful boundary condition on its own).
-   **`apply_boundary_condition(self, grid)`**: This method applies the condition by multiplying the grid values at the boundary indices by the stored coefficient.

### Design Rationale and Future Use

The design of this class suggests a more flexible system was envisioned. For example, one could create different `Boundary` objects for different walls (e.g., top, bottom, left, right) and assign each a different reflection coefficient. This would allow for more complex environments where some walls are more reflective than others.

While not currently integrated into the main simulation loop in `simulate.py`, this class provides a clear foundation for implementing more sophisticated boundary behaviors in the future.
