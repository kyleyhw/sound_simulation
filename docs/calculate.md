# Documentation for `calculate.py`

## 1. Purpose

The `calculate.py` script provides the core mathematical operation required for the FDTD simulation: the calculation of the Laplacian. It is encapsulated in its own class, `Calculate`, to maintain a clear separation between the mathematical operations and the simulation logic.

## 2. Scientific Principles

The FDTD method for solving the wave equation relies on approximating spatial and temporal derivatives. The spatial component of the wave equation is the Laplacian operator, `∇²`.

For a given pressure field `p` on a discrete grid, the Laplacian at a point is the sum of the second-order partial derivatives with respect to each spatial dimension. This script uses the `scipy.ndimage.laplace` function, which implements a standard central difference approximation for the Laplacian.

### Central Difference Approximation

The `scipy.ndimage.laplace` function computes the Laplacian by applying a kernel over the input grid. For a 2D grid, the operation at a point `(i, j)` is equivalent to:

```
L(i,j) = p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1] - 4*p[i, j]
```

This is then scaled by the square of the grid step, `1 / (gridstep**2)`, to produce the final approximation of the Laplacian, as shown in the equation from the main `README.md`:

$$ \nabla^2 p \approx \frac{p(x+\Delta x, y) - 2p(x, y) + p(x-\Delta x, y)}{\Delta x^2} + \frac{p(x, y+\Delta y) - 2p(x, y) + p(x, y-\Delta y)}{\Delta y^2} $$

## 3. Implementation Details

### Class: `Calculate`

-   **`__init__(self, dims=3)`**: The constructor is simple, primarily storing the number of dimensions. While the `laplacian_operator` method itself is dimension-agnostic (thanks to SciPy), this parameter was likely included for future extensions or for consistency with other parts of the simulation setup.

-   **`laplacian_operator(self, grid, gridstep)`**: This is the core method of the class.
    -   **Rationale**: By using `scipy.ndimage.laplace`, we leverage a highly optimized and well-tested implementation of the Laplacian operator. This avoids the need to manually implement the operation with nested loops, which would be significantly slower in Python.
    -   **Parameters**:
        -   `grid`: The N-dimensional NumPy or CuPy array representing the pressure field at the current time step.
        -   `gridstep`: The spatial discretization step (`Δx`).
    -   **Functionality**: It calls `sp.ndimage.laplace(grid)` and then divides the result by `gridstep**2`. This correctly scales the finite difference result to approximate the continuous differential operator.
