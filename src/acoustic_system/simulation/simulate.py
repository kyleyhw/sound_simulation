import warnings
from typing import List, Optional, Tuple

import numpy as np

from .calculate import Calculate
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

    def reset(self) -> None:
        """Zero pressure fields and the clock; preserve geometry and drivers."""
        self.p_prev.fill(0.0)
        self.p.fill(0.0)
        self.time = 0.0
        self.step_count = 0

    def step(self) -> None:
        """Advance the simulation by a single timestep."""
        laplacian = laplacian_operator(grid=self.p, gridstep=self.gridstep)
        p_next = 2.0 * self.p - self.p_prev + (self.wavespeed * self.timestep) ** 2 * laplacian

        # Hard-wall (Dirichlet, p=0) boundaries first so interior drivers are not erased.
        set_edge_values(arr=p_next, value=0)

        for driver in self.drivers:
            value = driver.get_value(self.time)
            if all(0 <= pos < size for pos, size in zip(driver.position, self.grid_shape)):
                p_next[tuple(driver.position)] += value

        self.p_prev = self.p
        self.p = p_next
        self.time += self.timestep
        self.step_count += 1
