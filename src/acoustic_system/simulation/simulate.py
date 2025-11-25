import numpy as np
import warnings

from .setup import Driver, Sensor
from .calculate import Calculate
from .utils import set_edge_values

laplacian_operator = Calculate().laplacian_operator

class Simulate:
        def __init__(self, grid_shape=(100, 100), drivers=None, sensors=None, wavespeed=1.0, timestep=0.1):
            """
            A stateful class for running FDTD simulations step-by-step.
            """
            self.grid_shape = grid_shape
            self.wavespeed = wavespeed
            self.timestep = timestep
    
            self.dims = len(grid_shape)
    
            self.drivers = drivers if drivers is not None else []
            self.sensors = sensors if sensors is not None else []
    
            # Internal state
            self.time = 0.0
            self.p_prev = np.zeros(shape=grid_shape, dtype=np.float32)
            self.p = np.zeros(shape=grid_shape, dtype=np.float32) # Current pressure grid
    
        def step(self):
            """Advances the simulation by a single time step."""
            # Calculate Laplacian
            laplacian = laplacian_operator(grid=self.p)
    
            # Update pressure grid
            p_next = 2 * self.p - self.p_prev + (self.wavespeed * self.timestep)**2 * laplacian
    
            # Apply simple boundary conditions (hard walls)
            set_edge_values(arr=p_next, value=0)

            # Apply drivers
            for driver in self.drivers:
                driver_value = driver.get_value(self.time)
                # Ensure driver position is within grid bounds
                if all(0 <= pos < size for pos, size in zip(driver.position, self.grid_shape)):
                    p_next[driver.position] += driver_value
    
            # Update state for the next step
            self.p_prev = self.p.copy()
            self.p = p_next
    
            # Advance time
            self.time += self.timestep