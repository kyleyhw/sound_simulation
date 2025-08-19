import numpy as np
from tqdm import tqdm
import warnings

from setup import Driver
from calculate import Calculate
from utils import set_edge_values
laplacian_operator = Calculate().laplacian_operator


class Simulate:
    def __init__(self, gridsize=(16, 16), gridstep=1, duration=3, timestep=0.1, wavespeed=330):
        self.gridsize = gridsize
        self.gridstep = gridstep
        self.duration = duration
        self.timestep = timestep
        self.wavespeed = wavespeed

        self.dims = len(gridsize)
        self.total_iterations = int(np.ceil(duration / timestep)) + 1
        self.history = np.zeros(shape=tuple((self.total_iterations,) + gridsize))
        # self.boundary_conditions = {}
        self.drivers = []

    def get_params(self):
        params = {'gridsize' : self.gridsize,
                  'gridstep' : self.gridstep,
                  'duration' : self.duration,
                  'timestep' : self.timestep,
                  'wavespeed': self.wavespeed}
        return params

    def _is_stable(self):
        sigma = self.wavespeed * self.timestep / self.gridstep
        return sigma <= 1/np.sqrt(self.dims)

    def check_stability(self):
        if not self._is_stable():
            warnings.warn('This is numerically unstable')


    # def set_initial_conditions(self, pressure_curr):
    #     pass

    # def add_boundary_conditions(self, Boundary):
    #     self.boundary_conditions[Boundary.boundary] = Boundary.boundary_coefficient
    #     return

    def _initialize(self, gridsize):
        pressure_prev = np.zeros(shape=gridsize)
        pressure_curr = np.zeros_like(pressure_prev)
        pressure_next = np.zeros_like(pressure_prev)
        time = 0
        return pressure_prev, pressure_curr, pressure_next, time

    def _update(self, pressure_prev, pressure_curr, laplacian):
        pressure_next = 2 * pressure_curr - pressure_prev + (self.wavespeed * self.timestep)**2 * laplacian
        return pressure_next

    def _apply_initial_conditions(self, pressure):
        # pressure[(1,) * self.dims] += 1 # needs changing
        return pressure

    def _apply_boundary_conditions(self, pressure):
        # for boundary in self.boundary_conditions.keys():
        #     boundary_coefficient = self.boundary_conditions[boundary]
        #     pressure = pressure[boundary] * boundary_coefficient
        pressure = set_edge_values(arr=pressure, value=0)
        return pressure

    def add_driver(self, driver:Driver):
        """Adds a driver object to the simulation."""
        # Check if the driver location is valid for the grid dimensions
        if len(driver.location) != self.dims:
            raise ValueError(f"Driver location has {len(driver.location)} dims, but grid has {self.dims} dims.")
        if driver.location > self.gridsize:
            raise ValueError('driver is out of bounds')
        self.drivers.append(driver)
        return

    def _apply_drivers(self, pressure, time):
        for driver in self.drivers:
            driver_value = driver.get_value(time)
            pressure[driver.location] += driver_value
        return pressure

    def _simulation_loop(self, pressure_prev, pressure_curr, time):
        laplacian = laplacian_operator(grid=pressure_curr, gridstep=self.gridstep)
        pressure_next = self._update(pressure_prev=pressure_prev, pressure_curr=pressure_curr,
                                     laplacian=laplacian)
        pressure_next = self._apply_drivers(pressure=pressure_next, time=time)
        pressure_next = self._apply_boundary_conditions(pressure=pressure_next)
        time += self.timestep
        return pressure_curr, pressure_next, time

    def run(self):
        pressure_prev, pressure_curr, pressure_next, time = self._initialize(gridsize=self.gridsize)

        pressure_curr = self._apply_initial_conditions(pressure=pressure_curr)

        self.history[0] = pressure_curr

        for iteration in tqdm(range(1, self.total_iterations), desc="simulating"):
            # print('iteration: ' + str(iteration))
            # print(pressure_curr)
            pressure_prev, pressure_curr, time = self._simulation_loop(pressure_prev=pressure_prev,
                                                                       pressure_curr=pressure_curr, time=time)
            self.history[iteration] = pressure_curr
        return self.history