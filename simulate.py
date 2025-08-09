import numpy as np
from calculate import Calculate
laplacian_operator = Calculate().laplacian_operator


class Simulate:
    def __init__(self, gridsize=(16, 16), gridstep=1, duration=3, timestep=0.1, wavespeed=330):
        self.gridsize = gridsize
        self.gridstep = gridstep
        self.duration = duration
        self.timestep = timestep
        self.wavespeed = wavespeed

        self.dims = len(gridsize)
        self.history = np.zeros(shape=tuple((duration/timestep,) + gridsize))

    def set_initial_conditions(self, pressure_curr):
        pressure_curr[0, 0] += 1
        return pressure_curr

    def set_boundary_conditions(self):
        pass

    def _initialize(self, gridsize):
        pressure_prev = np.zeros(shape=gridsize)
        pressure_curr = np.zeros_like(pressure_prev)
        pressure_next = np.zeros_like(pressure_prev)
        time = 0
        return pressure_prev, pressure_curr, pressure_next, time

    def _update(self, pressure_prev, pressure_curr, laplacian):
        pressure_next = 2 * pressure_curr - pressure_prev + (self.wavespeed * self.timestep)**2 * laplacian
        return pressure_next

    def _apply_boundary_conditions(self, pressure, boundary, boundary_condition):
        pressure[boundary] = boundary_condition
        return pressure

    def _simulation_loop(self, pressure_prev, pressure_curr, time):
        laplacian = laplacian_operator(grid=pressure_curr, gridstep=self.gridstep)
        pressure_next = self._update(pressure_prev=pressure_prev, pressure_curr=pressure_curr,
                                     laplacian=laplacian)
        pressure_next = self._apply_boundary_conditions(pressure=pressure_next,
                                                        boundary=self.boundary, boundary_condition=self.boundary_condition)
        time += self.timestep
        return pressure_curr, pressure_next, time

    def run(self):
        pressure_prev, pressure_curr, pressure_next, time = self._initialize(gridsize=self.gridsize)

        pressure_curr = self._apply_boundary_conditions(pressure=pressure_curr,
                                                        boundary=boundary, boundary_condition=boundary_condition)

        iteration = 0
        while time < self.duration:
            pressure_curr, pressure_next, time = self._simulation_loop(pressure_prev=pressure_prev, pressure_curr=pressure_curr, time=time)
            self.history[iteration] = pressure_curr
            iteration += 1
            print('iteration: ' + str(iteration))
        return self.history