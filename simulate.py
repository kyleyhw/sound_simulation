import numpy as np
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
        total_iterations = int(np.ceil(duration / timestep)) + 1
        self.history = np.zeros(shape=tuple((total_iterations,) + gridsize))
        # self.boundary_conditions = {}

    def check_stability(self):
        sigma = self.wavespeed * self.timestep / self.gridstep
        return sigma <= 1/np.sqrt(self.dims)


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
        # pressure[1, 1] += 1 # needs changing
        return pressure

    def _apply_boundary_conditions(self, pressure):
        # for boundary in self.boundary_conditions.keys():
        #     boundary_coefficient = self.boundary_conditions[boundary]
        #     pressure = pressure[boundary] * boundary_coefficient
        pressure = set_edge_values(arr=pressure, value=0)
        return pressure

    def _apply_driver(self, pressure, time):
        pressure[1, 1] += np.cos(2 * np.pi * 3 * time)
        return pressure

    def _simulation_loop(self, pressure_prev, pressure_curr, time):
        laplacian = laplacian_operator(grid=pressure_curr, gridstep=self.gridstep)
        pressure_next = self._update(pressure_prev=pressure_prev, pressure_curr=pressure_curr,
                                     laplacian=laplacian)
        pressure_next = self._apply_boundary_conditions(pressure=pressure_next)
        time += self.timestep
        return pressure_curr, pressure_next, time

    def run(self):
        pressure_prev, pressure_curr, pressure_next, time = self._initialize(gridsize=self.gridsize)

        pressure_curr = self._apply_initial_conditions(pressure=pressure_curr)

        iteration = 0
        while time < self.duration:
            print('iteration: ' + str(iteration))
            pressure_curr = self._apply_driver(pressure=pressure_curr, time=time)
            # print(pressure_curr)
            pressure_prev, pressure_curr, time = self._simulation_loop(pressure_prev=pressure_prev,
                                                                       pressure_curr=pressure_curr, time=time)
            self.history[iteration] = pressure_curr
            iteration += 1
        return self.history

