import numpy as np

class Boundary:
    def __init__(self, gridsize):
        self.gridsize = gridsize
        self.boundary = None
        self.boundary_coefficient = 0

    def set_boundary(self, boundary):
        self.boundary = boundary

    def set_boundary_condition(self, boundary_coefficient):
        self.boundary_coefficient = boundary_coefficient

    def apply_boundary_condition(self, grid):
        grid[self.boundary] = self.boundary_coefficient * grid[self.boundary]
        return grid
