import scipy as sp

class Calculate:
    def __init__(self, dims=3):
        self.dims = dims

    def laplacian_operator(self, grid, gridstep):
        return sp.ndimage.laplace(grid) / (gridstep**2)