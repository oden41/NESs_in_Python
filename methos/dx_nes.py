import numpy as np

class DX_NES():
    def __init__(self, dim, func, lam):
        self.dim = dim
        self.func = func
        self.lam = lam
        self.g = 0
        self.P_sigma = np.array([0] * dim)
        self.eps = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21 * dim**2))

    def initialize(self):