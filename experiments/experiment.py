import numpy as np

class Experiment():
    def __init__(self, dimension):
        self.dim = dimension

    def f(self, x):
        return 0

class Sphere(Experiment):
    def __init__(self, dimension):
        super().__init__(dimension)

    def f(self, x):
        assert len(x) == self.dim
        return np.sum(x**2)

class KTablet(Experiment):
    def __init__(self, dimension):
        super().__init__(dimension)

    def f(self, x):
        assert len(x) == self.dim
        k = self.dim // 4
        x[k:] = x[k:] * 100
        return np.sum(x**2)

class Rosenbrock(Experiment):
    def __init__(self, dimension):
        super().__init__(dimension)

    def f(self, x):
        assert len(x) == self.dim
        return sum(100.0*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
