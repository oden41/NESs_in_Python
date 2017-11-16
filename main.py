import numpy as np
import random as rand
import experiments.experiment as ex

if __name__ == '__main__':
    dimension = 40
    exp = ex.Sphere(dimension=dimension)

    vec = np.random.rand(dimension)
    print(vec)
    print(exp.f(vec))