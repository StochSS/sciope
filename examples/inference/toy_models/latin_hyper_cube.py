import numpy as np


def lhc_sampling(n, dim, dmin, dmax):
    points = []
    for i in range(dim):
        p = np.linspace(dmin[i], dmax[i], n)
        np.random.shuffle(p)
        points.append(p)
    points = np.array(points).T

    return points

