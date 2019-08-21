
import numpy as np


def normalize_data(data, dmin, dmax):
    dmin = np.array(dmin)
    dmax = np.array(dmax)
    return (data - dmin)/(dmax-dmin)

def denormalize_data(data, dmin, dmax):
    dmin = np.array(dmin)
    dmax = np.array(dmax)
    denorm = data * (dmax-dmin) + dmin
    return denorm