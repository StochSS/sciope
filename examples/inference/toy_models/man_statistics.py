import numpy as np


def man_statistics(ts):
    s = {"peaks":34}
    return s


def peak_finder(ts):
    len = ts.shape[0]
    for i in range(10,len-10):
        peaks=[]
        if ts[i]>=np.max(ts[i-10:i]) and ts[i]>=np.max(ts[i+1:i+11]):
            print("peak found at: ", i)
            peaks.append(i)
    return peaks