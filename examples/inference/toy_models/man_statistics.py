import numpy as np


def man_statistics(ts):
    s = {"peaks":34}
    return s


def peak_finder(ts):
    len = ts.shape[0]
    peaks = []

    for i in range(10,len-10):
        treshold_peak = np.max(ts)*0.5
        filter_start = np.maximum(0,i-20)
        filter_end = np.maximum(len,i+21)

        if ts[i]>np.max(ts[filter_start:i]) and ts[i]>np.max(ts[i+1:filter_end]) and ts[i]>treshold_peak:
            peaks.append(i)

    return peaks