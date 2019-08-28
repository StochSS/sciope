import numpy as np


def man_statistics(ts):
    s = {"peaks":34}
    return s


def peak_finder(ts):
    len = ts.shape[0]
    peaks = []
    filter_len = 20
    for i in range(10,len-filter_len):
        treshold_peak = np.max(ts)*0.5
        filter_start = np.maximum(0,i-filter_len)
        filter_end = np.minimum(len,i+1+filter_len)

        if ts[i]>np.max(ts[filter_start:i]) and ts[i]>np.max(ts[i+1:filter_end]) and ts[i]>treshold_peak:
            peaks.append(i)

    return peaks