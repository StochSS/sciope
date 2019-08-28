import numpy as np


def man_statistics(TS):
    peaks=[]
    for ts in TS.T:
        print("ts shape: ", ts.shape)
        peaks.append(peak_finder(ts))
    peaks_ind = np.asarray(peaks).T
    peaks_val = TS[0][peaks_ind]
    mean_peaks = np.mean(peaks_val,axis=1)
    print("peaks ind: ", peaks_ind)
    print("peaks val: ", peaks_val)

    s = {"mean_peaks":34}
    return s


def peak_finder(ts):
    len = ts.shape[0]
    peaks = []
    filter_len = 20
    for i in range(2,len-2):
        treshold_peak = np.max(ts)*0.5
        filter_start = np.maximum(0,i-filter_len)
        filter_end = np.minimum(len,i+1+filter_len)

        if ts[i]>np.max(ts[filter_start:i]) and ts[i]>np.max(ts[i+1:filter_end]) and ts[i]>treshold_peak:
            peaks.append(i)

    return peaks