import numpy as np


def man_statistics(TS):
    peaks=[]
    for ts in TS.T:
        print("ts shape: ", ts.shape)
        peaks.append(peak_finder(ts))
    peaks_ind = np.asarray(peaks).T
    peaks_val = TS[0][peaks_ind]
    print("peaks_val shape: ", peaks_val.shape)
    mean_peaks = np.mean(peaks_val,axis=1)
    max_peaks = np.max(peaks_val,axis=1)
    min_peaks = np.min(peaks_val,axis=1)
    print("min_peaks shape: ", min_peaks.shape)

    # print("peaks ind: ", peaks_ind)
    # print("peaks val: ", peaks_val)

    peak_dist = np.array(
        [[peaks_ind[i,0]-peaks_ind[i,1], peaks_ind[i,0]-peaks_ind[i,2], peaks_ind[i,1]-peaks_ind[i,2]] for i in range(8)])
    # print("peak dist shape: ", peak_dist.shape)
    peak_dist_mean = np.mean(peak_dist,axis=0)
    # print("peak dist mean: ", peak_dist_mean)
    s = {"mean_peaks":mean_peaks, "max_peaks":max_peaks, "min_peaks": min_peaks, "peak_dist_mean":peak_dist_mean}
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