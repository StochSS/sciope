import numpy as np


def find_peaks(ts):
    peaks = []
    for t in range(ts.shape[0]):
        left = np.maximum(0,t-10)
        right = np.minimum(ts.shape[0],t+10)
        ind =np.argmax(ts[left:right]) + left
        if ts[ind]>np.max(ts)*0.5 and ind==t:
            peaks.append(ind)
    return peaks


def peak_dist(peaks):
    dist = []
    for i in range(len(peaks)-1):
        dist.append(peaks[i+1]-peaks[i])
    return dist


def mean_dist(dist):
    if dist:
        return np.mean(dist)
    else:
        return -1


def std_dist(dist):
    if dist:
        return np.std(dist)
    else:
        return -1


def mean_peak(ts,peaks):
    if peaks:
        return np.mean(ts[peaks])
    else:
        return -1

def std_peak(ts,peaks):
    if peaks:
        return np.std(ts[peaks])
    else:
        return -1


def summarys(ts,ind=None):
    if ind:
        if ind%1000==0:
            print("ind: ", ind)
    peaks = find_peaks(ts)
    dist = peak_dist(peaks)
    m_dist = mean_dist(dist)
    s_dist = std_dist(dist)
    m_peaks = mean_peak(ts, peaks)
    s_peaks = std_peak(ts, peaks)

    return np.array([m_dist,s_dist, m_peaks, s_peaks])


