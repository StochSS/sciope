import numpy as np


def man_statistics(ts):
    s = {"peaks":34}
    return s


def peak_finder(ts):
    len = ts.shape[0]
    print("len: ", len)
    for i in range(10,len-10):
        peaks=[]
        print(i, ", ts[i]: ", ts[i], "max: ", np.max(ts[i-10:i+11]))
        if ts[i]>=np.max(ts[i-10:i+11]):
            print("peak found at: ", i)
            peaks = peaks.append(i)
    print("peaks: ", peaks)
    return peaks