import matplotlib.pyplot as plt

import pickle
import numpy as np
import time
import dask

species=6

def find_peaks(ts):
    peaks = []
    for t in range(ts.shape[0]):
        left = np.maximum(0,t-10)
        right = np.minimum(ts.shape[0],t+10)
        ind =np.argmax(ts[left:right]) + left

        # print("t", t, ", ind: ", ind, ", left: ", left, ", right: ", right)

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



def summarys(ts):
    peaks = find_peaks(ts)
    dist = peak_dist(peaks)
    m_dist = mean_dist(dist)
    s_dist = std_dist(dist)

    return np.array([m_dist,s_dist])


validation_ts = pickle.load(open('validation_ts.p', "rb" ) )[:,:,species]

print("data loaded")
start = time.time()
val_sum = np.array([summarys(ts) for ts in validation_ts])
end = time.time()
print("summarys generated in ", end-start)
print(val_sum[200:210])

# f,ax = plt.subplots(4)
# j=0
# for i in range(16,20):
#     max_value = np.max(validation_ts[i])
#     peaks = find_peaks(validation_ts[i])
#     ax[j].plot(validation_ts[i])
#     for p in peaks:
#         ax[j].plot([p,p], [max_value,0])
#     j+=1
# plt.show()
