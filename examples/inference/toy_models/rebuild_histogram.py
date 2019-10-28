import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection


#bins shape 15 x nr of slices +1
#data shape 15 x nr of slices
def re_hist(bins,data,true_param,dmin,dmax,color='g'):
    linew=5
    f, ax = plt.subplots(3, 5, figsize=(50, 20))

    for x in range(3):
        for y in range(5):
            j = x * 5 + y
            for k in range(len(bins[j])-1):
                l, r = bins[j][k], bins[j][k+1]
                t, b = data[j][k], 0
                print("l,r,t,b: ", l,r,t,b)
                # patches.append[plt.Rectangle((l, t), r - l, b - t, color=color, fill=True, alpha=0.3)]
                ax[x,y].plot([l,r,r,l,l],[t,t,b,b,t], c=color, lw=5)
            peakv = np.max(data[j])
            ax[x, y].plot([dmin[j], dmin[j]], [peakv, 0], c='b', lw=linew)
            ax[x, y].plot([dmax[j], dmax[j]], [peakv, 0], c='b', lw=linew)
            # ax[x, y].plot([data_pred[j], data_pred[j]], [peakv, 0], lw=linew, ls=':', c='silver')
            ax[x, y].plot([true_param[j], true_param[j]], [peakv, 0], lw=linew, ls='--', c='black')


            # ax[x,y].add_patch(patch)

    plt.savefig('re_hist')
