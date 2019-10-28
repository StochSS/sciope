import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#bins shape 15 x nr of slices +1
#data shape 15 x nr of slices
def re_hist(bins,data,color='g'):
    f, ax = plt.subplots(3, 5, figsize=(50, 20))

    for x in range(3):
        for y in range(5):
            j = x * 5 + y
            for k in range(len(bins[j])-1):
                l, r = bins[j][k], bins[j][k+1]
                t, b = data[j][k], 0
                print("l,r,t,b: ", l,r,t,b)
                patch = plt.Rectangle((l, t), r - l, b - t, color=color, fill=True, alpha=0.3)
                ax[x,y].add_patch(patch)

    plt.savefig('re_hist')
