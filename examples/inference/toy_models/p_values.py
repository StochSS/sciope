from scipy.stats import truncnorm
import numpy as np
from matplotlib import pylab as plt



def trunc_norm_pvalue(x,loc, scale, dmin, dmax):

    #x needs to be 2dim nparray, loc,scale,loc, scale, dmin, dmax 1 dim
    if x.ndim== 1:
        x = np.expand_dims(x,0)

    # print("shape: x, loc, scale,dmin,dmax: ", x.shape, loc.shape, scale.shape, dmin.shape, dmax.shape)

    y= np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # print("(", i, ",", j, "), term 1: ", truncnorm.cdf(x[i,j], (dmin[j]-loc[j])/scale[j], (dmax[j]-loc[j])/scale[j], loc=loc[j], scale=scale[j]),
            #       "term 2: ", truncnorm.cdf(2*loc[j]-x[i,j], (dmin[j]-loc[j])/scale[j], (dmax[j]-loc[j])/scale[j], loc=loc[j], scale=scale[j]))
            # print("x: ", x[i,j], ", loc: ", loc[j], ", scale: ", scale[j], ", dmin: ", dmin[j], ", dmax: ", dmax[j])

            y[i,j] = 1 - abs(truncnorm.cdf(x[i,j], (dmin[j]-loc[j])/scale[j], (dmax[j]-loc[j])/scale[j], loc=loc[j], scale=scale[j]) - \
            truncnorm.cdf(2*loc[j]-x[i,j], (dmin[j]-loc[j])/scale[j], (dmax[j]-loc[j])/scale[j], loc=loc[j], scale=scale[j]))
    return y







# test
# dims=3
# center = np.random.rand(dims)
# dmin = np.random.rand(dims)*center
# dmax = center + np.random.rand(dims)*(1-center)
# dmin = np.ones(dims)*(-10000)
# dmax = np.ones(dims)*(10000)
#
# mean = np.random.rand(dims)*(dmax-dmin)+dmin
#
# std = np.random.rand(dims)
#
#
# d=truncnorm.rvs((dmin-mean)/std, (dmax-mean)/std, loc=mean, scale=std, size=(10,dims), random_state=None)
# print("dmin, dmax: ", dmin, dmax, ", mean, std: ", mean, std)

# plt.hist(d,bins=100)
# plt.show()
# print("d shape: ", d.shape)
# pval_d = trunc_norm_pvalue(d,mean,std,dmin,dmax)
# print("pval shape: ", pval_d.shape)
# print("pval: ", pval_d[0:15])
# print("mean pval: ", np.mean(pval_d))
# for i in range(dims):
#     print("pval(", i, "): ", trunc_norm_pvalue(mean[i],mean[i],std[i],dmin[i],dmax[i]))
# print("pval(0.2): ", trunc_norm_pvalue(np.ones(dims)*0.2,mean,std,dmin,dmax))
# print("1-pval(0.2+0.6): ", 1 - trunc_norm_pvalue(mean+0.6,mean,std,dmin,dmax))
# print("1-pval(0.2+2*0.6): ", 1 - trunc_norm_pvalue(np.expand_dims(mean,0)+2*std,mean,std,dmin,dmax))
# print("1-pval(0.2+3*0.6): ", 1 -trunc_norm_pvalue(np.expand_dims(mean,0)+3*std,mean,std,dmin,dmax))