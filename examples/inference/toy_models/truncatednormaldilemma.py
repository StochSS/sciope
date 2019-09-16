import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import optimize
import numpy as np


dmin,dmax = 0,1
mean,std = 0.95,0.2
dims=1


y = stats.truncnorm.rvs((dmin-mean)/std, (dmax-mean)/std, loc=mean, scale=std, size=(100000,dims), random_state=None)


def mse(ypred):
    return np.mean((y-ypred)**2)

def mae(ypred):
    return np.mean(abs(y-ypred))

def norm_fit(params):
    ypred,sca = params
    # print("ypred: ", ypred, ", std: ", sca)
    a, b = (dmin - ypred)/sca, (dmax - ypred)/sca
    params = (a, b, ypred, sca)
    ret = stats.truncnorm.nnlf(params,y)
    # print("return: ", ret)
    return ret



print("test mse: ", mse(0.5))
mse = optimize.fmin(mse, 0.5)
print("ypred(mse): ", mse)

mae = optimize.fmin(mae, 0.5)
print("ypred(mae): ", mae)

nf,s = optimize.fmin(norm_fit, (0.5,1))
print("ypred, std(norm fit): ", nf, s)
print("optimized value norm fit: ", norm_fit((nf,s)))
print("real value norm fit: ", norm_fit((mean,std)))

l = np.linspace(-0.1,1.1,1000)
p = stats.truncnorm.pdf(l,(dmin-nf)/s,(dmax-nf)/s,nf,s)
tp = stats.truncnorm.pdf(l,(dmin-mean)/std,(dmax-mean)/std,mean,std)
fig, ax = plt.subplots(figsize=(20,10))
plt.tick_params(labelsize=22)
font = {'size'   : 22}
plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
plt.rc('axes', titlesize=22)     # fontsize of the axes title
plt.rc('font', **font)
ax.set(xlabel='$\hat{\\theta}$', ylabel='probability',
       title='Truncated normal posterior distribution, $\\mu$ = ' + str(mean) + ", $\\sigma$ = " + str(std))

ax.set_xlabel('$\hat{\\theta}$', size=22)
ax.set_ylabel('probability', size=22)

r = plt.hist(y, density=True, bins=20, alpha=0.6)# color='y')
peak_val = np.max(r[0])
ax.plot([mse, mse], [peak_val, 0],label='mean square error', c='r', ls='--')
ax.plot([mae, mae], [peak_val, 0],label='mean absolute error', c='r')
# if nf<dmax and nf>dmin:
#     plt.plot([nf, nf], [peak_val, 0],label='trunc norm fit loc', c='b')
# plt.plot(l,p, c = 'b', label='trunc norm fit')
ax.plot([dmax,dmax], [peak_val, 0],label='prior bound', c='b', lw=5)
ax.plot([dmin,dmin], [peak_val, 0], lw=5, c='b')
# ax.text(dmax-0.12,peak_val+0.03,s='upper prior bound')
# ax.text(dmin-0.12,peak_val+0.03,s='lower prior bound')

plt.plot([mean, mean], [peak_val, 0], label="$\\mu$", c='black', ls = ':')
plt.plot(l,tp, c='black')
ax.arrow(x = mse[0], y = peak_val/2, dx = abs(mse[0]-mean), dy=0, color='red', head_length = 0.025, head_width = 0.025, length_includes_head = True)
ax.arrow(mean, peak_val/2, -abs(mse[0]-mean), 0, color='red', head_length = 0.025, head_width = 0.025, length_includes_head = True)
ax.text(x = mse[0]+0.05, y = peak_val/2+.02,s='bias',color='black')

plt.legend()
plt.savefig('truncatednormdilemma')
plt.show()