import numpy as np



def simulate(param,n=100):

    m = len(param)
    g = np.random.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(0,n):
        # print("t: ", t)
        y[t] += g[t]
        # print("g: ", "{0:.2f}".format(g[t]))
        for p in range(0,np.minimum(t,m)):
            y[t] += y[t-1-p]*param[p]

    return y

def prior(n=10):

    p = []
    trials=0
    acc = 0
    while acc<n:
        trials+=1
        r=np.random.rand(2)*np.array([4,2])+np.array([-2,-1])

        # print("r: ", r)
        if r[1]<1+r[0] and r[1] < 1 - r[0]:
            p.append(r)
            acc+=1
    # print("trials: ", trials, ", acc: ", acc)
    return p


#
# param = prior(3)
# import matplotlib.pyplot as plt
#
# # plt.scatter(param[0],param[1])
#
#
#
# y = np.array([simulate(p,n=100) for p in param]).T
#
# plt.plot(y)
# plt.ylim(-10,10)
#
#
#
# plt.show()



