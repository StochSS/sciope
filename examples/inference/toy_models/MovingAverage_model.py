import numpy as np



def simulate(param,n=100):

    m = len(param)
    g = np.random.normal(0, 1, n)
    gy = np.random.normal(0,0.3,n)
    y = np.zeros(n)
    x = np.zeros(n)
    for t in range(0,n):
        # print("t: ", t)
        x[t] += g[t]
        # print("g: ", "{0:.2f}".format(g[t]))
        for p in range(0,np.minimum(t,m)):
            x[t] += g[t-1-p]*param[p]
        y[t] = x[t]+gy[t]

    return y

def prior(n=10):

    p = []
    trials=0
    acc = 0
    while acc<n:
        trials+=1
        r=np.random.rand(2)*np.array([4,2])+np.array([-2,-1])
        #print("r: ", r)
        if r[1]+r[0] >= -1 and r[1] - r[0] >= -1:
            p.append(r)
            acc+=1
    # print("trials: ", trials, ", acc: ", acc)
    return p