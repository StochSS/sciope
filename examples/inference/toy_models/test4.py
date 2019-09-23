
from scipy.stats import truncnorm, norm

import numpy as np
from matplotlib import pylab as plt


theta = np.random.rand(10000)

likelihood_f = lambda ts,theta: norm.pdf(ts,theta,0.1)

ts_span = np.linspace(-0.5,1.5,5)
theta_span = np.linspace(0,1,101)

d_theta_span = theta_span[1]-theta_span[0]

f, ax = plt.subplots(ts_span.shape[0],3,figsize=(20,20))

x=0
for ts in ts_span:
  post=likelihood_f(ts,theta_span)

  post = post / np.sum(post*d_theta_span)


  ax[x,0].set_title("posterior p(theta|ts=" + str(ts) + ")")
  ax[x,0].plot(theta_span,post)
  ax[x,0].set_xlabel('theta')
  ax[x,0].set_ylabel('p')
  x+=1




plt.show()