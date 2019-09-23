import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from scipy.stats import truncnorm, norm

import numpy as np
from matplotlib import pylab as plt
tfd = tfp.distributions
# tf.logging.set_verbosity(tf.logging.ERROR)

negloglik = lambda y, p_y: -p_y.log_prob(y)

batch_size = 32
learning_rate =0.001
epochs = 5
batch_mom = 0.99

# std_f = lambda x: abs(1 - x*(x-10))

std_f = lambda x: (10 - x) *0.1 +1


# Build model.

lower = 0
upper = 10

def stoch_p(theta,noise=True):
  ts = theta
  if noise:
    s = std_f(theta)
    a = (lower - ts)/s
    b = (upper - ts)/s
    ts = norm.rvs(ts,s)
  return ts


# trunc_ = tfd.TruncatedNormal(loc=0, scale=4, low=lower, high=upper)
#
# samp=trunc_.sample(100)
# plt.hist(samp)
# plt.show()


training_theta = np.random.rand(100000)*10
training_ts = stoch_p(training_theta)
print("training data produced")
#NN-model

# input = keras.Input(shape=(1,))
# layer = tf.keras.layers.Dense(1000,input_shape = (1,))(input)
# layer = tf.keras.layers.Activation("relu")(layer)
# layer =

model = tf.keras.Sequential([
  tf.keras.layers.Dense(1000,input_shape = (1,)),
  # tf.keras.layers.BatchNormalization(momentum=batch_mom),
  tf.keras.layers.Activation("relu"),

  # tf.keras.layers.Dense(100),
  # # tf.keras.layers.BatchNormalization(momentum=batch_mom),
  # tf.keras.layers.Activation("relu"),
  tf.keras.layers.Dense(2),
  tfp.layers.DistributionLambda(
    lambda t: tfd.TruncatedNormal(loc=tf.math.softplus(0.05 * t[..., :1])-2*(0.001 + tf.math.softplus(0.05 * t[...,1:])), scale=0.001 + tf.math.softplus(0.05 * t[...,1:]), low=lower,
                                  high=upper))

  # tfp.layers.DistributionLambda(lambda t: tfd.TruncatedNormal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.05 * t[...,1:2]), low=lower, high=upper)),
])

model.summary()
# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)
model.fit(training_ts, training_theta, epochs=epochs, batch_size=batch_size)#, verbose=False)

# model2 = tf.keras.Sequential([
#   tf.keras.layers.Dense(1000,input_shape = (1,)),
#   # tf.keras.layers.BatchNormalization(momentum=batch_mom),
#   tf.keras.layers.Activation("relu"),
#   # tf.keras.layers.Dense(100),
#   # # tf.keras.layers.BatchNormalization(momentum=batch_mom),
#   # tf.keras.layers.Activation("relu"),
#   tf.keras.layers.Dense(1),
#
# ])

# model2.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss='mse')
# model2.fit(training_ts, training_theta, epochs=epochs, batch_size=batch_size)#, verbose=False)



# Make predictions.
test_theta = np.expand_dims(np.random.rand(1000)*10,1)
test_ts = stoch_p(test_theta)



si = np.argsort(test_ts,axis=0)
print("si shape: ", si.shape)
# print("yhat shape: ", y.shape)
test_theta_pred = model(test_ts)
test_theta_pred_mean = test_theta_pred._loc.numpy()[:,0]
test_theta_pred_std = test_theta_pred._scale.numpy()[:,0]
print("test_theta_pred_mean shape: ", test_theta_pred_mean.shape)
print("test_theta_pred_std shape: ", test_theta_pred_std.shape)

ymin = truncnorm.ppf(0.05, a=(lower-test_theta_pred_mean)/test_theta_pred_std,
                     b=(upper-test_theta_pred_mean)/test_theta_pred_std, loc=test_theta_pred_mean, scale=test_theta_pred_std)
ymax = truncnorm.ppf(0.95, a=(lower-test_theta_pred_mean)/test_theta_pred_std,
                     b=(upper-test_theta_pred_mean)/test_theta_pred_std, loc=test_theta_pred_mean, scale=test_theta_pred_std)

# truncnorm((lower-test_theta_pred_mean)/test_theta_pred_std, (upper-test_theta_pred_mean)/test_theta_pred_std)

likelihood_f = lambda ts,theta: norm.pdf(ts,theta,std_f(theta))

theta_span = np.linspace(0,10,101)
d_theta_span = theta_span[1]-theta_span[0]

test_theta_pred_mse = model2.predict(test_ts)


def post_max(ts):
  # print("inside post_max ts: ", ts)
  post=likelihood_f(ts,theta_span)
  post = post / np.sum(post*d_theta_span)
  i=np.argmax(post)
  ret = theta_span[i]
  # print("ts: ", ts, ", theta: ", ret)
  return ret


rstd = std_f(test_ts)

# rymin = ts_postmax - 2*rstd
# rymax = ts_postmax + 2*rstd
#
#
# print("accuracy model 1: ", np.sum(abs(test_theta_pred_mean-ts_postmax)))
# print("accuracy model 2: ", np.sum(abs(test_theta_pred_mse-ts_postmax)))

# for xt,yt,ys in zip(x_tst,yhat,y_std):
#     print("x: ", xt, ", y: mean: ", yt, ", std", ys)


# print("x{si] shape: ", x[si].shape)
# print("ymean{si] shape: ", ymean[si].shape)
print("test_ts shape: ", test_ts.shape)



pred_theta_postmax = np.array([post_max(test_ts_[0]) for test_ts_ in test_ts])
# print("pred_theta_postmax shape: ", pred_theta_postmax.shape)
# print("pred_theta_postmax: ", pred_theta_postmax.shape)


ts_sample = 5

post=likelihood_f(ts_sample,theta_span)
post = post / np.sum(post*d_theta_span)
i=np.argmax(post)
ret = theta_span[i]
# print("ret: ", ret)

# plt.plot(theta_span,post)
# plt.show()
# print("test_theta_pred_mse: ", test_theta_pred_mse)
# print("si: ", si.shape, " - - ", si)
# print("test_ts[si][:,0],pred_theta_postmax[si] shape: ", test_ts[si][:,0].shape ,pred_theta_postmax[si].shape)
# print("ymin shape ", ymin.shape, "ymin[si] shape", ymin[si].shape)
# print("test_theta_pred_mean[si] shape: ", test_theta_pred_mean[si].shape, ", std: ", test_theta_pred_std[si].shape)
# print("test_theta_pred_mean[si][0,0] = ", test_theta_pred_mean[si][0,0],
#       ", test_theta_pred_std[si][0,0] = ", test_theta_pred_std[si][0,0], ", lower =  ", lower, ", upper = ", upper)
#
# print("test_theta_pred_mean[si][0,-1] = ", test_theta_pred_mean[si][0][-1],
#       ", test_theta_pred_std[si][0,-1] = ", [si][0][-1], ", lower =  ", lower, ", upper = ", upper)
f, ax = plt.subplots(2,1)

ax[0].scatter(test_ts[si][:,0],test_theta[si][:,0],c='black', alpha=0.3)
ax[0].plot(test_ts[si][:,0],test_ts[si][:,0],c='yellow', linestyle=':')

# ax[0].plot(test_ts[si][:,0],test_theta_pred_mean[si], label = "model1", c='b')
# ax[0].plot(test_ts[si][:,0],test_theta_pred_mse[si][:,0], label = "model2",c='g')
# ax[0].plot(test_ts[si][:,0],pred_theta_postmax[si], label = "true posterior maximum", c = 'black')
ax[0].set_xlabel('ts')
ax[0].set_ylabel('theta')
ax[0].set_ylim((-2,12))
ax[1].plot(test_ts[si][:,0],pred_theta_postmax[si], label = "true posterior maximum", c = 'black')
ax[1].set_xlabel('ts')
ax[1].set_ylabel('theta')

ax[0].plot(test_ts[si][:,0],ymin[si], c='b', linestyle='--', label='ymin')
ax[0].plot(test_ts[si][:,0],ymax[si], c='b', linestyle='--', label='ymax')
# ax[0].plot(x[:,0], rymax[:,0], c = 'black', linestyle='--')
# ax[0].plot(x[:,0], rymin[:,0], c = 'black', linestyle='--')
#
# ax[1].plot(x[:,0],ymean[:,0], label = "model1", c='b')
# ax[1].plot(x[:,0],ymse[:,0], label = "model2",c='g')
# ax[1].plot(x[:,0],y_true[:,0], label = "true", c = 'black')

# ax[1].plot(x[:,0],ymin[:,0], c='b', linestyle='--')
# ax[1].plot(x[:,0],ymax[:,0], c='b', linestyle='--')
# ax[1].plot(x[:,0], rymax[:,0], c = 'black', linestyle='--')
# ax[1].plot(x[:,0], rymin[:,0], c = 'black', linestyle='--')

ax[1].plot(test_ts[si][:,0], test_theta_pred_std[si],label='predicted std')
# ax[2].plot(x[:,0], rstd[:,0],label='true std')



# ax[0].plot(x[si][:,0], y_std[si][:,0])
# ax[0].plot(x[si][:,0],1-x[si][:,0]*0.1)
ax[0].legend()
# ax[1].legend()
# ax[2].legend()
plt.show()

test_theta = np.random.rand(1000000)*10
test_ts = stoch_p(test_theta)

f, ax = plt.subplots(3,3, figsize=(30,30))

for x in range(3):
  for y in range(3):

    data = x*3+y
    ind = np.argpartition(abs(test_ts-data),1000)[:1000]
    accepted_theta = test_theta[ind]
    accepted_ts = test_ts[ind]
    # print("accepted ts min:max = ", np.min(accepted_ts),":",np.max(accepted_ts))

    data_dist = model(np.array([[data]]))

    lo,sc = data_dist._loc.numpy()[0,0],data_dist._scale.numpy()[0,0]
    l = np.linspace(-1,11,311)
    p = truncnorm(a=-lo/sc, b=(10-lo)/sc, loc=lo, scale=sc).pdf(l)

    # print("lo,sc: ", lo, ", ", sc)
    ax[x, y].set_title("data: " + str(data) + ", loc: " + "{0:.2f}".format(lo) + ", scale: " + "{0:.2f}".format(sc))
    ax[x,y].hist(accepted_theta,density=True)
    ax[x,y].plot(l,p)
plt.show()
