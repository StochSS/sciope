import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Def a linear data generator with gaussian noise
def gen_data(n=1000, m=2, k=4):
    x=np.random.rand(n,1)
    y=k*x + m + np.random.normal(0,1,(n,1))
    return x,y



x,y = gen_data(100000)

x_tst,y_tst = gen_data(100)
print("y shape: ", y.shape)
print("x shape: ", x.shape)


# Build model.
# negloglik = lambda y, p_y: -p_y.log_prob(y)

model = tf.keras.Sequential([
  keras.layers.Dense(1, input_shape=(1,)),

  tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])

# Do inference.
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
model.fit(x, y, epochs=5)#, verbose=False)

# Make predictions.
yhat = model(x_tst).eval()

print("type yhat: ", type(yhat))
print("shape yhat: ", yhat.shape)
yhat = np.asarray(yhat)
print("shape yhat: ", yhat.shape)

plt.scatter(x_tst,yhat)
plt.scatter(x_tst,y_tst)
plt.savefig('takeme')


