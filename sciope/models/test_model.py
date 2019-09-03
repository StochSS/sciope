import tensorflow as tf
import tensorflow_probability as tfp
import numpy
tfd = tfp.distributions
import matplotlib.pyplot as plt


# Def a linear data generator with gaussian noise
def gen_data(n=1000, m=2, k=4):
    x=np.random.rand(n)
    y=k*x + m + np.random.normal(0,1,n)
    return x,y



x,y = gen_data(100000)

x_tst,y_tst = gen_data(100)


# Build model.
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])

# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x, y, epochs=500, verbose=False)

# Make predictions.
yhat = model(x_tst)

plt.plot(x_tst,yhat)
plt.plot(x_tst,y_tst)
plt.show()



