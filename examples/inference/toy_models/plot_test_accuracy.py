import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import numpy as np
import pickle
from normalize_data import normalize_data, denormalize_data
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel

endtime = 200
num_timestamps = 401

modelname = "vilar_ACR_prior5_" + str(endtime) + "_" + str(num_timestamps)


dmin = [40, 200,  0,  20, 10, 1,  1, 0, 0,  0,  0.5, 0.2, 0, 0, 20]
dmax = [80, 600, 0.1, 60, 60, 7, 12, 2, 3, 0.7, 2.5,  2,  3, 70, 120]



ts_len = 401
# choose neural network model
nnm = CNNModel(input_shape=(ts_len,3), output_shape=(15), con_len=2, con_layers=[25,50,100])
# nnm = PEN_CNNModel(input_shape=(ts_len,3), output_shape=(15), pen_nr=10)
# nnm = ANNModel(input_shape=(ts_len, 3), output_shape=(15))

nnm.load_model()
print("model loaded")

test_thetas = pickle.load(open('datasets/' + modelname + '/test_thetas.p', "rb" ) )
test_ts = pickle.load(open('datasets/' + modelname + '/test_ts.p', "rb" ) )
test_pred = nnm.predict(test_ts)
test_pred = np.reshape(test_pred,(-1,15))
test_pred = denormalize_data(test_pred,dmin,dmax)

print("are we here?")
f, ax = plt.subplots(3,5,figsize=(30,30))# ,sharex=True,sharey=True)

for x in range(5):
    for y in range(3):
        i = x*3 + y
        ax[y,x].scatter(test_thetas[:,i],test_pred[:,i])
        ax[y,x].plot([dmin[i], dmin[i], dmax[i], dmax[i],dmin[i]],[dmin[i], dmax[i], dmax[i],dmin[i], dmin[i]])

plt.savefig('testplot')
