
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from abc_inference_allspecies_func import abc_inference
import vilar
import numpy as np

import pickle
import time
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
from vilar_all_species import Vilar_model


plt.plot(np.random.rand(100))
plt.savefig('randomplot.png')

num_timestamps=401
endtime=200
modelname = "vilar_ACR_" + str(endtime) + "_" + str(num_timestamps) + '_all_species'
# parameter range
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]

#5,6,7,8
#6=C, 7=A, 8=R
species = [6]





clay=[32,48,64,96]
ts_len = 401
nr_of_species = len(species)
print("species: ", species)

# choose neural network model
nnm = CNNModel(input_shape=(ts_len,nr_of_species), output_shape=(15), con_len=3, con_layers=clay, dense_layers=[100,100,100])
# nnm = ANNModel(input_shape=(ts_len, train_ts.shape[2]), output_shape=(15), layers=[100,100,100])
# nnm = PEN_CNNModel(input_shape=(train_ts.shape[1],train_ts.shape[2]), output_shape=(15), pen_nr=3, con_layers=clay, dense_layers=[100,100,100])

print("Model name: ", nnm.name)


# true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
# true_param = np.squeeze(np.array(true_param))
# print("true_param shape: ", true_param.shape)
# data = pickle.load(open('datasets/' + modelname + '/obs_data_pack.p', "rb" ) )
# # data = data[3]
# # data = np.expand_dims(data,0)
# print("data shape: ", data.shape)

nnm.load_model()

test_thetas = pickle.load(open('datasets/' + modelname + '/test_thetas.p', "rb" ) )
test_ts = pickle.load(open('datasets/' + modelname + '/test_ts.p', "rb" ) )
test_ts = test_ts[:,:,species]
test_thetas_n = normalize_data(test_thetas,dmin,dmax)
test_pred = nnm.predict(test_ts)
print("test_pred shape: ", test_pred.shape)
test_pred = np.reshape(test_pred,(-1,15))
print("test_pred shape: ", test_pred.shape)

test_pred_d = denormalize_data(test_pred,dmin,dmax)
test_mse = np.mean((test_thetas-test_pred)**2)
test_mae = np.mean(abs(test_thetas-test_pred_d))
test_ae = np.mean(abs(test_thetas-test_pred_d),axis=0)
test_ae_norm = np.mean(abs(test_thetas_n-test_pred),axis=0)



f, ax = plt.subplots(3,5,figsize=(40,40))

for x in range(3):
    for y in range(5):
        i = x*5+y
        print("i: ", i)
        b = np.linspace(dmin[i], dmax[i], 26)
        ax[x,y].hist2d(test_thetas[:,i],test_pred_d[:,i], bins=b)
        ax[x,y].plot([dmin[i], dmin[i], dmax[i], dmax[i], dmin[i]], [dmin[i], dmax[i], dmax[i], dmin[i], dmin[i]], c='white',linewidth=10)

plt.savefig("heatmap_testset")





print("Model name: ", nnm.name)
print("mean square error: ", test_mse)
print("mean rel absolute error: ", np.mean(test_ae_norm))


nrs = 1000

Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)
simulate = Vilar_.simulate

true_params = [50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]
# obs_data = np.zeros((nrs,num_timestamps,1))
# abc_pred = np.zeros((nrs,15))
# abc_post = np.zeros((nrs,15,100))
#
# for i in range(nrs):
#     print("i: ", i)
#     od = simulate(np.array(true_params))[:,species]
#     print("od shape: ", od.shape)
#     obs_data[i,:,:] = od
#     # Mean_Vector, Cov_Matrix, Posterior_fit = abc_inference(data=np.expand_dims(od,0), true_param=true_params[0], abc_trial_thetas=test_thetas,
#     #                                         abc_trial_ts=test_ts, nnm=nnm, dmin=dmin, dmax=dmax, nr_of_accept=100,
#     #                                         nr_of_accept_cross=100,index=i)
#     # abc_pred[i] = Mean_Vector
#     # abc_post[i] = Posterior_fit
#
#
# pickle.dump( obs_data, open( 'datasets/' + modelname + '/obs_data_1k_pack.p', "wb" ) )

obs_data = pickle.load(open('datasets/' + modelname + '/obs_data_1k_pack.p', "rb" ) )[:,:,[6]]

print("obs_data shape: ", obs_data.shape)

pred_data = nnm.predict(obs_data)
pred_data = denormalize_data(pred_data,dmin,dmax)
print("pred_data shape: ", pred_data.shape)

para_names = vilar.get_parameter_names()

f, ax = plt.subplots(3,5,figsize=(40,40))

for x in range(3):
    for y in range(5):
        i = x*5+y
        print("i: ", i)

        b = np.linspace(dmin[i],dmax[i],11)
        d = ax[x,y].hist(pred_data[:,i], bins=b, density=True)
        peak_val = np.max(d[0])
        ax[x,y].plot([dmin[i], dmin[i]],[peak_val,0],c='b')
        ax[x,y].plot([dmax[i], dmax[i]],[peak_val,0],c='b')
        ax[x,y].plot([true_params[i], true_params[i]],[peak_val,0],c='black')
        ax[x,y].set_xlabel("true " + para_names,fontsize=20)
        ax[x,y].set_xlabel("predicted " + para_names, fontsize=20)



plt.savefig("prediction_dist")