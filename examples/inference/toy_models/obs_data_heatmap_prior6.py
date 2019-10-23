
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from abc_inference_allspecies_func import abc_inference

import numpy as np

import pickle
import time
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
from vilar_all_species import Vilar_model



num_timestamps=401
endtime=200

modelname = "vilar_ACR_prior6_" + str(endtime) + "_" + str(num_timestamps)

dmin = [0,    100,    0,   20,   10,   1,    1,   0,   0,   0, 0.5,    0,   0,    0,   0]
dmax = [80,   600,    4,   60,   60,   7,   12,   2,   3, 0.7, 2.5,   4,   3,   70,   300]


#5,6,7,8
#6=C, 7=A, 8=R
species = [0]





clay=[32,48,64,96]
ts_len = 401
nr_of_species = len(species)
print("species: ", species)

# choose neural network model
nnm = CNNModel(input_shape=(ts_len,nr_of_species), output_shape=(15), con_len=3, con_layers=clay, dense_layers=[100,100,100],dataname='speciesC')
# nnm = ANNModel(input_shape=(ts_len, train_ts.shape[2]), output_shape=(15), layers=[100,100,100])
# nnm = PEN_CNNModel(input_shape=(train_ts.shape[1],train_ts.shape[2]), output_shape=(15), pen_nr=3, con_layers=clay, dense_layers=[100,100,100])

print("Model name: ", nnm.name)


true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
true_param = np.squeeze(np.array(true_param))
print("true_param shape: ", true_param.shape)
data = pickle.load(open('datasets/' + modelname + '/obs_data_pack_1k.p', "rb" ) )
# data = data[3]
# data = np.expand_dims(data,0)
print("data shape: ", data.shape)

nnm.load_model()




# nrs = 1000
#
Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)
simulate = Vilar_.simulate
#
# true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
# obs_data = np.zeros((nrs,num_timestamps,9))
# abc_pred = np.zeros((nrs,15))
# abc_post = np.zeros((nrs,15,100))
#
# for i in range(nrs):
#     print("i: ", i)
#     od = simulate(np.array(true_params))
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

obs_data = pickle.load(open('datasets/' + modelname + '/obs_data_pack_1k.p', "rb" ) )

pred_param = nnm.predict(obs_data[:,:,[6]])
pred_param = denormalize_data(pred_param,dmin,dmax)
print("pred param shape: ", pred_param.shape)
pred_param_m = np.mean(pred_param,0)
# print("start generating data")
# gen_data = np.array([simulate(pred_param_m) for i in range(1000)])
#
# pickle.dump( gen_data, open( 'datasets/' + modelname + '/gen_data_from_predicted_param_m.p', "wb" ) )
# print("done!")

gen_data = pickle.load(open('datasets/' + modelname + '/gen_data_from_predicted_param_m.p', "rb" ) )

nr_bins = 50

nr = int(obs_data.shape[0])
ts_len = int(obs_data.shape[1])
print("nr: ", nr, ", ts_len: ", ts_len)
print("obs_data shape: ", obs_data.shape)
density_data = np.zeros((3,ts_len,nr_bins))


specie = 6
peak_value = np.max(obs_data[:,:,specie])
bins = np.linspace(0, int(peak_value) + 1, nr_bins + 1)
for i in range(ts_len):
    # print("i: ", i)
    density_data[0,i] = np.histogram(obs_data[:,i,specie],bins=bins)[0]
    # print("den i: ", i, " - mean: ", np.mean(density_data[i]), ", std: ", np.std(density_data[i]), ", max: ", np.max(density_data[i]))

for i in range(ts_len):
    # print("i: ", i)
    density_data[1,i] = np.histogram(gen_data[:,i,specie],bins=bins)[0]
    # print("den i: ", i, " - mean: ", np.mean(density_data[i]), ", std: ", np.std(density_data[i]), ", max: ", np.max(density_data[i]))


plt.clf()
density_data = density_data[:,:,::-1]
density_data = density_data
density_data = density_data**(1/2)

f, ax = plt.subplots(2,1)#,figsize=(30,15))
ax[0].imshow(density_data[0].T, aspect='auto', extent=[0,201,0,peak_value])
ax[0].set_title('Species C from true parameters')
ax[0].set_xlabel('time')
ax[0].set_ylabel('# of species')
ax[1].imshow(density_data[1].T, aspect='auto', extent=[0,201,0,peak_value])
ax[1].set_title('Species C from predicted mean parameters')
ax[1].set_xlabel('time')
ax[1].set_ylabel('# of species')
plt.savefig('obs_data_density_speciesC_prior6')


# pred_param = denormalize_data(nnm.predict(obs_data),dmin,dmax)
#
# gen_data = np.zeros((nrs,num_timestamps,1))
#
# for i in range(nrs):
#     od = simulate(abc_pred[i])[:,species]
#     # print("od shape: ", od.shape)
#     gen_data[i,:,:] = od
#
#
# f,ax = plt.subplots(3,1,figsize=(45,15))
# linew = 1
# t= np.linspace(0,200,401)
# first = True
# for ts in obs_data:
#     rcol = np.random.rand(3)*np.array([0.2,0.2,0.5]) + np.array([0,0,0.5])
#
#     ax[0].plot(t,ts[:,0],c=rcol,lw=linew)
#     if first:
#         ax[2].plot(t,ts[:,0],c=rcol,label='true param.',lw=linew)
#         first = False
#     else:
#         ax[2].plot(t, ts[:, 0], c=rcol,lw=linew)
#
#
# ax[0].set_title("Specie C from true parameter")
# ax[1].set_title("Specie C from predicted parameter")
# ax[2].set_title("Specie C from both true parameter and predicted parameter")
# first = True
#
# for ts in gen_data:
#     rcol = np.random.rand(3)*np.array([0.5,0.2,0.2]) + np.array([0.5,0,0])
#     ax[1].plot(t,ts[:,0],c=rcol,lw=linew)
#     if first:
#         ax[2].plot(t,ts[:,0],c=rcol,label='pred param.',lw=linew)
#         first = False
#     else:
#         ax[2].plot(t,ts[:,0],c=rcol,lw=linew)
#
#
# ax[2].legend()
#
#
# plt.savefig('comp.png')
#
# od = [simulate(abc_pred[i])[:, species] for i in range(10)]
# f,ax = plt.subplots(3,1,figsize=(45,15))
# ax[0].plot(t,obs_data[0,:,0],c='b')
# for o in od:
#     ax[1].plot(t,o,c='r')
#     ax[2].plot(t,o,c='r')
#
#
# ax[2].plot(t,obs_data[0,:,0],c='b')
#
# plt.savefig('comp2.png')
#
#
#
# f,ax = plt.subplots(3,5,figsize=(15,25))
#
# for x in range(3):
#     for y in range(5):
#         i = x*5 +y
#         ax[x, y].plot([dmin[i], dmin[i]], [1, 0], c='black')
#         ax[x, y].plot([dmax[i], dmax[i]], [1, 0], c='black')
#
#         # ax[x,y].hist(pred_param[:,i], density=True)
#         for p in pred_param[:,i]:
#             ax[x, y].plot([p,p], [1, 0], c='r')
#
#         pm = np.mean(pred_param[:,i])
#         ax[x, y].plot([pm, pm], [1, 0], c='yellow')
#
#         ax[x,y].plot([true_param[i], true_param[i]], [1, 0],c='b')
#
# plt.savefig('dist.png')
#
#
# f,ax = plt.subplots(3,5,figsize=(15,25))
#
# for x in range(3):
#     for y in range(5):
#         i = x*5 +y
#         ax[x, y].plot([dmin[i], dmin[i]], [1, 0], c='black')
#         ax[x, y].plot([dmax[i], dmax[i]], [1, 0], c='black')
#
#         # ax[x,y].hist(pred_param[:,i], density=True)
#         l = np.linspace(dmin[i],dmax[i],100)
#         for p in abc_post[:,i]:
#             ax[x, y].plot(l,p, c='r')
#
#         pm = np.prod(abc_post,0)[i]
#         ax[x, y].plot(l,pm, c='yellow')
#
#         ax[x,y].plot([true_param[i], true_param[i]], [1, 0],c='b')
#
# plt.savefig('post.png')
#
#
#
#
#
#
#
#
#
#
#
#
#
