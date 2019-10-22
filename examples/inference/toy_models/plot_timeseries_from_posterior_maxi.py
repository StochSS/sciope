
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from abc_inference_diag_hist import abc_inference

import numpy as np

import pickle
import time
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
from vilar_all_species import Vilar_model



def plot(posterior,bins_nr,nr,dmin,dmax,true_param):
    f, ax = plt.subplots(5, 3, figsize=(30, 30))  # ,sharex=True,sharey=True)
    # f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial), fontsize=16)

    for i in range(15):
        bins = np.linspace(dmin[i], dmax[i], bins_nr + 1)
        bin_c = (bins[:-1]+bins[1:])/2
        y = i % 3
        x = i // 3
        ax[x, y].plot(bin_c,posterior[i])
        peakv = np.max(posterior[i])
        ax[x, y].plot[[true_param[i], true_param[i]],[peakv, 0]]

    plt.savefig('posterior_plots/posterior_abc_prod' + str(nr))


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


true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
true_param = np.squeeze(np.array(true_param))
print("true_param shape: ", true_param.shape)
data = pickle.load(open('datasets/' + modelname + '/obs_data_pack.p', "rb" ) )
# data = data[3]
# data = np.expand_dims(data,0)
print("data shape: ", data.shape)

nnm.load_model()


test_thetas = pickle.load(open('datasets/' + modelname + '/test_thetas.p', "rb" ) )
test_ts = pickle.load(open('datasets/' + modelname + '/test_ts.p', "rb" ) )
test_ts = test_ts[:,:,species]
test_thetas_n = normalize_data(test_thetas,dmin,dmax)
test_pred = nnm.predict(test_ts)
test_pred = np.reshape(test_pred,(-1,15))
test_pred_d = denormalize_data(test_pred,dmin,dmax)
test_mse = np.mean((test_thetas-test_pred)**2)
test_mae = np.mean(abs(test_thetas-test_pred_d))
test_ae = np.mean(abs(test_thetas-test_pred_d),axis=0)
test_ae_norm = np.mean(abs(test_thetas_n-test_pred),axis=0)

train_thetas, train_ts = load_spec(modelname=modelname, type = "train", species=species)
print("train_ts shape: ", train_ts.shape)




print("Model name: ", nnm.name)
print("mean square error: ", test_mse)
print("mean rel absolute error: ", np.mean(test_ae_norm))

nrs = 10

Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)
simulate = Vilar_.simulate
bins_nr=50
true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
obs_data = np.zeros((nrs,num_timestamps,1))
abc_pred = np.zeros((nrs,15))
abc_post = np.zeros((nrs,15,bins_nr))

for i in range(nrs):
    print("i: ", i)
    od = simulate(np.array(true_params))[:,species]
    print("od shape: ", od.shape)
    obs_data[i,:,:] = od
    Posterior_fit = abc_inference(data=np.expand_dims(od,0), true_param=true_params[0], abc_trial_thetas=test_thetas,
                                            abc_trial_ts=test_ts, nnm=nnm, dmin=dmin, dmax=dmax, nr_of_accept=1000,
                                            index=i,bins_nr = bins_nr)
    abc_post[i] = Posterior_fit

    plot(np.prod(abc_post[:i+1],0), bins_nr=bins_nr, nr=i, dmin=dmin, dmax=dmax, true_param=true_params)


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








