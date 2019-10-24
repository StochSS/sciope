
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from abc_inference_diag_hist_full import abc_inference

import numpy as np

import pickle
import time
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
from vilar_all_species import Vilar_model
import vilar



def plot(posterior,bins,nr,dmin,dmax,true_param):
    f, ax = plt.subplots(5, 3, figsize=(30, 30))  # ,sharex=True,sharey=True)
    # f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial), fontsize=16)

    for i in range(15):

        bin_c = (np.array(bins[i][:-1])+np.array(bins[i][1:]))/2
        print("i: ", i, ", bin_c shape: ", bin_c.shape)
        y = i % 3
        x = i // 3
        ax[x, y].plot(bin_c,posterior[i])
        peakv = np.max(posterior[i])
        ax[x, y].plot([true_param[i], true_param[i]],[peakv, 0])

    plt.savefig('posterior_plots/posterior_abc_prod' + str(nr))
    plt.close()



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
end_step=401
step = 1
train_ts = train_ts[:,:end_step:step,species]



print("Model name: ", nnm.name)
print("mean square error: ", test_mse)
print("mean rel absolute error: ", np.mean(test_ae_norm))


#create bins
bins = []
nr_of_bins = []
for i in range(15):
    nr_of_bins.append(int(4/test_ae_norm[i]))
    bin_ = np.linspace(dmin[i],dmax[i], nr_of_bins[i]+1)
    bins.append(bin_)


nrs = 10

Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)
simulate = Vilar_.simulate
bins_nr=25
true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
obs_data = np.zeros((nrs,num_timestamps,1))
abc_pred = np.zeros((nrs,15))
# abc_post = np.zeros((nrs,15,bins_nr))
abc_post = []

prod = []

# print("start generationg obs data pack big")
# obs_data_big = np.array([simulate(np.array(true_params)) for i in range(1000)])
#
# pickle.dump( obs_data_big, open( 'datasets/' + modelname + '/obs_data_pack_1k.p', "wb" ) )
# print("done generationg obs data pack big")

obs_data_big = pickle.load(open('datasets/' + modelname + '/obs_data_pack_1k.p', "rb" ) )
print("obs_data_big shape: ", obs_data_big.shape)

obs_data_big = obs_data_big[:,:,[6]]
print("obs_data_big shape: ", obs_data_big.shape)

for i in range(nrs):
    print("i: ", i)
    # od = simulate(np.array(true_params))[:,[6]]
    od = obs_data_big[i]
    print("od shape: ", od.shape)
    obs_data[i,:,:] = od
    Posterior_fit = abc_inference(data=np.expand_dims(od,0), true_param=true_params[0], abc_trial_thetas=train_thetas,
                                            abc_trial_ts=train_ts, nnm=nnm, dmin=dmin, dmax=dmax, bins=bins, nr_of_accept=500,
                                            index=i)
    # abc_post.append(Posterior_fit)
    #
    #
    # if i == 0:
    #     prod = Posterior_fit
    # else:
    #     for i in range(15):
    #         prod[i] *= Posterior_fit[i]
    #
    # print("prod shape: ", prod.shape)
    # for e in prod:
    #     print("e shape: ", e.shape)
    # plot(prod, bins=bins, nr=i, dmin=dmin, dmax=dmax, true_param=true_params[0])

pred_param = nnm.predict(obs_data_big)
pred_param = denormalize_data(pred_param,dmin,dmax)

pred_param_m = np.mean(pred_param,0)

gen_data = np.zeros((nrs,num_timestamps,1))
abc_pred_m = np.mean(abc_pred,0)
for i in range(nrs):
    od = simulate(pred_param_m)[:,[6]]
    # print("od shape: ", od.shape)
    gen_data[i,:,:] = od


f,ax = plt.subplots(3,1,figsize=(45,15))
linew = 1
t= np.linspace(0,200,401)
first = True
for ts in obs_data:
    rcol = np.random.rand(3)*np.array([0.2,0.2,0.5]) + np.array([0,0,0.5])
    print("obs data ts shape: ", ts.shape)
    ax[0].plot(t,ts[:,0],c=rcol,lw=linew)
    if first:
        ax[2].plot(t,ts[:,0],c=rcol,label='true param.',lw=linew)
        first = False
    else:
        ax[2].plot(t, ts[:, 0], c=rcol,lw=linew)


ax[0].set_title("Specie C from true parameter")
ax[1].set_title("Specie C from predicted parameter")
ax[2].set_title("Specie C from both true parameter and predicted parameter")
first = True

for ts in gen_data:
    print("gen data ts shape: ", ts.shape)

    rcol = np.random.rand(3)*np.array([0.5,0.2,0.2]) + np.array([0.5,0,0])
    ax[1].plot(t,ts[:,0],c=rcol,lw=linew)
    if first:
        ax[2].plot(t,ts[:,0],c=rcol,label='pred param.',lw=linew)
        first = False
    else:
        ax[2].plot(t,ts[:,0],c=rcol,lw=linew)


ax[2].legend()


plt.savefig('comp_prior6.png')

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
para_names = vilar.get_parameter_names()

print("pred param shape: ", pred_param.shape)
print("true_param: ", true_param)
f,ax = plt.subplots(3,5,figsize=(40,20))

for x in range(3):
    for y in range(5):
        i = x*5 +y


        ret = ax[x,y].hist(pred_param[:,i], density=True,bins=25)
        peakv = np.max(ret[0])
        # for p in pred_param[:,i]:
        #     ax[x, y].plot([p,p], [1, 0], c='r')
        print("i: ", i, ", peakv: ", peakv)
        pm = np.mean(pred_param[:,i])
        ax[x, y].plot([pm, pm], [peakv, 0], c='yellow')
        ax[x, y].plot([dmin[i], dmin[i]], [peakv, 0], c='blue')
        ax[x, y].plot([dmax[i], dmax[i]], [peakv, 0], c='blue')
        ax[x, y].plot([true_param[i], true_param[i]], [peakv, 0],c='black', ls='--')
        ax[x, y].set_xlabel("predicted " + para_names[i])

plt.savefig('dist_prior6.png')
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








