
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from abc_rejection_sampling import abc_inference, abc_inference_marginal

import numpy as np

import pickle
import time
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
from vilar_all_species import Vilar_model
import vilar
from pred_true_plot import heatmap


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

# choose neural network model
nnm = CNNModel(input_shape=(ts_len,nr_of_species), output_shape=(15), con_len=3, con_layers=clay, dense_layers=[100,100,100],dataname='speciesC')

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

heatmap(true_thetas=test_thetas, pred_thetas=test_pred_d, dmin=dmin, dmax=dmax, true_point=None, pred_point=None)

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


nrs = 0

Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)
simulate = Vilar_.simulate
bins_nr=25
true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
obs_data = np.zeros((nrs,num_timestamps,1))
abc_pred = np.zeros((nrs,15))
# abc_post = np.zeros((nrs,15,bins_nr))
abc_post = []

prod = []


obs_data_big = pickle.load(open('datasets/' + modelname + '/obs_data_pack_1k.p', "rb" ) )
print("obs_data_big shape: ", obs_data_big.shape)

obs_data_big = obs_data_big[:,:,[6]]
print("obs_data_big shape: ", obs_data_big.shape)

accepted_para_hist = []

for i in range(10,10+nrs):
    print("i: ", i)

    od = obs_data_big[[i]]
    # print("od shape: ", od.shape)
    accepted_para, accepted_pred, data_pred = abc_inference_marginal(data=od, abc_trial_thetas=test_thetas,abc_trial_ts=test_ts, nnm=nnm, nr_of_accept = 1000)
    # print("data_pred shape: ", data_pred.shape)
    accepted_para_hist.append(accepted_para)
    data_pred = denormalize_data(data_pred,dmin,dmax)
    accepted_pred = denormalize_data(accepted_pred,dmin,dmax)
    nr_of_accept2 = 10
    # print("accepted para shape: ", accepted_para.shape)

    linew = 3
    f, ax = plt.subplots(3,5,figsize=(50,20))

    for x in range(3):
        for y in range(5):
            j = x*5+y
            points = int((1/test_ae_norm[i])**1.5)+1
            # points = int((1/np.std(normalize_data(accepted_para,dmin,dmax)[:,j]))**1.5)
            bins = np.linspace(dmin[j],dmax[j],points)
            ret = ax[x,y].hist(accepted_para[:,j], bins=bins, color='y',alpha=0.3)
            peakv = np.max(ret[0])
            ax[x,y].plot([dmin[j], dmin[j]], [peakv, 0], c='b', lw=linew)
            ax[x,y].plot([dmax[j], dmax[j]], [peakv, 0], c='b', lw=linew)
            ax[x,y].plot([data_pred[j], data_pred[j]], [peakv, 0],lw=linew, ls=':', c='silver')

            ax[x,y].plot([true_params[0][j], true_params[0][j]], [peakv, 0],lw=linew, ls='--', c='black')




    plt.savefig("check_corr_marginal_" + str(i))
    plt.close()

print("accepted_para_hist len: ", len(accepted_para_hist))
print("accepted_para_hist[0] shape: ", accepted_para_hist[0].shape)
f, ax = plt.subplots(3, 5, figsize=(50, 20))

for x in range(3):
    for y in range(5):
        j = x*5+y
        points = int((1/test_ae_norm[i])**1.5)+1
        # points = int((1/np.std(normalize_data(accepted_para,dmin,dmax)[:,j]))**1.5)
        bins = np.linspace(dmin[j],dmax[j],points)
        x_points = (np.array(bins[1:])+np.array(bins[0:-1]))/2
        print("points: ", points, ", x_points shape: ", x_points.shape)

        ap = []
        for k in range(nrs):

            print("k: ", k, ", j: ", j)
            print("accepted_para_hist[k] shape: ", accepted_para_hist[k].shape)
            # ap.append(accepted_para_hist[k][:,j])
            ap.append(np.histogram(accepted_para_hist[k][:,j],bins=bins, density=True)[0])

            print("ap shape: ", np.array(ap).shape)


        ap = np.prod(np.array(ap),0)
        ap = ap/np.trapz(ap,x_points)
        print("ap shape: ", ap.shape)
        # ret = ax[x,y].hist(ap, bins=bins, alpha=0.3)
        # peakv = np.max(ret[0])
        peakv = np.max(ap)
        ax[x,y].plot(x_points,ap)
        ax[x,y].plot([dmin[j], dmin[j]], [peakv, 0], c='b', lw=linew)
        ax[x,y].plot([dmax[j], dmax[j]], [peakv, 0], c='b', lw=linew)
        # ax[x,y].plot([data_pred[j], data_pred[j]], [peakv, 0],lw=linew, ls=':', c='silver')

        ax[x,y].plot([true_params[0][j], true_params[0][j]], [peakv, 0],lw=linew, ls='--', c='black')

    plt.savefig("check_corr_prod")



