import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from load_data_from_julia import load_data
import numpy as np
from AutoRegressive_model import simulate, prior
# from MovingAverage_model import simulate, prior
from sklearn.metrics import mean_absolute_error
import pickle
from normalize_data import normalize_data
from load_data import load_spec


# choose neural network model
nnm = CNNModel(input_shape=(401,3), output_shape=(15))
# nnm = PEN_CNNModel(input_shape=(201,3), output_shape=(15), pen_nr=10)
# nm = ANNModel(input_shape=(100,1), output_shape=(2))

nnm.load_model()


#ABC algorithm


modelname = "vilar_ACR_200_401"
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]

true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
true_param = np.squeeze(np.array(true_param))
print("true_param shape: ", true_param.shape)
data = pickle.load(open('datasets/' + modelname + '/obs_data.p', "rb" ) )
print("data shape: ", data.shape)
# data_exp = np.expand_dims( np.expand_dims(data,axis=0), axis=2 )
# print("data_exp shape: ", data_exp.shape)

data_pred = nnm.predict(data)

abc_trial_thetas = pickle.load(open('datasets/' + modelname + '/abc_trial_thetas.p', "rb" ) )
abc_trial_thetas = normalize_data(abc_trial_thetas,dmin,dmax)
abc_trial_ts = pickle.load(open('datasets/' + modelname + '/abc_trial_ts.p', "rb" ) )
abc_trial_pred = nnm.predict(abc_trial_ts)
mean_dev = np.mean(abs(abc_trial_thetas-abc_trial_pred), axis=0)
print("mean dev shape: ", mean_dev.shape)
print("mean deviation(", np.mean(mean_dev), "):: ", mean_dev)

bpi = np.argsort(mean_dev)[:4] # best_param_ind


nr_of_trial = abc_trial_thetas.shape[0]
nr_of_accept = 100


dist = np.linalg.norm(abc_trial_pred[:,bpi] - data_pred[:,bpi],axis=1)
accepted_ind = np.argpartition(dist,nr_of_accept)[0:nr_of_accept]
accepted_para = abc_trial_thetas[accepted_ind]
accepted_mean = np.mean(accepted_para,axis=0)
accepted_std = np.std(accepted_para,axis=0)
print("posterior dev: ", accepted_mean-true_param)
print("posterior std: ", accepted_std)
print("accepted dist max: ", np.max(dist[accepted_ind]))
print("accepted dist mean: ", np.mean(dist[accepted_ind]))
print("trial dist mean: ", np.mean(dist))

data_pred = np.squeeze(data_pred)
accepted_dist = dist[accepted_ind]

print("accepted dist mean: ", np.mean(accepted_dist), ", max: ", np.max(accepted_dist), ", min: ", np.min(accepted_dist))


# bpi = np.argsort(accepted_std)[:4] # best_param_ind



true_param = normalize_data(true_param,dmin,dmax)
# plt.axis('equal')
f, ax = plt.subplots(4,4,figsize=(15,15))# ,sharex=True,sharey=True)
f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial),fontsize=16)
for i in range(16):
    x = i // 4
    y = i % 4
    if x == y:
        ret = ax[x, y].hist(accepted_para[:, bpi[x]], density=True)
        peak_val = np.max(ret[0])
        ax[x, y].plot([true_param[bpi[x]], true_param[bpi[x]]], [0,peak_val], c='black')
        ax[x, y].plot([accepted_mean[bpi[x]], accepted_mean[bpi[x]]], [0,peak_val], c='red')

        ax[x, y].plot([1, 1], [0, peak_val], c='b')
        ax[x, y].plot([0, 0], [0, peak_val], c='b')
    else:
        ax[x,y].scatter(accepted_para[:, bpi[x]], accepted_para[:, bpi[y]], color="green", s=2)
        ax[x,y].scatter(true_param[bpi[x]],true_param[bpi[y]], color="black", marker="*")
        ax[x,y].scatter(accepted_mean[bpi[x]],accepted_mean[bpi[y]], color="red", marker="x")
        ax[x,y].scatter(data_pred[bpi[x]],data_pred[bpi[y]], color="gray", marker="o")
        ax[x, y].plot([0,1,1,0,0],[0,0,1,1,0])
# plt.scatter(more_pred[:,0],more_pred[:,1], color="gold", marker="o")

# plt.plot([0,1,1,0,0],[0,0,1,1,0])

#plt.plot([-2,2,0,-2],[1,1,-1,1],color="red")
#plt.plot([-2,2,0,-2],[-1,-1,1,-1],color="red")


plt.savefig('posterior_abc')