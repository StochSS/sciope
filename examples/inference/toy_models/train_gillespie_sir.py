import numpy as np
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
import pickle
import time
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
import matplotlib.pyplot as plt
import os
from gillespie_algorithm import *
from scipy import stats, optimize


# ex S-I-R model
channels = 3
nr_of_thetas = 2

transitions = np.array([[-1, 1, 0],[0, -1, 1]])
eq = lambda v: np.array([v[0]*v[1], v[1]])
theta = np.array([1, 20])

init = np.array([16, 4, 0])
delta_t, final_t = 0.1, 5
GM = Gillespie_model(channels, transitions, eq, theta, delta_t, final_t, init)
dmin = np.array([0, 0])
dmax = np.array([1, 1])

dmin_hist = []
dmax_hist = []
dmin_hist.append(dmin)
dmax_hist.append(dmax)
modelname = "SIRmodel_" + str(delta_t) + "_" + str(final_t) + "_"+ str(dmin) + "_" + str(dmax)

#Defining true parameter value and generating data from it
true_param = np.array([0.7, 0.8])
GM.theta = true_param
t, data = GM.simulate_discreate()
data = np.array([data])
inference_rounds = 1
data_pred_hist = []
print("data shape: ", data.shape)
for i in range(channels):
    plt.plot(t, data[0,:, i])

plt.show()


def nnlf(params, data):
    # print("params: ", params)

    loc = params[0]
    scale = params[1]
    lower = params[2]
    upper = params[3]
    # print("loc, scale, lower, upper: ", loc, scale, lower, upper)

    left_trunc_norm = (lower - loc)/scale
    right_trunc_norm = (upper - loc) / scale
    theta = (left_trunc_norm, right_trunc_norm, loc, scale)
    value = stats.truncnorm.nnlf(theta, data)
    return value

print("update?")
print("data shape: ", data.shape)
ts_len = data.shape[1]



for rounds in range(inference_rounds):

    modelname = "SIRmodel_" + str(delta_t) + "_" + str(final_t) + "_" + str(dmin) + "_" + str(dmax)


    if rounds == 0:
        save_data = True
    else:
        save_data = False
    train_thetas, train_ts, validation_thetas, validation_ts = load_data(GM, dmin, dmax, modelname,
                              packs=["training", "validation"], nr=[100000, 10000], save=save_data)

    #Normalize parameter values
    train_thetas = normalize_data(train_thetas,dmin,dmax)
    validation_thetas = normalize_data(validation_thetas,dmin,dmax)
    #train model
    training_epochs = 20
    # choose neural network model
    # nnm = PEN_CNNModel(input_shape=(ts_len,2), output_shape=(2), pen_nr=10)
    # nnm = ANNModel(input_shape=(ts_len, 2), output_shape=(2),modelname=modelname)

    # nnm.load_model()
    print("train_ts shape: ", train_ts.shape)
    nnm = CNNModel(input_shape=(ts_len, channels), output_shape=(nr_of_thetas), con_len=3, con_layers=[10,20,40,80,160])
    nnm.train(inputs=train_ts, targets=train_thetas,validation_inputs=validation_ts,validation_targets=validation_thetas,
              batch_size=5, epochs=training_epochs, learning_rate=0.001, plot_training_progress=False)

    validation_pred = nnm.predict(validation_ts)
    validation_deviation = abs(validation_thetas - validation_pred)
    mean_validation_deviation = np.mean(validation_deviation,axis=0)
    print("mean validation deviation: ", mean_validation_deviation)

    data_pred = nnm.predict(data)
    data_pred_denorm = denormalize_data(data_pred, dmin, dmax)
    data_pred_hist.append(data_pred_denorm)

    trial_thetas, trial_ts = load_data(GM, dmin, dmax, modelname, packs=["trial"], nr=[100000], save=save_data)

    trial_pred = nnm.predict(trial_ts)
    trial_pred_denorm = denormalize_data(trial_pred, dmin, dmax)

    mean_trial_deviation = np.mean(abs(normalize_data(trial_thetas, dmin, dmax) - trial_pred), axis=0)
    print("mean trial deviation: ", mean_trial_deviation)

    nr_of_accept = 200
    nr_of_trial = trial_thetas.shape[0]
    dist = np.linalg.norm(trial_pred - data_pred, axis=1)
    print("dist shape: ", dist.shape)
    accepted_ind = np.argpartition(dist, nr_of_accept)[0:nr_of_accept]

    print("max accepted dist: ", np.max(dist[accepted_ind]))

    print("accepted_ind shape: ", accepted_ind.shape)
    # accepted_pred = trial_pred[accepted_ind]
    accepted_para = trial_thetas[accepted_ind]
    # new_min = np.min(accepted_para, axis=0)
    # new_max = np.max(accepted_para, axis=0)
    loc_opt = np.zeros(nr_of_thetas)
    scale_opt = np.zeros(nr_of_thetas)
    for x in range(nr_of_thetas):
        # print("x: ", x)
        ret = optimize.fmin(nnlf, (np.mean(accepted_para[:, x]), np.std(accepted_para[:, x]),dmin[x],dmax[x]),
                                       args=(accepted_para[:, x],), disp=False)
        # print("ret: ", ret)
        loc_opt[x], scale_opt[x] = ret[:2]
    loc_opt[loc_opt<0] = 0
    new_min = loc_opt - scale_opt*2
    new_max = loc_opt + scale_opt*2
    new_min[new_min<0]=0
    print("new_min: ", new_min)
    print("new_max: ", new_max)

    f, ax = plt.subplots(2, 2, figsize=(20, 20))  # ,sharex=True,sharey=True)
    # plt.axis('equal')
    print("data pred shape: ", data_pred.shape)
    f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial), fontsize=16)
    bins_nr = 10
    bins = np.linspace(0, 1, bins_nr + 1)
    for x in range(2):
        for y in range(x, 2):
            if x == y:
                ret = ax[x, y].hist(accepted_para[:, x], density=True)
                peak_val = np.max(ret[0])
                ax[y, x].plot([true_param[x], true_param[x]], [0, peak_val], c='black', lw=4)
                ax[y, x].plot([dmin[x], dmin[x]], [0, peak_val], c='b')
                ax[y, x].plot([dmax[x], dmax[x]], [0, peak_val], c='b')
                ax[y, x].plot([new_min[x], new_min[x]], [0, peak_val], c='r')
                ax[y, x].plot([new_max[x], new_max[x]], [0, peak_val], c='r')

                ax[y, x].plot([loc_opt[x], loc_opt[x]], [0, peak_val], c='r')
                left_trunc_norm = (dmin[x] - loc_opt[x]) / scale_opt[x]
                right_trunc_norm = (dmax[x] - loc_opt[x]) / scale_opt[x]

                l = np.linspace(dmin[x], dmax[x], 100)
                p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt[x], scale_opt[x])
                ax[y, x].plot(l, p, c='r')

            else:
                ax[x, y].scatter(accepted_para[:, y], accepted_para[:, x], c='blue', s=0.5)

                ax[x, y].scatter(data_pred_denorm[0, y], data_pred_denorm[0, x], color="gray", marker="o")
                ax[x, y].plot([dmin[1], dmin[1], dmax[1], dmax[1], dmin[1]],
                              [dmin[0], dmax[0], dmax[0], dmin[0], dmin[0]])
                ax[x, y].plot([new_min[1], new_min[1], new_max[1], new_max[1], new_min[1]],
                              [new_min[0], new_max[0], new_max[0], new_min[0], new_min[0]], c='r')

                ax[x, y].scatter(loc_opt[y], loc_opt[x], c='r', marker='*')


                ax[x, y].scatter(true_param[y], true_param[x], color="black", marker="*")
                for i in range(rounds):
                    ax[x, y].plot([dmin_hist[i][1], dmin_hist[i][1], dmax_hist[i][1], dmax_hist[i][1], dmin_hist[i][1]],
                              [dmin_hist[i][0], dmax_hist[i][0], dmax_hist[i][0], dmin_hist[i][0], dmin_hist[i][0]])

    # plt.show()
    plt.savefig('sis_abc_inference' + str(rounds))

    dmin = new_min
    dmax = new_max
    dmin_hist.append(dmin)
    dmax_hist.append(dmax)


