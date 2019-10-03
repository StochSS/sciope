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
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
import vilar
from scipy import stats, optimize



# choose neural network model
# nnm = CNNModel(input_shape=(401,3), output_shape=(15))

# nnm = PEN_CNNModel(input_shape=(201,3), output_shape=(15), pen_nr=10)
# nm = ANNModel(input_shape=(100,1), output_shape=(2))
clay=[32,48,64,96]
nnm = CNNModel(input_shape=(401,3), output_shape=15, con_len=3, con_layers=clay, dense_layers=[200,200,200])
nnm.load_model('saved_models/cnn_light10')


#ABC algorithm


num_timestamps=401
endtime=200

modelname = "vilar_ACR_prior6_" + str(endtime) + "_" + str(num_timestamps)

dmin = [0,    100,    0,   20,   10,   1,    1,   0,   0,   0, 0.5,    0,   0,    0,   0]
dmax = [80,   600,    4,   60,   60,   7,   12,   2,   3, 0.7, 2.5,   4,   3,   70,   300]

true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
true_param = np.squeeze(np.array(true_param))
print("true_param shape: ", true_param.shape)
data = pickle.load(open('datasets/' + modelname + '/obs_data.p', "rb" ) )
data = np.expand_dims(data,0)
print("data shape: ", data.shape)
# data_exp = np.expand_dims( np.expand_dims(data,axis=0), axis=2 )
# print("data_exp shape: ", data_exp.shape)

data_pred = nnm.predict(data)


abc_trial_thetas = pickle.load(open('datasets/' + modelname + '/abc_trial_thetas.p', "rb" ) )
# abc_trial_thetas = normalize_data(abc_trial_thetas,dmin,dmax)
abc_trial_ts = pickle.load(open('datasets/' + modelname + '/abc_trial_ts.p', "rb" ) )

abc_trial_thetas, abc_trial_ts  = load_spec(modelname=modelname, type = "train")


abc_trial_pred = nnm.predict(abc_trial_ts)
mean_dev = np.mean(abs(abc_trial_thetas-abc_trial_pred), axis=0)
print("mean dev shape: ", mean_dev.shape)
print("mean deviation(", np.mean(mean_dev), "):: ", mean_dev)

bpi = np.argsort(mean_dev)[:4] # best_param_ind


data_pack = pickle.load(open('datasets/' + modelname + '/obs_data_pack.p', "rb" ) )
data_pack_pred = nnm.predict(data_pack)
print("data_pack_pred shape: ", data_pack_pred.shape)


nr_of_trial = abc_trial_thetas.shape[0]
nr_of_accept = 1000
nr_of_accept_cross = 1000
range_color = '#1f77b4'
fsize=30
lwith=3
scattersize = 5


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

para_names = vilar.get_parameter_names()

for i in range(15):
    pk = para_names[i]
    pks = pk.split("_")
    if len(pks) > 1:
        pk_p = "\hat{\\" + pks[0].lower() + "}_{" + pks[1].upper() + "}"
        pk = pks[0].lower() + "_{" + pks[1].upper() + "}"
    if len(pks) == 3:
        print("len 3: ", pks[2])
        if pks[2] == 'prime':
            pk_p = pk_p + "'"
            pk = pk + "'"

    para_name_p = "$" + pk_p + "$"
    para_names[i] = "$\\" + pk + "$"


# true_param = normalize_data(true_param,dmin,dmax)
# plt.axis('equal')

# plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
# plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

f, ax = plt.subplots(15,15,figsize=(60,60))# ,sharex=True ) #,sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
# f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial),fontsize=16)
bins_nr = 10
bins = np.linspace(0,1,bins_nr+1)
hist_data = np.ones((15,bins_nr))
hist_data_add = np.ones((15,bins_nr))
hist_data_all = np.ones((15,15,bins_nr))
# gaussian_data_all = np.zeros((15,15,100))


# lower, upper = 0, 1

def nnlf(params, datapack):
    loc, scale,  = params
    data, lower, upper = datapack
    left_trunc_norm = (lower - loc)/scale
    right_trunc_norm = (upper - loc) / scale
    theta = (left_trunc_norm, right_trunc_norm, loc, scale)
    value = stats.truncnorm.nnlf(theta, data)
    return value

def smart_ticks(dmin,dmax):
    range = dmax-dmin
    if range > 80:
        ticks = list(np.arange(dmin,dmax,20)) + [dmax]
    elif range > 40:
        ticks = list(np.arange(dmin, dmax, 10)) + [dmax]
    elif range > 20:
        ticks = list(np.arange(dmin, dmax, 5)) + [dmax]
    elif range > 8:
        ticks = list(np.arange(dmin, dmax, 2)) + [dmax]
    elif range > 4:
        ticks = list(np.arange(dmin, dmax, 1)) + [dmax]
    elif range > 1:
        ticks = list(np.arange(dmin, dmax, 0.5)) + [dmax]
    else:
        ticks = list(np.around(np.linspace(dmin,dmax,5),2))

    return ticks

dist = np.linalg.norm(abc_trial_pred - data_pred, axis=1)
accepted_ind = np.argpartition(dist, nr_of_accept_cross)[0:nr_of_accept_cross]
accepted_para_full = abc_trial_thetas[accepted_ind]

for x in range(15):
    ax[0, x].set_title(para_names[x], fontsize=fsize)
    if x<14:
        ax[x,14].yaxis.set_label_position("right")
        ax[x,14].set_ylabel(para_names[x], rotation=0, fontsize=fsize, labelpad=20)


    for y in range(x,15):
        print("x: ", x, ", y: ", y)
        print("abc_trial_pred.shape: ", abc_trial_pred.shape, ", data_pred.shape: ", data_pred.shape)
        if x == y:

            #Real posterior
            loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para_full[:, x]), np.std(accepted_para_full[:, x])),
                                               args=([accepted_para_full[:, x], dmin[x], dmax[x]],), disp=False)

            left_trunc_norm = (dmin[x] - loc_opt) / scale_opt
            right_trunc_norm = (dmax[x] - loc_opt) / scale_opt

            l = np.linspace(dmin[x], dmax[x], 100)
            p_full = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)

            dist = abs(abc_trial_pred[:, x] - data_pred[x])
            accepted_ind = np.argpartition(dist, nr_of_accept)[0:nr_of_accept]
            accepted_para = abc_trial_thetas[accepted_ind]
            # accepted_mean = np.mean(accepted_para, axis=0)

            loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para[:, x]), np.std(accepted_para[:, x])),
                                               args=([accepted_para[:, x],dmin[x],dmax[x]],), disp=False)

            left_trunc_norm = (dmin[x] - loc_opt) / scale_opt
            right_trunc_norm = (dmax[x] - loc_opt) / scale_opt

            l = np.linspace(dmin[x], dmax[x], 100)
            p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)



            ax[x, y].tick_params(labelleft=False)
            ax[y, x].plot(l, p, c='green', lw=lwith)
            ax[y, x].plot(l, p_full, c='red', lw=lwith, ls='-')
            ax[x, y].set_xlabel(para_names[x], fontsize=fsize)
            # ax[x, y].yaxis.set_label_position("left")
            ax[x, y].set_ylabel('density', fontsize=fsize, rotation=90)

            ret = ax[x, y].hist(accepted_para[:, x], density=True, bins=25, color='green', alpha=1)
            ax[x, y].hist(accepted_para_full[:, x], density=True, bins=25, color='red', alpha=0.1)

            peak_val = np.maximum(np.max(ret[0]), np.max(p))
            ax[x, y].plot([true_param[x], true_param[x]], [0, peak_val], c='black')
            # ax[x, y].plot([accepted_mean[x], accepted_mean[x]], [0, peak_val], c='red')
            # ax[x, y].plot([data_pred[x], data_pred[x]], [0, peak_val], c='gray')

            ax[x, y].plot([dmax[x], dmax[x]], [0, peak_val], c=range_color, lw=lwith)
            ax[x, y].plot([dmin[x], dmin[x]], [0, peak_val], c=range_color, lw=lwith)



        else:
            dist = np.linalg.norm(abc_trial_pred[:, [x,y]] - data_pred[[x,y]], axis=1)
            accepted_ind = np.argpartition(dist, nr_of_accept_cross)[0:nr_of_accept_cross]
            accepted_para = abc_trial_thetas[accepted_ind]
            accepted_mean = np.mean(accepted_para, axis=0)
            print("accepted para shape: ", accepted_para.shape)
            print("(", x, ",", y, ") mean x: " + "{0:.2f}".format(np.mean(accepted_para[:, x])) + ", mean y: " + "{0:.2f}".format(np.mean(accepted_para[:, y])))

            # ax[x, y].scatter(abc_trial_thetas[:, y], abc_trial_thetas[:, x], color="yellow", s=2)
            ax[x, y].scatter(accepted_para_full[:, y], accepted_para_full[:, x], color="red", s=scattersize, alpha=0.1)

            ax[x, y].scatter(accepted_para[:, y], accepted_para[:, x], color="green", s=scattersize, alpha=1)

            ax[x, y].scatter(true_param[y], true_param[x], color="black", marker="x", s=20)
            # ax[x, y].scatter(accepted_mean[y], accepted_mean[x], color="red", marker="x")
            # ax[x, y].scatter(data_pred[y], data_pred[x], color="gray", marker="o")
            ax[x, y].plot([dmin[y], dmin[y], dmax[y], dmax[y], dmin[y]], [dmin[x], dmax[x], dmax[x], dmin[x], dmin[x]], lw=lwith, c = range_color)

            # loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para[:, x]), np.std(accepted_para[:, x])),
            #                                    args=([accepted_para[:, x], dmin[x], dmax[x]],), disp=False)
            #
            # left_trunc_norm = (dmin[x] - loc_opt) / scale_opt
            # right_trunc_norm = (dmax[x] - loc_opt) / scale_opt
            #
            # l = np.linspace(dmin[x], dmax[x], 100)
            # p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)
            # ax[x, x].plot(l, p, c='gray', alpha=0.2, lw=1)
            # gaussian_data_all[x,y,:] = p
            #
            #
            # loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para[:, y]), np.std(accepted_para[:, y])),
            #                                    args=([accepted_para[:, y], dmin[y], dmax[y]],), disp=False)
            #
            # left_trunc_norm = (dmin[y] - loc_opt) / scale_opt
            # right_trunc_norm = (dmax[y] - loc_opt) / scale_opt
            #
            # l = np.linspace(dmin[y], dmax[y], 100)
            # p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)
            # ax[y, y].plot(l, p, c='gray', alpha=0.2, lw = 1)
            # gaussian_data_all[y, x,:] = p



            if y < 15:
                ax[x, y].tick_params(labelbottom=False)

            ax[x, y].set_yticks(ticks=[dmin[x], dmax[x]], minor=False)
            ax[x, y].tick_params(labelleft=False)
            ax[x, y].tick_params(right=True)
            ax[x, y].tick_params(left=False)

            if y == 14:
                ax[x, y].tick_params(labelright=True)


            f.delaxes(ax[y, x])
        ax[x, y].set_xticks(ticks=[dmin[y], dmax[y]], minor=False)
        ax[x, y].tick_params(axis='both', which='major', labelsize=20)
        # ax.tick_params(axis='both', which='minor', labelsize=8)

# for x in range(15):
#     l = np.linspace(dmin[x], dmax[x], 100)
#     pr = np.mean(gaussian_data_all[x, :, :], axis=0)
#     print("pr shape: ", pr.shape)
#     ax[x,x].plot(l, pr, c='red')

print("updated posterior figure.")
plt.savefig('posterior_abc_new6')
