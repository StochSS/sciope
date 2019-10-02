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


data_pack = pickle.load(open('datasets/' + modelname + '/obs_data_pack.p', "rb" ) )
data_pack_pred = nnm.predict(data_pack)
print("data_pack_pred shape: ", data_pack_pred.shape)


nr_of_trial = abc_trial_thetas.shape[0]
nr_of_accept = 1000
nr_of_accept_cross = 100


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


true_param = normalize_data(true_param,dmin,dmax)
# plt.axis('equal')
f, ax = plt.subplots(15,15,figsize=(30,30))# ,sharex=True,sharey=True)
f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial),fontsize=16)
bins_nr = 10
bins = np.linspace(0,1,bins_nr+1)
hist_data = np.ones((15,bins_nr))
hist_data_add = np.ones((15,bins_nr))
hist_data_all = np.ones((15,15,bins_nr))

lower, upper = 0, 1

def nnlf(params, data):
    loc, scale = params
    left_trunc_norm = (lower - loc)/scale
    right_trunc_norm = (upper - loc) / scale
    theta = (left_trunc_norm, right_trunc_norm, loc, scale)
    value = stats.truncnorm.nnlf(theta, data)
    return value

for x in range(15):
    ax[0, x].set_title(para_names[x])
    for y in range(x,15):
        print("x: ", x, ", y: ", y)
        print("abc_trial_pred.shape: ", abc_trial_pred.shape, ", data_pred.shape: ", data_pred.shape)
        if x == y:
            dist = abs(abc_trial_pred[:, x] - data_pred[x])
            accepted_ind = np.argpartition(dist, nr_of_accept)[0:nr_of_accept]
            accepted_para = abc_trial_thetas[accepted_ind]
            accepted_mean = np.mean(accepted_para, axis=0)

            loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para[:, x]), np.std(accepted_para[:, x])),
                                               args=(accepted_para[:, x],), disp=False)

            left_trunc_norm = (lower - loc_opt) / scale_opt
            right_trunc_norm = (upper - loc_opt) / scale_opt

            l = np.linspace(lower, upper, 100)
            p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)

            ax[y, x].plot(l, p, c='green')

            ret = ax[x, y].hist(accepted_para[:, x], density=True, bins=20, color='green', alpha=0.5)
            peak_val = np.maximum(np.max(ret[0]), np.max(p))
            ax[x, y].plot([true_param[x], true_param[x]], [0, peak_val], c='black')
            # ax[x, y].plot([accepted_mean[x], accepted_mean[x]], [0, peak_val], c='red')
            # ax[x, y].plot([data_pred[x], data_pred[x]], [0, peak_val], c='gray')

            ax[x, y].plot([1, 1], [0, peak_val], c='b')
            ax[x, y].plot([0, 0], [0, peak_val], c='b')



        else:
            dist = np.linalg.norm(abc_trial_pred[:, [x,y]] - data_pred[[x,y]], axis=1)
            accepted_ind = np.argpartition(dist, nr_of_accept_cross)[0:nr_of_accept_cross]
            accepted_para = abc_trial_thetas[accepted_ind]
            accepted_mean = np.mean(accepted_para, axis=0)
            print("accepted para shape: ", accepted_para.shape)
            print("(", x, ",", y, ") mean x: " + "{0:.2f}".format(np.mean(accepted_para[:, x])) + ", mean y: " + "{0:.2f}".format(np.mean(accepted_para[:, y])))

            # ax[x, y].scatter(abc_trial_thetas[:, y], abc_trial_thetas[:, x], color="yellow", s=2)
            ax[x, y].scatter(accepted_para[:, y], accepted_para[:, x], color="green", s=1, alpha=0.5)


            ax[x, y].scatter(true_param[y], true_param[x], color="black", marker="*")
            # ax[x, y].scatter(accepted_mean[y], accepted_mean[x], color="red", marker="x")
            # ax[x, y].scatter(data_pred[y], data_pred[x], color="gray", marker="o")
            ax[x, y].plot([0,1,1,0,0],[0,0,1,1,0])

            # loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para[:, y]), np.std(accepted_para[:, y])),
            #                                   args=(accepted_para[:, y],), disp=False)
            #
            # left_trunc_norm = (lower - loc_opt) / scale_opt
            # right_trunc_norm = (upper - loc_opt) / scale_opt
            #
            # l = np.linspace(lower, upper, 100)
            # p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)
            # ax[y, y].plot(l, p, c='red', alpha=0.2)
            # ax[x,y].hist(accepted_para[:, y], color='red', alpha=0.1, density=True)
            #
            # ax[x, y].plot(l, p, c='red', alpha=0.2)
            #
            #
            # loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para[:, x]), np.std(accepted_para[:, x])),
            #                                   args=(accepted_para[:, x],), disp=False)
            #
            # left_trunc_norm = (lower - loc_opt) / scale_opt
            # right_trunc_norm = (upper - loc_opt) / scale_opt
            #
            # l = np.linspace(lower, upper, 100)
            # p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)
            # ax[x, x].plot(l, p, c='gray', alpha=0.4)
            # ax[x,y].hist(accepted_para[:, x], color='gray', alpha=0.1, density=True)
            #
            # ax[x, y].plot(l, p, c='gray', alpha=0.4)



plt.savefig('posterior_abc_new')
