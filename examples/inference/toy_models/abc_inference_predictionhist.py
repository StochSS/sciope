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
from vilar import Vilar_model
from p_values import trunc_norm_pvalue


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
num_timestamps=401
endtime=200

# true_param = np.ones((15))*0.8


Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)


simulate = Vilar_.simulate
print("before simulation")
data = np.array([np.squeeze(simulate(normalize_data(true_param,dmin,dmax))) for i in range(1000)])
print("data shape: ", data.shape)

print("data shape: ", data.shape)
data_pred = nnm.predict(data)
data_pred = np.squeeze(data_pred)
data_pred_denorm = denormalize_data(data_pred,dmin,dmax)
print("data pred shape: ", data_pred.shape)
print("data_pred mae: ", np.mean(abs(data_pred - normalize_data(true_param, dmin, dmax))))
abc_trial_thetas = pickle.load(open('datasets/' + modelname + '/abc_trial_thetas.p', "rb" ) )
abc_trial_ts = pickle.load(open('datasets/' + modelname + '/abc_trial_ts.p', "rb" ) )
abc_trial_pred = nnm.predict(abc_trial_ts)

abc_trial_pred_denorm = denormalize_data(abc_trial_pred,dmin,dmax)



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


plt.axis('equal')
f, ax = plt.subplots(5,15,figsize=(90,20))# ,sharex=True,sharey=True)
# f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial),fontsize=16)
bins_nr = 10
bins = np.linspace(0,1,bins_nr+1)



# lower, upper = 0, 1
print("true param: ", true_param)
def nnlf(params, data,lower,upper):
    # print("inside nnlf: data shape: ", data.shape)
    # print("inside nnlf: lower, upper ", lower, upper)
    loc, scale = params
    left_trunc_norm = (lower - loc)/scale
    right_trunc_norm = (upper - loc) / scale
    theta = (left_trunc_norm, right_trunc_norm, loc, scale)
    value = stats.truncnorm.nnlf(theta, data)
    return value




for x in range(15):
    bins = np.linspace(dmin[x],dmax[x],10)
    ret = ax[0,x].hist(data_pred_denorm[:, x],bins=bins, density=True, color='green')
    peak_val = np.max(ret[0])

    ax[0,x].plot([true_param[x], true_param[x]], [0,peak_val], c='black', lw=4)
    ax[0,x].plot([dmax[x], dmax[x]], [0, peak_val], c='b')
    ax[0,x].plot([dmin[x], dmin[x]], [0, peak_val], c='b')

    ret = ax[1, x].hist(abc_trial_pred_denorm[:, x], density=True, color='green')
    peak_val = np.max(ret[0])

    ax[1, x].plot([true_param[x], true_param[x]], [0, peak_val], c='black', lw=4)
    ax[1, x].plot([dmax[x], dmax[x]], [0, peak_val], c='b')
    ax[1, x].plot([dmin[x], dmin[x]], [0, peak_val], c='b')

    # loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(data_pred_denorm[:, x]), np.std(data_pred_denorm[:, x])),
    #                                    args=(data_pred_denorm[:, x],dmin[x],dmax[x]), disp=False)
    #
    # left_trunc_norm = (dmin[x] - loc_opt) / scale_opt
    # right_trunc_norm = (dmax[x] - loc_opt) / scale_opt
    #
    # l = np.linspace(dmin[x], dmax[x], 1000)
    # p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)
    # ax[x].plot(l, p)
    # col ='red'
    # if loc_opt<dmin[x] or loc_opt>dmax[x]:
    #     col = 'orange'
    #     if loc_opt<dmin[x]:
    #         loc_opt=dmin[x]
    #     if loc_opt>dmax[x]:
    #         loc_opt=dmax[x]
    # ax[x].plot([loc_opt, loc_opt], [peak_val, 0], c=col, ls='--')







plt.savefig('posterior_abc_predict')

