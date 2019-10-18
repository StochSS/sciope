import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
import numpy as np
import pickle
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
import vilar
from scipy import stats, optimize



def abc_inference(data, true_param, abc_trial_thetas,abc_trial_ts, nnm,dmin,dmax, nr_of_accept = 100, nr_of_accept_cross=100):
    data_pred = nnm.predict(data)
    data_pred = np.squeeze(data_pred)
    lwith = 1
    range_color = 'blue'


    abc_trial_pred = nnm.predict(abc_trial_ts)

    nr_of_trial = abc_trial_thetas.shape[0]

    para_names = vilar.get_parameter_names()



    true_param = normalize_data(true_param,dmin,dmax)
    # plt.axis('equal')
    f, ax = plt.subplots(18,15,figsize=(30,30))# ,sharex=True,sharey=True)
    f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial),fontsize=16)
    bins_nr = 10
    bins = np.linspace(0,1,bins_nr+1)
    hist_data = np.ones((15,bins_nr))
    hist_data_add = np.ones((15,bins_nr))
    hist_data_all = np.ones((15,15,bins_nr))


    dist = np.linalg.norm(abc_trial_pred - data_pred, axis=1)
    print("dist shape: ", dist.shape)
    accepted_ind = np.argpartition(dist,nr_of_accept)
    print("accepted_ind shape: ", accepted_ind.shape)
    accepted_ind = np.argpartition(dist, nr_of_accept)[0:nr_of_accept]
    accepted_para = abc_trial_thetas[accepted_ind]
    accepted_mean = np.mean(accepted_para, axis=0)

    lower, upper = 0, 1
    bg=0

    def nnlf(params, data):
        loc, scale = params
        left_trunc_norm = (lower - loc)/scale
        right_trunc_norm = (upper - loc) / scale
        theta = (left_trunc_norm, right_trunc_norm, loc, scale)
        value = stats.truncnorm.nnlf(theta, data)
        return value

    Cov_Matrix = np.zeros((15,15))
    Mean_Vector = np.zeros(15)
    for x in range(15):
        ax[0, x].set_title(para_names[x])
        for y in range(x,15):

            if x == y:
                dist = abs(abc_trial_pred[:, x] - data_pred[x])
                accepted_ind = np.argpartition(dist, nr_of_accept)[0:nr_of_accept]
                accepted_para = abc_trial_thetas[accepted_ind]
                ret = ax[x, y].hist(accepted_para[:, x], density=True, bins=bins, color='green')
                peak_val = np.max(ret[0])
                ax[x, y].plot([true_param[x], true_param[x]], [0,peak_val], c='black', lw=4)
                # ax[x, y].plot([accepted_mean[x], accepted_mean[x]], [0,peak_val], c='red')
                ax[x, y].plot([data_pred[x], data_pred[x]], [0,peak_val], c='gray', ls='--')

                ax[x, y].plot([dmax[x], dmax[x]], [0, peak_val], c='b')
                ax[x, y].plot([dmin[x], dmax[x]], [0, peak_val], c='b')

                loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para[:, x]), np.std(accepted_para[:, x])),
                                                   args=(accepted_para[:, x],), disp=False)
                Cov_Matrix[x,y] = scale_opt
                Mean_Vector[y] = loc_opt
                left_trunc_norm = (lower - loc_opt) / scale_opt
                right_trunc_norm = (upper - loc_opt) / scale_opt

                l = np.linspace(lower, upper, 100)
                p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)

                ax[x, x].plot(l, p)
                col ='red'
                if loc_opt<lower or loc_opt>upper:
                    col = 'orange'
                    if loc_opt<lower:
                        loc_opt=lower
                    if loc_opt>upper:
                        loc_opt=upper
                ax[x, x].plot([loc_opt, loc_opt], [peak_val, 0], c=col, ls='--')

            else:
                dist = np.linalg.norm(abc_trial_pred[:, [x, y]] - data_pred[[x, y]], axis=1)
                accepted_ind = np.argpartition(dist, nr_of_accept_cross)[0:nr_of_accept_cross]
                accepted_para = abc_trial_thetas[accepted_ind]
                loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para[:, y]), np.std(accepted_para[:, y])),
                                                   args=([accepted_para[:, y], dmin[y], dmax[y]],), disp=False)

                Cov_Matrix[x,y] = scale_opt
                Cov_Matrix[y,x] = scale_opt

                left_trunc_norm = (dmin[y] - loc_opt) / scale_opt
                right_trunc_norm = (dmax[y] - loc_opt) / scale_opt

                l = np.linspace(dmin[y], dmax[y], 100)
                p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)
                ax[y, y].plot(l, p, c='gray', alpha=0.2, lw = 1)
                ax[x, y].scatter(accepted_para[:, y], accepted_para[:, x], color="green", s=1, alpha=0.5)
                ax[x, y].scatter(true_param[y],true_param[x], color="black", marker="*")
                ax[x, y].scatter(accepted_mean[y],accepted_mean[x], color="red", marker="x")
                ax[x, y].scatter(data_pred[y],data_pred[x], color="gray", marker="o")
                ax[x, y].plot([dmin[y], dmin[y], dmax[y], dmax[y], dmin[y]],
                              [dmin[x], dmax[x], dmax[x], dmin[x], dmin[x]], lw=lwith, c=range_color)

    plt.savefig('posterior_abc6')

    return Mean_Vector, Cov_Matrix