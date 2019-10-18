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
import os



def abc_inference(data, true_param, abc_trial_thetas,abc_trial_ts, nnm,dmin,dmax, nr_of_accept = 100,
                  nr_of_accept_cross=100, index = 0):


    data_pred = nnm.predict(data)
    data_pred = np.squeeze(data_pred)
    lwith = 1
    range_color = 'blue'
    nr_of_trial = abc_trial_thetas.shape[0]
    nr_of_accept = 3000
    nr_of_accept_cross = 1000
    range_color = '#1f77b4'
    fsize = 80
    lwith = 6
    scattersize = 5


    abc_trial_pred = nnm.predict(abc_trial_ts)

    nr_of_trial = abc_trial_thetas.shape[0]

    para_names = vilar.get_parameter_names()

    for p in para_names:
        print(p)



    # true_param = normalize_data(true_param,dmin,dmax)
    # plt.axis('equal')
    f, ax = plt.subplots(15,15,figsize=(30,30))# ,sharex=True,sharey=True)
    f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial),fontsize=16)
    bins_nr = 10
    bins = np.linspace(0,1,bins_nr+1)




    bg=0

    def nnlf(params, datapack):
        loc, scale, = params
        data, lower, upper = datapack
        left_trunc_norm = (lower - loc) / scale
        right_trunc_norm = (upper - loc) / scale
        theta = (left_trunc_norm, right_trunc_norm, loc, scale)
        value = stats.truncnorm.nnlf(theta, data)
        return value

    Cov_Matrix = np.zeros((15,15))
    Mean_Vector = np.zeros(15)
    Posterior_fit = np.zeros(15,100)
    for x in range(15):
        # ax[0, x].set_title(para_names[x])
        for y in range(x,15):

            if x == y:
                dist = abs(abc_trial_pred[:, x] - data_pred[x])
                accepted_ind = np.argpartition(dist, nr_of_accept)[0:nr_of_accept]
                accepted_para = abc_trial_thetas[accepted_ind]

                loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para[:, x]), np.std(accepted_para[:, x])),
                                                   args=([accepted_para[:, x], dmin[x], dmax[x]],), disp=False)

                left_trunc_norm = (dmin[x] - loc_opt) / scale_opt
                right_trunc_norm = (dmax[x] - loc_opt) / scale_opt
                print("x: ", x, ", params full: ", left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)

                l = np.linspace(dmin[x], dmax[x], 100)
                p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)
                Posterior_fit[x] = p
                ax[x, y].tick_params(labelleft=False)
                # ax[y, x].plot(l, p, c='green', lw=lwith)
                # ax[y, x].plot(l, p_full, c='red', lw=lwith, ls='--')
                # ax[x, y].set_xlabel(para_names[x], fontsize=fsize, y=5)
                # ax[x, y].yaxis.set_label_position("left")
                # ax[x, y].set_ylabel('density', fontsize=fsize, rotation=90)

                ret = ax[x, y].hist(accepted_para[:, x], density=True, bins=20, color='green', alpha=1)
                # ax[x, y].hist(accepted_para_full[:, x], density=True, bins=20, color='red', alpha=0.3)

                peak_val = np.maximum(np.max(ret[0]), np.max(p))
                ax[x, y].plot([true_param[x], true_param[x]], [0, peak_val], c='black')

                ax[x, y].plot([dmax[x], dmax[x]], [0, peak_val], c=range_color, lw=lwith)
                ax[x, y].plot([dmin[x], dmin[x]], [0, peak_val], c=range_color, lw=lwith)
                ax[x, y].plot([true_param[x], true_param[x]], [0, peak_val], c='black', lw=lwith, ls='--')



                loc_opt, scale_opt = optimize.fmin(nnlf, (np.mean(accepted_para[:, x]), np.std(accepted_para[:, x])),
                                                   args=([accepted_para[:, x], dmin[x], dmax[x]],), disp=False)
                Cov_Matrix[x,y] = scale_opt
                left_trunc_norm = (dmin[x] - loc_opt) / scale_opt
                right_trunc_norm = (dmax[x] - loc_opt) / scale_opt

                l = np.linspace(dmin[x],dmax[x], 100)
                p = stats.truncnorm.pdf(l, left_trunc_norm, right_trunc_norm, loc_opt, scale_opt)

                ax[x, x].plot(l, p)
                col ='red'
                if loc_opt<dmin[x] or loc_opt>dmax[x]:
                    col = 'orange'
                    if loc_opt<dmin[x]:
                        loc_opt=dmin[x]
                    if loc_opt>dmax[x]:
                        loc_opt=dmax[x]
                ax[x, x].plot([loc_opt, loc_opt], [peak_val, 0], c=col, ls='--')
                Mean_Vector[y] = loc_opt


            else:
                dist = np.linalg.norm(abc_trial_pred[:, [x, y]] - data_pred[[x, y]], axis=1)
                accepted_ind = np.argpartition(dist, nr_of_accept_cross)[0:nr_of_accept_cross]
                accepted_para = abc_trial_thetas[accepted_ind]
                # print("accepted para shape: ", accepted_para.shape)
                # print("(", x, ",", y, ") mean x: " + "{0:.2f}".format(np.mean(accepted_para[:, x])) + ", mean y: " + "{0:.2f}".format(np.mean(accepted_para[:, y])))

                # ax[x, y].scatter(abc_trial_thetas[:, y], abc_trial_thetas[:, x], color="yellow", s=2)
                # ax[x, y].scatter(accepted_para_full[:, y], accepted_para_full[:, x], color="red", s=scattersize,
                #                  alpha=0.1)

                ax[x, y].scatter(accepted_para[:, y], accepted_para[:, x], color="green", s=scattersize, alpha=1)

                ax[x, y].scatter(true_param[y], true_param[x], color="black", marker="x", s=100)
                # ax[x, y].scatter(accepted_mean[y], accepted_mean[x], color="red", marker="x")
                # ax[x, y].scatter(data_pred[y], data_pred[x], color="gray", marker="o")
                ax[x, y].plot([dmin[y], dmin[y], dmax[y], dmax[y], dmin[y]],
                              [dmin[x], dmax[x], dmax[x], dmin[x], dmin[x]], lw=lwith, c=range_color)


    print("index before saving: ", index)
    if not os.path.exists('posterior_plots'):
        os.mkdir('posterior_plots')
    plt.savefig('posterior_plots/posterior_abc' + str(index))
    print("saved")
    return Mean_Vector, Cov_Matrix, Posterior_fit