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



def abc_inference(data, true_param, abc_trial_thetas,abc_trial_ts, nnm,dmin,dmax, bins, nr_of_accept = 1000,
                  index = 0):


    data_pred = nnm.predict(data)
    data_pred = np.squeeze(data_pred)
    lwith = 1
    range_color = 'blue'
    nr_of_trial = abc_trial_thetas.shape[0]
    range_color = '#1f77b4'
    fsize = 80
    lwith = 6
    scattersize = 5


    abc_trial_pred = nnm.predict(abc_trial_ts)

    nr_of_trial = abc_trial_thetas.shape[0]

    para_names = vilar.get_parameter_names()

    # for p in para_names:
    #     print(p)

    dist = np.linalg.norm(abc_trial_pred - data_pred, axis=1)
    accepted_ind = np.argpartition(dist, nr_of_accept)[0:nr_of_accept]
    accepted_para = abc_trial_thetas[accepted_ind]

    # true_param = normalize_data(true_param,dmin,dmax)
    # plt.axis('equal')
    f, ax = plt.subplots(5,3,figsize=(30,30))# ,sharex=True,sharey=True)
    f.suptitle('Accepted/Trial = ' + str(nr_of_accept) + '/' + str(nr_of_trial),fontsize=16)
    Posterior_fit = []
    for i in range(15):
        # bins = np.linspace(dmin[i], dmax[i], bins_nr + 1)



        y = i%3
        x = i//3
        d = ax[x,y].hist(accepted_para[:,i],bins=bins[i],density=True)
        Posterior_fit.append(d[0])
        peakv = np.max(d[0])
        ax[x, y].plot([true_param[i], true_param[i]],[peakv, 0])


    plt.savefig('posterior_plots/posterior_abc' + str(index))
    plt.close()

    return np.array(Posterior_fit)