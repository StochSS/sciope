import numpy as np


def abc_inference(data, abc_trial_thetas,abc_trial_ts, nnm, nr_of_accept = 1000):

    data_pred = nnm.predict(data)
    data_pred = np.squeeze(data_pred)

    abc_trial_pred = nnm.predict(abc_trial_ts)

    dist = np.linalg.norm(abc_trial_pred - data_pred, axis=1)
    accepted_ind = np.argpartition(dist, nr_of_accept)[0:nr_of_accept]
    accepted_para = abc_trial_thetas[accepted_ind]
    accepted_pred = abc_trial_pred[accepted_ind]

    return accepted_para, accepted_pred, data, data_pred
