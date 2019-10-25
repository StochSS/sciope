import numpy as np


def abc_inference(data, abc_trial_thetas,abc_trial_ts, nnm, nr_of_accept = 1000):

    data_pred = nnm.predict(data)
    data_pred = np.squeeze(data_pred)

    abc_trial_pred = nnm.predict(abc_trial_ts)

    dist = np.linalg.norm(abc_trial_pred - data_pred, axis=1)
    accepted_ind = np.argpartition(dist, nr_of_accept)[0:nr_of_accept]
    accepted_para = abc_trial_thetas[accepted_ind]
    accepted_pred = abc_trial_pred[accepted_ind]

    return accepted_para, accepted_pred, data_pred

def abc_inference_marginal(data, abc_trial_thetas,abc_trial_ts, nnm, nr_of_accept = 1000):

    data_pred = nnm.predict(data)
    data_pred = np.squeeze(data_pred)

    abc_trial_pred = nnm.predict(abc_trial_ts)
    accepted_para = np.zeros((nr_of_accept,abc_trial_thetas.shape[1]))
    accepted_pred = np.zeros((nr_of_accept,abc_trial_thetas.shape[1]))
    print("abc_trial_pred shape: ", abc_trial_pred.shape, "data_pred shape: ", data_pred.shape)
    for i in range(abc_trial_thetas.shape[1]):
        dist = abs(abc_trial_pred[:,i] - data_pred[i])
        print("i: ", i, ", dist shape: ", dist.shape)
        accepted_ind = np.argpartition(dist, nr_of_accept)[0:nr_of_accept]
        accepted_para[:,i] = abc_trial_thetas[accepted_ind][:,i]
        accepted_pred[:,i] = abc_trial_pred[accepted_ind][:,i]

    return accepted_para, accepted_pred, data_pred
