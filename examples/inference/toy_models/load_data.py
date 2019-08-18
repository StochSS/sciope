
import pickle
import os
import numpy as np


def load_spec(modelname="vilar_ACR_100_201", type = "train"):
    loaded_thetas = None
    loaded_ts = None
    nr=0
    while os.path.isfile("datasets/"+modelname+"/"+type+"_"+str(nr)) and os.path.isfile("datasets/"+modelname+"/"+type+"_"+str(nr)):
        thetas = pickle.load(open('datasets/' + modelname + '/' + type + '_thetas_' + str(nr) + '.p', "rb"))
        ts = pickle.load(open('datasets/' + modelname + '/' + type + '_ts_' + str(nr) + '.p', "rb"))

        if nr == 0:
            loaded_thetas = thetas
            loaded_ts = ts
        else:
            loaded_thetas = np.concatenate(loaded_thetas,thetas,axis=0)
            loaded_ts = np.concatenate(loaded_ts,ts,axis=0)
        nr+=1
    return loaded_thetas, loaded_ts


def load_data(modelname='auto_regression2'):
    true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
    data = pickle.load(open('datasets/' + modelname + '/obs_data.p', "rb" ) )

    n=1000000
    train_thetas = pickle.load(open('datasets/' + modelname + '/train_thetas.p', "rb" ) )

    train_ts = pickle.load(open('datasets/' + modelname + '/train_ts.p', "rb" ) )

    validation_thetas = pickle.load(open('datasets/' + modelname + '/validation_thetas.p', "rb" ) )
    validation_ts = pickle.load(open('datasets/' + modelname + '/validation_ts.p', "rb" ) )

    test_thetas = pickle.load(open('datasets/' + modelname + '/test_thetas.p', "rb" ) )
    test_ts = pickle.load(open('datasets/' + modelname + '/test_ts.p', "rb" ) )

    abc_trial_thetas = pickle.load(open('datasets/' + modelname + '/abc_trial_thetas.p', "rb" ) )
    abc_trial_ts = pickle.load(open('datasets/' + modelname + '/abc_trial_ts.p', "rb" ) )

    return true_param, data, train_thetas, train_ts, validation_thetas, validation_ts, test_thetas, test_ts, \
           abc_trial_thetas, abc_trial_ts

