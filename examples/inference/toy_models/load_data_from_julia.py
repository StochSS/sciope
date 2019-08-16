
import pickle
import pandas as pd
import numpy as np


def load_data(modelname='auto_regression2'):
    true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
    data = pickle.load(open('datasets/' + modelname + '/obs_data.p', "rb" ) )

    train_thetas = np.array( pd.read_csv(r"C:\Users\ma10s\Documents\PENs-and-ABC\data\MA2 noisy data\y_training.csv"))
    train_ts = np.expand_dims(np.array(pd.read_csv(r"C:\Users\ma10s\Documents\PENs-and-ABC\data\MA2 noisy data\X_training.csv")),2)

    validation_thetas = np.array(pd.read_csv(r"C:\Users\ma10s\Documents\PENs-and-ABC\data\MA2 noisy data\y_val.csv"))
    validation_ts = np.expand_dims(np.array(pd.read_csv(r"C:\Users\ma10s\Documents\PENs-and-ABC\data\MA2 noisy data\X_val.csv")),2)

    test_thetas = np.array(pd.read_csv(r"C:\Users\ma10s\Documents\PENs-and-ABC\data\MA2 noisy data\y_test.csv"))
    test_ts= np.expand_dims(np.array(pd.read_csv(r"C:\Users\ma10s\Documents\PENs-and-ABC\data\MA2 noisy data\X_test.csv")),2)

    abc_trial_thetas = np.array(pd.read_csv(r"C:\Users\ma10s\Documents\PENs-and-ABC\data\MA2 noisy data\abc_data_parameters.csv"))
    abc_trial_ts = np.expand_dims(np.array(pd.read_csv(r"C:\Users\ma10s\Documents\PENs-and-ABC\data\MA2 noisy data\abc_data_data.csv")),2)

    return true_param, data, train_thetas, train_ts, validation_thetas, validation_ts, test_thetas, test_ts, \
           abc_trial_thetas, abc_trial_ts

