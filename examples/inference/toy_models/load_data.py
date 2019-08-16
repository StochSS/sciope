
import pickle



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

