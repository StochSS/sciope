from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
# from sciope.models.cnn_regressor_normal import CNNModel

from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from load_data_from_julia import load_data
import numpy as np
import os
import pickle
import time
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
import vilar
from vilar import Vilar_model



def train_routine(modelname, dmin, dmax, species = [0,2], training_size = 300000, step=1, end_step=401,clay=[32,48,64,96],dlay=[100,100,100],
                  model='CNN',load_model=False, verbose=2, pooling_len=3, dataname ="",
                  res_folder="Random_folder"+str(np.random.rand(4))):



    #Load data
    train_thetas, train_ts = load_spec(modelname=modelname, type = "train", species=species)
    validation_thetas = pickle.load(open('datasets/' + modelname + '/validation_thetas.p', "rb" ) )
    validation_ts = pickle.load(open('datasets/' + modelname + '/validation_ts.p', "rb" ) )[:,:,species]
    #cut training data (if training_size<300 000)
    train_thetas = train_thetas[0:training_size]
    train_ts = train_ts[0:training_size]
    #Normalize parameter values
    train_thetas = normalize_data(train_thetas,dmin,dmax)
    validation_thetas = normalize_data(validation_thetas,dmin,dmax)
    #Resample data
    train_ts = train_ts[:,:end_step:step,:]
    print("validation_ts shape: ", validation_ts.shape)
    validation_ts = validation_ts[:,:end_step:step,:]

    ts_len = train_ts.shape[1]
    # choose neural network model
    if model == 'CNN':
        nnm = CNNModel(input_shape=(ts_len,train_ts.shape[2]), output_shape=15, con_len=3, con_layers=clay,
                       dense_layers=dlay, pooling_len=pooling_len, dataname=dataname)
    elif model == 'DNN':
        nnm = ANNModel(input_shape=(ts_len, train_ts.shape[2]), output_shape=(15), layers=dlay)
    elif model == 'PEN':
        nnm = PEN_CNNModel(input_shape=(train_ts.shape[1],train_ts.shape[2]), output_shape=(15), pen_nr=10, con_layers=clay, dense_layers=dlay)
    else:
        print("invalid model name!")

    print("save results as: ", 'results/training_results_' + modelname + '_' + nnm.name + '_training_size_' + str(training_size) +
                     '_step_' + str(step) + '_endstep_' + str(end_step) + '_species_' + str(species) + '.p', "wb")
    print("Model name: ", nnm.name)
    para_names = vilar.get_parameter_names()

    if load_model:
        start_time = time.time()
        nnm.load_model()
    else:
        start_time = time.time()
        history1 = nnm.train(inputs=train_ts, targets=train_thetas,validation_inputs=validation_ts,validation_targets=validation_thetas,
                  batch_size=32, epochs=40*100, val_freq=1, early_stopping_patience=5, plot_training_progress=False, verbose=verbose)


        pickle.dump(history1.history, open('history1.p', "wb"))

        history2 = nnm.train(inputs=train_ts, targets=train_thetas,validation_inputs=validation_ts,validation_targets=validation_thetas,
                  batch_size=4096, epochs=40*100, val_freq=1, early_stopping_patience=5, plot_training_progress=False, verbose=verbose)


        end_time = time.time()
        training_time = end_time - start_time

    #predict validation and train data
    validation_pred = nnm.predict(validation_ts)
    validation_pred = np.reshape(validation_pred,(-1,15))
    train_pred = nnm.predict(train_ts)
    train_pred = np.reshape(train_pred,(-1,15))
    print("training time: ", training_time)
    print("validation mean square error: ", np.mean((validation_thetas-validation_pred)**2))
    print("validation mean absolute error: ", np.mean(abs(validation_thetas-validation_pred)))
    print("training mean absolute error: ", np.mean(abs(train_thetas-train_pred)))

    # Load and predict test data
    test_thetas = pickle.load(open('datasets/' + modelname + '/test_thetas.p', "rb" ) )
    test_ts = pickle.load(open('datasets/' + modelname + '/test_ts.p', "rb" ) )
    test_ts = test_ts[:,:end_step:step,species]
    test_thetas_n = normalize_data(test_thetas,dmin,dmax)
    test_pred = nnm.predict(test_ts)
    test_pred = np.reshape(test_pred,(-1,15))
    test_pred_d = denormalize_data(test_pred,dmin,dmax)
    test_mse = np.mean((test_thetas-test_pred)**2)
    test_mae = np.mean(abs(test_thetas-test_pred_d))
    test_ae = np.mean(abs(test_thetas-test_pred_d),axis=0)
    test_ae_norm = np.mean(abs(test_thetas_n-test_pred),axis=0)

    print("test data mean absolute error: ", test_mae)


    test_results = {"model name": nnm.name, "training_time": training_time, "mse": test_mse, "mae": test_mae, "ae": test_ae, "rel_test_ae": test_ae_norm}

    if not os.path.exists('results/' + res_folder):
        os.mkdir('results/' + res_folder)

    pickle.dump(test_results,
                open('results/' + res_folder +'/training_results_' + modelname + '_' + nnm.name + '_training_size_' +
                     str(training_size) + '_step_' + str(step) + '_endstep_' + str(end_step) + '_species_' +
                     str(species) + '.p', "wb"))

