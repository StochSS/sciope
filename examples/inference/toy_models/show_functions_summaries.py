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

    # nnm.model.summery()