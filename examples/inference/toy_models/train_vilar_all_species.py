from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
# from sciope.models.cnn_regressor_normal import CNNModel

from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from load_data_from_julia import load_data
import numpy as np
from AutoRegressive_model import simulate, prior
# from MovingAverage_model import simulate, prior
from sklearn.metrics import mean_absolute_error
import pickle
import time
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
import vilar
from vilar import Vilar_model





num_timestamps=401
endtime=200
modelname = "vilar_ACR_" + str(endtime) + "_" + str(num_timestamps) + '_all_species'
# parameter range
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]

#5,6,7,8
#6=C, 7=A, 8=R
species = [0,2]




#Load data
train_thetas, train_ts = load_spec(modelname=modelname, type = "train", species=species)
print("load train data done!")
validation_thetas = pickle.load(open('datasets/' + modelname + '/validation_thetas.p', "rb" ) )
validation_ts = pickle.load(open('datasets/' + modelname + '/validation_ts.p', "rb" ) )[:,:,species]

training_size = 300000

train_thetas = train_thetas[0:training_size]
train_ts = train_ts[0:training_size]
#Normalize parameter values
train_thetas = normalize_data(train_thetas,dmin,dmax)
validation_thetas = normalize_data(validation_thetas,dmin,dmax)
step=2
end_step = 401
print("end_step: ", end_step)
train_ts = train_ts[:,:end_step:step,:]
print("ts shape: ", train_ts.shape)
validation_ts = validation_ts[:,:end_step:step,:]
clay=[32,48,64,96]
ts_len = train_ts.shape[1]
print("species: ", species)
# choose neural network model
nnm = CNNModel(input_shape=(ts_len,train_ts.shape[2]), output_shape=(15), con_len=3, con_layers=[64], dense_layers=[100,100,100])
# nnm = ANNModel(input_shape=(ts_len, train_ts.shape[2]), output_shape=(15), layers=[100,100,100])
# nnm = PEN_CNNModel(input_shape=(train_ts.shape[1],train_ts.shape[2]), output_shape=(15), pen_nr=3, con_layers=clay, dense_layers=[100,100,100])
print("save results as: ", 'results/training_results_' + modelname + '_' + nnm.name + '_training_size_' + str(training_size) +
                 '_step_' + str(step) + '_endstep_' + str(end_step) + '_species_' + str(species) + '.p', "wb")
print("Model name: ", nnm.name)
para_names = vilar.get_parameter_names()

print("parameter names: ")
for p in para_names:
    print(p)

# nnm.load_model()
start_time = time.time()
history1 = nnm.train(inputs=train_ts, targets=train_thetas,validation_inputs=validation_ts,validation_targets=validation_thetas,
          batch_size=32, epochs=40*10, val_freq=1, early_stopping_patience=5, plot_training_progress=False)


pickle.dump(history1.history, open('history1.p', "wb"))

history2 = nnm.train(inputs=train_ts, targets=train_thetas,validation_inputs=validation_ts,validation_targets=validation_thetas,
          batch_size=4096, epochs=5*10, val_freq=1, early_stopping_patience=5, plot_training_progress=False)


end_time = time.time()
training_time = end_time - start_time
validation_pred = nnm.predict(validation_ts)
validation_pred = np.reshape(validation_pred,(-1,15))
train_pred = nnm.predict(train_ts)
train_pred = np.reshape(train_pred,(-1,15))
print("training time: ", training_time)
print("validation mean square error: ", np.mean((validation_thetas-validation_pred)**2))
print("validation mean absolute error: ", np.mean(abs(validation_thetas-validation_pred)))
print("training mean absolute error: ", np.mean(abs(train_thetas-train_pred)))










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




print("Model name: ", nnm.name)
print("mean square error: ", test_mse)
print("mean absolute error: ", test_mae)
# print("test_ae: ")
# for ta in test_ae:
#     print(ta)
# print("mean test_ae: ", np.mean(test_ae))

test_results = {"model name": nnm.name, "training_time": training_time, "mse": test_mse, "mae": test_mae, "ae": test_ae, "rel_test_ae": test_ae_norm}
pickle.dump(test_results,
            open('results/training_results_' + modelname + '_' + nnm.name + '_training_size_' + str(training_size) +
                 '_step_' + str(step) + '_endstep_' + str(end_step) + '_species_' + str(species) + '.p', "wb"))

