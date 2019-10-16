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





modelname = "vilar_ACR_200_401"
# parameter range
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]

#Load data
train_thetas, train_ts = load_spec(modelname=modelname, type = "train")
validation_thetas = pickle.load(open('datasets/' + modelname + '/validation_thetas.p', "rb" ) )
validation_ts = pickle.load(open('datasets/' + modelname + '/validation_ts.p', "rb" ) )

training_size = 300000

train_thetas = train_thetas[0:training_size]
train_ts = train_ts[0:training_size]
#Normalize parameter values
train_thetas = normalize_data(train_thetas,dmin,dmax)
validation_thetas = normalize_data(validation_thetas,dmin,dmax)
step=1
end_step = 401
species = [0]
print("end_step: ", end_step)
train_ts = train_ts[:,:end_step:step,species]
print("ts shape: ", train_ts.shape)
validation_ts = validation_ts[:,:end_step:step,species]
clay=[32,48,64,96]

ts_len = train_ts.shape[1]
# choose neural network model
print("big neural net.")
nnm = CNNModel(input_shape=(ts_len,train_ts.shape[2]), output_shape=15, con_len=3, con_layers=[64], dense_layers=[200,200,200],dataname='vilar_prior1')
# nnm = PEN_CNNModel(input_shape=(ts_len,train_ts.shape[2]), output_shape=(15), pen_nr=10, con_layers=clay, dense_layers=[100,100,100])
# nnm = ANNModel(input_shape=(ts_len, train_ts.shape[2]), output_shape=(15), layers=[200,200,00])
print("Model name: ", nnm.name)
verb = 2
print("verbose: ", verb)
print("species: ", species)
# nnm.load_model('saved_models/None_DNNModel')
print("batch size 128")

start_time = time.time()
history1 = nnm.train(inputs=train_ts, targets=train_thetas,validation_inputs=validation_ts,validation_targets=validation_thetas,
          batch_size=128, epochs=40*10, learning_rate=0.001, val_freq=1, early_stopping_patience=5, plot_training_progress=False, verbose=verb)


print("history: ", history1.history)
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

validation_mae = np.mean(abs(validation_thetas-validation_pred), axis=0)

para_names = vilar.get_parameter_names()

validation_thetas = denormalize_data(validation_thetas, dmin, dmax)
validation_pred = denormalize_data(validation_pred, dmin, dmax)

mean_dev = np.mean(abs(validation_thetas-validation_pred), axis=0)



i=0
for dev, rdev, n in zip(mean_dev,validation_mae ,para_names):
    print(n, " mean deviation: ", "{0:.4f}".format(dev), " real mean deviation: ", "{0:.4f}".format(rdev), ", range: ", dmin[i], " - ", dmax[i])
    i+=1


true_param = pickle.load(open('datasets/' + modelname + '/true_param.p', "rb" ) )
true_param = np.squeeze(np.array(true_param))
print("true_param shape: ", true_param.shape)
num_timestamps=401
endtime=200

# true_param = np.ones((15))*0.8


Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)


simulate = Vilar_.simulate
print("before simulation")
data = np.array([np.squeeze(simulate(true_param)) for i in range(100)])
data = data[:,:end_step:step,species]
data_pred = nnm.predict(data)
data_pred = np.squeeze(data_pred)
data_pred_denorm = denormalize_data(data_pred,dmin,dmax)

data_pred_meandev = np.mean( abs(data_pred_denorm- true_param), axis=0)

true_param = np.squeeze(np.array(true_param))
# print("true_param shape: ", true_param.shape)
#
# print("abs(data_pred_denorm - true_param) = ", abs(data_pred_denorm - true_param) )

rel_e1 = np.mean(abs(data_pred_denorm - true_param),axis=0) / np.mean(abs( (np.array(dmin)+np.array(dmax))/2 - true_param),axis=0)

print("rel_e1 shape: ", rel_e1.shape)

i=0
for dev, re, n in zip(data_pred_meandev,rel_e1,para_names):
    print(n, ", true: ", true_param[i], ", predicted: ", "{0:.4f}".format(data_pred_denorm[0,i]), ", mean deviation: ",
          "{0:.4f}".format(dev), ", rel dev: ", "{0:.4f}".format(re), ", range: ", dmin[i], " - ", dmax[i])
    i+=1


# nnm.load_model()
# validation_pred = np.array([nnm.predict(validation_ts[i*100:(i+1)*100]) for i in range(500)])


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

rel_e = np.mean(abs(test_thetas-test_pred),axis=0) / np.mean(abs(1/2 - test_thetas),axis=0)
print("rel_e shape: ", rel_e.shape)


i=0
for dev, re, n in zip(test_ae, test_ae_norm, para_names):
    print(n, " mean deviation: ", "{0:.4f}".format(dev), ", rel dev: ", "{0:.4f}".format(re), ", range: ", dmin[i], " - ", dmax[i])
    i+=1


print("Model name: ", nnm.name)
print("mean square error: ", test_mse)
print("mean absolute error: ", test_mae)

test_results = {"model name": nnm.name, "training_time": training_time, "mse": test_mse, "mae": test_mae, "ae": test_ae, "rel_e": rel_e, "rel_test_ae": test_ae_norm}
pickle.dump(test_results,
            open('results/training_results_' + modelname + '_' + nnm.name + '_training_size_' + str(training_size) +
                 '_step_' + str(step) + '_endstep_' + str(end_step) + '_species_' + str(species) + '.p', "wb"))

