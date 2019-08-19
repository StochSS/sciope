from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from load_data_from_julia import load_data
import numpy as np
from AutoRegressive_model import simulate, prior
# from MovingAverage_model import simulate, prior
from sklearn.metrics import mean_absolute_error
import pickle
from normalize_data import normalize_data
from load_data import load_spec

# sim = simulate

# true_param = [0.2,-0.13]
# data = simulate(true_param)
# n=1000000
# train_thetas = np.array(prior(n=n))
# train_ts = np.expand_dims(np.array([simulate(p,n=100) for p in train_thetas]),2)
# validation_thetas = np.array(prior(n=10000))
# validation_ts = np.expand_dims(np.array([simulate(p,n=100) for p in validation_thetas]),2)
print("update")
modelname = "vilar_ACR_200_401"
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
# train_thetas = pickle.load(open('datasets/' + modelname + '/train_thetas.p', "rb" ) )
# train_ts = pickle.load(open('datasets/' + modelname + '/train_ts.p', "rb" ) )

train_thetas, train_ts = load_spec(modelname="vilar_ACR_100_201", type = "train")

validation_thetas = pickle.load(open('datasets/' + modelname + '/validation_thetas.p', "rb" ) )
validation_ts = pickle.load(open('datasets/' + modelname + '/validation_ts.p', "rb" ) )

# print("train thetas min: ", np.min(train_thetas,0), ", max: ", np.max(train_thetas,0))

train_thetas = normalize_data(train_thetas,dmin,dmax)
validation_thetas = normalize_data(validation_thetas,dmin,dmax)

# print("train thetas min: ", np.min(train_thetas,0), ", max: ", np.max(train_thetas,0))


print("validation_thetas shape: ", validation_thetas.shape)
print("validation_ts shape: ", validation_ts.shape)

print("training_thetas shape: ", train_thetas.shape)
print("training_ts shape: ", train_ts.shape)

# choose neural network model
# nnm = CNNModel(input_shape=(201,3), output_shape=(15))
nnm = PEN_CNNModel(input_shape=(401,3), output_shape=(15), pen_nr=2)
# nm = ANNModel(input_shape=(100,1), output_shape=(2))

# nnm.load_model()

nnm.train(inputs=train_ts, targets=train_thetas,validation_inputs=validation_ts,validation_targets=validation_thetas,
          plot_training_progress=False)

# nnm.load_model()
#validation_pred = np.array([nnm.predict(validation_ts[i*100:(i+1)*100]) for i in range(500)])
validation_pred = nnm.predict(validation_ts)
print("validation_pred shape: ", validation_pred.shape)
validation_pred = np.reshape(validation_pred,(-1,15))


print("mean squared error: ", np.mean((validation_thetas-validation_pred)**2))

print("Model name: ", nnm.name)


