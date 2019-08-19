from sciope.inference import abc_inference
from sciope.models.cnn_regressor import CNNModel
from sciope.models.pen_regressor_beta import PEN_CNNModel
from sciope.models.dnn_regressor import ANNModel
from load_data_from_julia import load_data
import numpy as np
from AutoRegressive_model import simulate, prior
# from MovingAverage_model import simulate, prior
from sklearn.metrics import mean_absolute_error


# sim = simulate

# true_param = [0.2,-0.13]
# data = simulate(true_param)
# n=1000000
# train_thetas = np.array(prior(n=n))
# train_ts = np.expand_dims(np.array([simulate(p,n=100) for p in train_thetas]),2)
# validation_thetas = np.array(prior(n=10000))
# validation_ts = np.expand_dims(np.array([simulate(p,n=100) for p in validation_thetas]),2)
print("update")
true_param, data, train_thetas, train_ts, validation_thetas, validation_ts, test_thetas, test_ts,\
abc_trial_thetas, abc_trial_ts = load_data('moving_average2')

print("validation_thetas shape: ", validation_thetas.shape)
print("validation_ts shape: ", validation_ts.shape)

print("training_thetas shape: ", train_thetas.shape)
print("training_ts shape: ", train_ts.shape)

# choose neural network model
# nnm = CNNModel(input_shape=(100,1), output_shape=(2))
nnm = PEN_CNNModel(input_shape=(100,1), output_shape=(2), pen_nr=2)
# nm = ANNModel(input_shape=(100,1), output_shape=(2))


# nnm.load_model()

# nnm.train(inputs=train_ts, targets=train_thetas,validation_inputs=validation_ts,validation_targets=validation_thetas,
#           plot_training_progress=False)

nnm.load_model()
#validation_pred = np.array([nnm.predict(validation_ts[i*100:(i+1)*100]) for i in range(500)])
validation_pred = nnm.predict(validation_ts)
print("validation_pred shape: ", validation_pred.shape)
validation_pred = np.reshape(validation_pred,(-1,2))


print("mean squared error: ", np.mean((validation_thetas-validation_pred)**2))

print("Model name: ", nnm.name)


#