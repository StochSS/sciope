
import pickle
import numpy as np
from sciope.models.cnn_regressor import CNNModel

# Get the data
data_path = '/home/ubuntu/sciope/sciope/utilities/datagenerator/ds_vilar_ft100_ts501_tr1_speciesall/ds_vilar_ft100_ts501_tr1_speciesall0.p'
dataset = pickle.load(open( data_path, "rb" ) )
theta = dataset.x
ts = dataset.ts

print("theta shape: ", theta.shape, ", timeseries shape: ", ts.shape)

# Remove trajectory dimension
theta = np.squeeze(theta, axis=1)
ts = np.squeeze(ts, axis=1)

print("theta shape: ", theta.shape, ", timeseries shape: ", ts.shape)

# Define a CNN model
input_shape = ts.shape[1:]
output_shape = theta.shape[1]
CNN = CNNModel(input_shape=input_shape,output_shape=output_shape)

# Split data into training and validation
theta_train = theta[0:9000]
ts_train = ts[0:9000]

theta_val = theta[9000:]
ts_val = ts[9000:]



# Train the CNN model
CNN.train(inputs=ts_train, targets = theta_train,validation_inputs=ts_val,validation_targets=theta_val,
              save_as='test')
