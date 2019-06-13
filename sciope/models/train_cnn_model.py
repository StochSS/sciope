
import pickle
import os
import numpy as np
from sciope.models.cnn_regressor import CNNModel

# Get the data
data_path = '/home/ubuntu/sciope/sciope/utilities/datagenerator/ds_vilar_ft100_ts501_tr1_speciesall' #/ds_vilar_ft100_ts501_tr1_speciesall0.p'

theta = None

for filename in os.listdir(data_path):
    dataset = pickle.load(open(data_path + '/' + filename, "rb" ) )
    if theta is not None:
        theta = np.append(theta, dataset.x, axis=0)
        ts = np.append(ts, dataset.ts, axis=0)
    else:
        theta = dataset.x
        ts = dataset.ts

print("theta shape: ", theta.shape, ", timeseries shape: ", ts.shape)

# Remove trajectory dimension
theta = np.squeeze(theta, axis=1)
ts = np.squeeze(ts, axis=1)

print("theta shape: ", theta.shape, ", timeseries shape: ", ts.shape)

#transpose the dimension of ts to match the CNN
ts = np.transpose(ts,(0,2,1))

dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
# Normalize the parameter to 0-1
for t in range(len(dmin)):
    theta[:,t]=(theta[:,t]-dmin[t]) / (dmax[t] - dmin[t])

# Quick check if all values are between 0-1
print("theta min: ", np.min(theta), ", theta max: ", np.max(theta),
      ", theta mean: ", np.mean(theta), ", theta std: ", np.std(theta))


# Define a CNN model
input_shape = ts.shape[1:]
output_shape = theta.shape[1]
print("input_shape: ", input_shape, ", output_shape: ", output_shape)
CNN = CNNModel(input_shape=input_shape,output_shape=output_shape)

# Split data into training and validation
theta_train = theta[0:9000]
ts_train = ts[0:9000]

theta_val = theta[9000:]
ts_val = ts[9000:]



# Train the CNN model
CNN.train(inputs=ts_train, targets = theta_train,validation_inputs=ts_val,validation_targets=theta_val,
              save_as='test')
