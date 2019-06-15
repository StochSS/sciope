# Copyright 2019 Prashant Singh, Fredrik Wrede and Andreas Hellander
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vilar Model: Approximate Bayesian Computation Test Run
"""

# Imports
from sciope.utilities.datagenerator.vilar_class import Vilar

from sciope.utilities.priors import uniform_prior
from sciope.inference import abc_inference_pre
from sciope.utilities.summarystats import burstiness as bs
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import vilar
from sklearn.metrics import mean_absolute_error

# Load data
data = np.loadtxt("datasets/vilar_dataset_specieA_50trajs_15time.dat", delimiter=",")

# Set up the prior
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
mm_prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
bs_stat = bs.Burstiness(mean_trajectories=True, use_logger=False)

true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
stoch_model = Vilar(species='all')
sim = stoch_model.simulate
data = np.array([sim(np.array(true_params))[0,7,:]])

#test bs_stat
print("data shape: ", data.shape)
print("bs_stat of data: ", bs_stat.compute(data).compute())

# Get the data
data_path = '/home/ubuntu/sciope/sciope/utilities/datagenerator/ds_vilar_ft100_ts501_tr1_speciesall' #/ds_vilar_ft100_ts501_tr1_speciesall0.p'

theta = None

file_nr=0
for filename in os.listdir(data_path):
    if file_nr<1:
        file_nr+=1
        dataset = pickle.load(open(data_path + '/' + filename, "rb" ) )
        if theta is not None:
            theta = np.append(theta, dataset.x, axis=0)
            ts = np.append(ts, dataset.ts[:,:,7,:], axis=0)
        else:
            theta = dataset.x
            ts = dataset.ts[:,:,7,:]

print("theta shape: ", theta.shape, ", timeseries shape: ", ts.shape)

# Remove trajectory dimension
theta = np.squeeze(theta, axis=1)
#ts = np.squeeze(ts, axis=2)

print("theta shape: ", theta.shape, ", timeseries shape: ", ts.shape)

# Transpose the dimension of ts to match the CNN
#ts = np.transpose(ts, (0,2,1))


#ts = ts[:,:,7]


# Set up ABC
abc_instance = abc_inference_pre.ABC(data,param=theta,time_series=ts, epsilon=0.1,
                                 summaries_function=bs_stat)


print("before dddinfer")
# Perform ABC; require 200 samples
abc_instance.infer(num_samples=200)

# Results
true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
#print('Inferred parameters: ', abc_instance.results['inferred_parameters'])
#print('Inference error in MAE: ', mean_absolute_error(true_params, abc_instance.results['inferred_parameters']))
#print('Trial count:', abc_instance.results['trial_count'])



f, axes = plt.subplots(15, 1)
f.set_figheight(10*5)
f.set_figwidth(10)
fontsize = 12
bins = np.linspace(0,1,11)
for i in range(15):
    axes[i].hist(abc_instance.results['accepted_samples'][:, i])

plt.savefig('histogram')