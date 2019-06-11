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

from sciope.utilities.priors import uniform_prior
from sciope.inference import abc_inference



from sciope.utilities.summarystats import burstiness as bs
import numpy as np
import vilar
from sklearn.metrics import mean_absolute_error
from toolz import partition_all
import dask




# Load data
data = np.loadtxt("datasets/vilar_dataset_specieA_50trajs_15time.dat", delimiter=",")

# Set up the prior
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
mm_prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
bs_stat = bs.Burstiness(mean_trajectories=True, use_logger=False)

# Set up ABC

#print data
print("data: ", data)

abc_instance = abc_inference.ABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior,
                                 summaries_function=bs_stat)

print("data 2",abc_instance.data)
# Perform ABC; require 30 samples



chunk_size=100



data_chunked = partition_all(chunk_size, data)

# compute summary stats on fixed data
#stats = [bs_stat.compute(x) for x in data_chunked]
stats = [bs_stat.computem(bs_stat,x) for x in data]


print("data shape: ", data.shape)

mean = dask.delayed(np.mean)

# reducer 1 mean for each batch


bsx=bs_stat.computem(bs_stat,[1,2,3,4,5])

print("bsx compute: ", bsx.compute())


rand_mat = np.random.rand(2500,1000)
mat_chunked = partition_all(5, data)#rand_mat)
m = mean(mat_chunked,axis=0)
m = mean(m,axis=0,keepdims=True)

print("m computed mean: ", m.compute().shape)


print("stats type ",type(stats), ", len: ", len(stats), " -- > ", type(stats[0]), ": ", stats[0].compute() )
stats_mean = mean(stats)




sm=stats_mean.compute()#,axis=0)

print("stats_mean shape: ", sm.shape())
# reducer 2 mean over batches
stats_mean = mean(stats_mean, axis=0, keepdims=True).compute()

fixed_mean = np.copy(stats_mean)
del stats_mean
print(fixed_mean)


#abc_instance.infer(num_samples=200, batch_size=100)

# Results
#true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
#print('Inferred parameters: ', abc_instance.results['inferred_parameters'])
#print('Inference error in MAE: ', mean_absolute_error(true_params, abc_instance.results['inferred_parameters']))
#print('Trial count:', abc_instance.results['trial_count'])
