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

from sciope.utilities.priors import uniform_prior
from sciope.inference import abc_inference
from sciope.utilities.summarystats import burstiness as bs
import numpy as np
import vilar
from sklearn.metrics import mean_absolute_error
from sciope.utilities.distancefunctions.euclidean import EuclideanDistance
from dask.distributed import Client

# Load data and check dim
data = np.loadtxt("datasets/vilar_dataset_specieA_50trajs_15time.dat", delimiter=",")
print(data.shape)

dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
mm_prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
bs_stat = bs.Burstiness(mean_trajectories=False, use_logger=False)
euc = EuclideanDistance(use_logger=False)

# Set up ABC
abc_instance = abc_inference.ABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior,
                                 summaries_function=bs_stat, distance_function=euc)

test = abc_instance.rejection_sampling(num_samples=50, chunk_size=10, batch_size=100)

# Results
true_value = np.array(['50.0', '500.0', '0.01', '50.0', '50.0', '5.0', '10.0', '0.5', '1.0', '0.2', '1.0', '1.0', '2.0'
                          , '50.0', '100.0'], dtype=float)
print('Inferred parameters: ', abc_instance.results['inferred_parameters'])
print('Inference error in MAE: ', mean_absolute_error(true_params, abc_instance.results['inferred_parameters']))
print('Trial count:', abc_instance.results['trial_count'])

np.abs(abc_instance.results["inferred_parameters"] - true_params)
