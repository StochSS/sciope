# Copyright 2017 Prashant Singh, Fredrik Wrede and Andreas Hellander
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
Michaelis-Menten Model: Approximate Bayesian Computation Test Run
"""

# Imports
from mio.utilities.priors import uniform_prior
from mio.utilities.summarystats import burstiness as bs
from mio.inference import abc_inference
import numpy as np
import mm
from sklearn.metrics import mean_absolute_error

# Load data
data = np.loadtxt("mm_dataset1000_t500.dat", delimiter=",")

# Set up the prior and summary statistic
dmin = [0.0001, 0.2, 0.05]
dmax = [0.05, 0.6, 0.3]
mm_prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
bs_stat = bs.Burstiness(mean_trajectories=False)

# Set up ABC
abc_instance = abc_inference.ABC(data, mm.simulate, epsilon=0.1, prior_function=mm_prior, summaries_function=bs_stat)

# Perform ABC; require 30 samples
abc_instance.infer(30)

# Results
true_params = [[0.0017, 0.5, 0.1]]
print('Inferred parameters: ', abc_instance.results['inferred_parameters'])
print('Inference error in MAE: ', mean_absolute_error(true_params, abc_instance.results['inferred_parameters']))
print('Trial count:', abc_instance.results['trial_count'])

