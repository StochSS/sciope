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
Example: use the modules from mio to perform surrogate-based parameter inference from existing data
"""
# Imports
import mio
import numpy as np
import mm_sim as m2s
from mio.models import *
from mio.designs import *
from sklearn.metrics import mean_squared_error

# Domain
dmin = [0.0001, 0.2, 0.05]
dmax = [0.05, 0.6, 0.3]


# Call to simulator
# The simulator computes the distance between a simulated value and a random value from an existing dataset
def obj(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(0, n - 1):
        y[i] = m2s.compute(x[i, :])
    return y


# Set up MIO components
lhd = latin_hypercube_sampling.LatinHypercube(dmin, dmax)
ml_model = svm_regressor.SVRModel()
num_points = 200
problem = obj

# Instantiate
mio_instance = mio.MIO(problem=obj, initial_design=lhd, initial_design_size=num_points, surrogate=ml_model)

# Train a surrogate
mio_instance.model()


# Use the surrogate as an objective
# This now corresponds to minimizing the distance between simulated values and the given fixed dataset
# The optima corresponds to the inferred parameters
def obj_surrogate(xt):
    n = xt.size
    x = xt.reshape(1, n)
    return mio_instance.surrogate.predict(x)


# Optimize the surrogate
mio_optimizer = mio.MIO(problem=obj_surrogate, initial_design=lhd, surrogate=ml_model)
mio_optimizer.optimize()

# Sanity check to verify that the model is accurate enough
rnds = random_sampling.RandomSampling(dmin, dmax)
xtest = rnds.generate(100)
ytest = obj(xtest)
ypredicted = mio_instance.surrogate.predict(xtest)
mse_svm = mean_squared_error(ytest, ypredicted)
print('Mean Squared Error on 100 random test points = {0}'.format(mse_svm))
