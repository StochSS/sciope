# Copyright 2017 Prashant Singh, Andreas Hellander and Fredrik Wrede
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
# Set up paths
import sys
sys.path.append("../../mio")
sys.path.append("../../mio/models")
sys.path.append("../../mio/initialDesigns")


# Imports
import mio
import numpy as np
import mmSim as m2s
from models import *
from initialDesigns import *
from sklearn.metrics import mean_squared_error

# Domain
min = [0.1, 80, 5, 5]
max = [3, 135, 15, 15]

# Call to simulator
# The simulator computes the distance between a simulated value and a random value from an existing dataset
def obj(X):
	n = len(X)
	Y = np.zeros(n)
	for i in range(0,n-1):
		Y[i] = m2s.compute(X[i,:])
	return Y

# Set up MIO components
lhd = latinHypercubeSampling.LatinHypercube(min,max)
mlModel = svmRegressor.SVRModel()
numPoints = 200
problem = obj

# Instantiate
mioInstance = mio.MIO(problem=obj, initialDesign=lhd, initialDesignSize=numPoints, surrogate=mlModel)

# Train a surrogate
mioInstance.model()


# Use the surrogate as an objective
# This now corresponds to minimizing the distance between simulated values and the given fixed dataset
# The optima corresponds to the inferred parameters
def objSurrogate(X):
	n = X.size
	x = X.reshape(1,n)
	return mioInstance.surrogate.predict(x)


# Optimize the surrogate
mioOptimizer = mio.MIO(problem=objSurrogate, initialDesign=lhd, surrogate=mlModel)
mioOptimizer.optimize()

# Sanity check to verify that the model is accurate enough
rnds = randomSampling.RandomSampling(min, max)
XTest = rnds.generate(100)
YTest = obj(XTest)
YPredicted = mioInstance.surrogate.predict(XTest)
mseSVM = mean_squared_error(YTest, YPredicted)
print 'Mean Squared Error on 100 random test points = {0}'.format(mseSVM)