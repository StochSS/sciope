
# Set up paths
import sys
sys.path.append("../../")
sys.path.append("../../models")
sys.path.append("../../initialDesigns")


# Imports
import mio
import numpy as np
import mmSim as m2s
import svmRegressor as svr
import latinHypercubeSampling as lhs

# Domain
min = [0.1, 80, 5, 5]
max = [3, 135, 15, 15]

# Call to simulator
def obj(X):
	n = len(X)
	Y = np.zeros(n)
	for i in range(0,n-1):
		Y[i] = m2s.compute(X[i,:])
	return Y

# Set up MIO components
lhd = lhs.LatinHypercube(min,max)
mlModel = svr.SVRModel()
numPoints = 200
problem = obj

# Instantiate
mioInstance = mio.MIO(problem=obj, initialDesign=lhd, initialDesignSize=numPoints, surrogate=mlModel)

# Train a surrogate
mioInstance.model()


# Use the surrogate as an objective
def objSurrogate(X):
	n = X.size
	x = X.reshape(1,n)
	return mioInstance.surrogate.predict(x)


# Optimize the surrogate
mioOptimizer = mio.MIO(problem=objSurrogate, initialDesign=lhd, surrogate=mlModel)
mioOptimizer.optimize()