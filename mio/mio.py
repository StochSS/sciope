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
Surrogate Modeling and Optimization Class
Forms the backbone of the following tasks:
* training accurate surrogate models
* performing model-based optimization
* performing parameter inference, hotspot detection and inverse modeling
"""

# Imports
from scipy.optimize import basinhopping
from types import *
import numpy as np
import sys

# Set up paths
sys.path.append('initialDesigns')
sys.path.append('sampling')
sys.path.append('models')

# Import mio modules
from initialDesigns import *
from models import *
from sampling import *

# Class definition
class MIO(object):
	"""
	Class for (M)odeling, (I)nference and (O)ptimization. 
	
	Properties/variables:
	* name
	* dataset							(object of dataset)
	* problem							(a Python function to be minimized)
	* initial design					(object of initialDesignBase)
	* initial design size				(integer indicating the desired size of the initial design)
	* batchSize							(integer indicating the batch size for sampling per iteration)
	* evaluationBudget					(integer indicating the objective function evaluation budget)
	* surrogate							(object of modelBase)
	* samplingAlgorithm					(object of samplingBase)
	
	Methods:
	* model								(trains a surrogate model with specified settings)
	* optimize							(optimize a given surrogate model or simulator)
	* infer								(infer the parameters of a simulator, given some data)
	
	"""
	
	def __init__(self, name='default', dataset=None, problem=None, initialDesign=None, initialDesignSize=10, batchSize=5, evaluationBudget=100, surrogate=svmRegressor.SVRModel(), samplingAlgorithm=None):
		# Validate
		assert type(initialDesignSize) is IntType, "initialDesignSize is not an integer: %r" % initialDesignSize
		assert type(batchSize) is IntType, "batchSize is not an integer: %r" % batchSize
		assert type(evaluationBudget) is IntType, "evaluationBudget is not an integer: %r" % evaluationBudget
		assert type(name) is StringType, "name is not a string: %r" % name
		assert isinstance(initialDesign, initialDesignBase.InitialDesignBase), 'initialDesign: Argument of wrong type! Must be a derivative of InitialDesignBase'
		assert isinstance(surrogate, modelBase.ModelBase), 'surrogate: Argument of wrong type! Must be a derivative of ModelBase'
		#assert isinstance(samplingAlgorithm, samplingBase.SamplingBase), 'samplingAlgorithm: Argument of wrong type! Must be a derivative of SamplingBase'
		#@ToDo: assert for sampling, dataset, problem
				
		# Assign
		self.name = name
		self.dataset = dataset
		self.problem = problem
		self.initialDesign = initialDesign
		self.initialDesignSize = initialDesignSize
		self.batchSize = batchSize
		self.evaluationBudget = evaluationBudget
		self.surrogate = surrogate
		self.samplingAlgorithm = samplingAlgorithm

		
	def model(self):
		"""
		Trains an accurate surrogate model according to specified settings
		"""
		# @ToDo: handle errors - undefined variables
		X = self.initialDesign.generate(self.initialDesignSize)
		y = self.problem(X)
		y = y.reshape(len(X),1)
		self.surrogate.train(X,y)
			
	
	def optimize(self):
		"""
		Optimizes a given surrogate model or a simulator
		"""
		bds = [(low, high) for low, high in zip(self.initialDesign.xmin, self.initialDesign.xmax)]
		idx = np.argmin(self.surrogate.y)
		x0 = self.surrogate.X[idx,:]
		minimizer_kwargs = dict(method="L-BFGS-B", bounds=bds)
		res = basinhopping(self.problem, x0, minimizer_kwargs=minimizer_kwargs)
		print res
		
	
	def infer(self):
		"""
		Perform parameter inference or inverse modeling/hotspot detection
		"""