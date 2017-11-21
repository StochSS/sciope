"""
Surrogate Modeling and Optimization Class
Forms the backbone of the following tasks:
* training accurate surrogate models
* performing model-based optimization
* performing parameter inference, hotspot detection and inverse modeling
"""

# Imports
from scipy.optimize import basinhopping
import numpy as np

# Class definition
class MIO(object):
	"""
	Class for (M)odeling, (I)nference and (O)ptimization. 
	
	Properties/variables:
	* initial design					(object of initialDesignBase)
	* surrogate model					(object of modelBase)
	* sequential sampling algorithm		(object of samplingBase)
	* dataset							(object of dataset)
	
	Methods:
	* model								(trains a surrogate model with specified settings)
	* optimize							(optimize a given surrogate model or simulator)
	* infer								(infer the parameters of a simulator, given some data)
	
	"""
	
	def __init__(self, name='default', dataset=None, problem=None, initialDesign=None, initialDesignSize=10, batchSize=5, evaluationBudget=100, surrogate=None):
		self.name = name
		self.dataset = dataset
		self.problem = problem
		self.initialDesign = initialDesign
		self.initialDesignSize = initialDesignSize
		self.batchSize = batchSize
		self.evaluationBudget = evaluationBudget
		self.surrogate = surrogate

		
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