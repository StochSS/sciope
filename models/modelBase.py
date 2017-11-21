"""
Surrogate Model Base Class
"""

# Imports
from abc import ABCMeta, abstractmethod
import numpy as np

# Class definition
class ModelBase(object):
	"""
	Base class for surrogate models. 
	Must not be used directly!
	Each model type must implement the methods described herein:
	
	* ModelBase.train(X,y)
	* ModelBase.predict(Xt)
	
	The following variables are available to derived classes:
	
	* self.X			(training inputs)
	* self.y			(training targets)
	* self.my			(mean for scaling)
	* self.sy			(std dev for scaling)
	* self.N			(num training points)
	"""
	__metaclass__ = ABCMeta
	
	def __init__(self, name):
		self.name = name
	
	# pre-process training data
	def scaleTrainingData(self, X, y):
		# Take care of NaNs
		if np.isnan(y).any():
			yr = y[~np.isnan(y)]
			my = np.mean(yr)
			y[np.isnan(y)] = my
		
		# ...imaginaries
		y = np.real(y)
		self.N = X.shape[0]
		
		# scale
		self.my = np.mean(y)
		self.sy = np.std(y)
		ry = np.ones((self.N,1)) * self.my
		y = (y - ry) / self.sy
		
		# set data
		self.X = X
		self.y = y.ravel()
		
	
	@abstractmethod
	def train(self, inputs, targets):
		"""
		Sub-classable method for training a given surrogate. Each derived class must implement.
		"""
	
	@abstractmethod
	def predict(self, testInput):
		"""
		Sub-classable method for predicting test data. Each derived class must implement.
		"""