"""
Dataset Class
"""

# Imports
import numpy as np


# Class definition
class dataset(object):
	"""
	Class for defining a dataset for a modeling/optimization/inference run 
	
	Properties/variables:
	* X					(inputs)
	* y					(targets)
	
	Methods:
	* setData			(set inputs and targets)
	* getSize			(returns current size of the dataset)
	* addPoints			(updates the dataset to include new points)
	
	"""
	
	def __init__(self, name):
		self.name = name
		
	def setData(self, inputs, targets):
		"""
		Sets the inputs and target variables
		"""
		this.X = inputs
		this.y = targets
		this.size = this.X.shape[0]
	
	def getSize(self):
		"""
		Returns the current number of points in the dataset
		"""
		return this.size
		
	def addPoints(self, inputs, targets):
		"""
		Updates the dataset to include new points
		"""
		this.X = np.concatenate(this.X, inputs)
		this.y = np.concatenate(this.y, targets)
		this.size = this.X.shape[0]