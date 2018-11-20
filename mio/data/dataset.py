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