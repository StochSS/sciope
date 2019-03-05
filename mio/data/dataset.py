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
Dataset Class
"""

# Imports
import numpy as np


# Class definition
class DataSet(object):
	"""
	Class for defining a dataset for a modeling/optimization/inference run 
	
	Properties/variables:
	* x					(inputs)
	* y					(targets)
	
	Methods:
	* set_data			(set inputs and targets)
	* get_size			(returns current size of the dataset)
	* add_points		(updates the dataset to include new points)
	
	"""
	
	def __init__(self, name):
		self.name = name
		self.x = None
		self.y = None
		self.size = 0
		
	def set_data(self, inputs, targets):
		"""
		Sets the inputs and target variables
		"""
		self.x = inputs
		self.y = targets
		self.size = self.x.shape[0]
	
	def get_size(self):
		"""
		Returns the current number of points in the dataset
		"""
		return self.size
		
	def add_points(self, inputs, targets):
		"""
		Updates the dataset to include new points
		"""
		self.x = np.concatenate(self.x, inputs)
		self.y = np.concatenate(self.y, targets)
		self.size = self.x.shape[0]
