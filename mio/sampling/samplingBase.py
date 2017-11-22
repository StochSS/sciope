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
Sequential Sampling Base Class
"""

# Imports
from abc import ABCMeta, abstractmethod

# Class definition
class SamplingBase(object):
	"""
	Base class for sequential sampling. 
	Must not be used directly!
	Each sampling algorithm must implement the methods described herein:
	
	* SamplingBase.selectPoint()
	* SamplingBase.selectPoints(n)
	
	The following variables are available to derived classes:
	*
	"""
	__metaclass__ = ABCMeta
	
	def __init__(self, name):
		self.name = name
		
	@abstractmethod
	def selectPoint(self):
		"""
		Sub-classable method for selecting one new point. Each derived class must implement.
		"""
	
	@abstractmethod
	def selectPoints(self, n):
		"""
		Sub-classable method for selecting 'n' new points. Each derived class must implement.
		"""