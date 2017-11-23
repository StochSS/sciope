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
Initial Design Base Class
"""

# Imports
from abc import ABCMeta, abstractmethod

# Class definition
class InitialDesignBase(object):
	"""
	Base class for initial designs. 
	Must not be used directly!
	Each initial design type must implement the methods described herein:
	
	* InitialDesignBase.generate(n,domain)
	"""
	__metaclass__ = ABCMeta
	
	def __init__(self, name, xmin, xmax):
		self.name = name
		self.xmin = xmin
		self.xmax = xmax
		
	@abstractmethod
	def generate(self, n):
		"""
		Sub-classable method for generating 'n' points within a given domain. Each derived class must implement.
		"""
	