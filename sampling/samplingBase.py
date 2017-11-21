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