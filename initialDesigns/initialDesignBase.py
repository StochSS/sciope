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
	def generate(self, n, domain):
		"""
		Sub-classable method for generating 'n' points within a given 'domain'. Each derived class must implement.
		"""
	