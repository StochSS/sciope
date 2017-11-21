"""
Latin Hypercube Sampling Initial Design
"""

# Imports
from initialDesignBase import InitialDesignBase
import gpflowopt

# Class definition
class LatinHypercube(InitialDesignBase):
	"""
	Latin Hypercube Sampling implemented through pyDOE
	
	* InitialDesignBase.generate(n,domain)
	"""
	
	def __init__(self, xmin, xmax):
		name = 'LatinHypercube'
		super(LatinHypercube,self).__init__(name, xmin, xmax)
		
	def generate(self, n):
		"""
		Sub-classable method for generating 'n' points in the given 'domain'.
		"""
		numVariables = len(self.xmin)
		gpfDomain = gpflowopt.domain.ContinuousParameter('x0', self.xmin[0], self.xmax[0])
		for i in range(1, numVariables):
			varName = 'x' + `i`
			gpfDomain = gpfDomain + gpflowopt.domain.ContinuousParameter(varName, self.xmin[i], self.xmax[i])
		
		design = gpflowopt.design.LatinHyperCube(n, gpfDomain)
		return design.generate()
	