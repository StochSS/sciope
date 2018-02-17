# The 'Burstiness' summary statistic
# - Calculates the burstiness summary statistic of a time series
# (c) Andreas Hellander, Prashant Singh; Uppsala University, Sweden.
#
# Burstiness = (sigma-mu)/(sigma+mu)
#
# ver 0.1 08 Sep 2017
#
# Notes: Burstiness and memory in complex systems, Europhys. Let., 81, pp. 48002, 2008.
# --

import numpy as np
import math as mt

def compute(y):
	r = np.std(y)/np.mean(y)
	
	# original burstiness due to Goh and Barabasi
	out1 = (r-1)/(r+1)
	return out1
	# improvement by Kim & Ho, 2016 (arxiv)
	#N = len(y)
	#out2 = (mt.sqrt(N+1)*r - mt.sqrt(N-1))/((mt.sqrt(N+1)-2)*r + mt.sqrt(N-1))
	#return out2
