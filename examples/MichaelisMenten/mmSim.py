import mm
import numpy as np
from random import randint
    
def compute(param):
	# param is n-d array passed from GPyOpt so we take param[0]
	#computedValue = model2.simulate(param[0])
	computedValue = mm.simulate(param)
	lines = np.loadtxt("mmDataset.dat")
	numLines = len(lines)
	randIdx = randint(0,numLines-1)
	dist = (lines.item(randIdx) - computedValue) ** 2
	return dist
