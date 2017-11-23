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
Example: Michaelis-Menten simulator pass-through
Loads a fixed dataset and computes the distance from simulated values
"""
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
