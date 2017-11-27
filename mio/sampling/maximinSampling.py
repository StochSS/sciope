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
Maximin space-filling sampling algorithm
Ranks monte-carlo samples such that minimum distance between them is maximized
"""

# Imports
from samplingBase import SamplingBase
from scipy.spatial import distance_matrix
import numpy as np

# Class definition
class MaximinSampling(SamplingBase):
	"""
	Algorithm:
	1. Generate MC candidate samples
	2. Compute pairwise distance between existing samples and candidates
	3. Select new samples that maximize the minimum distance
	
	Key reference:
	Johnson, Mark E., Leslie M. Moore, and Donald Ylvisaker. 
	"Minimax and maximin distance designs." 
	Journal of statistical planning and inference 26.2 (1990): 131-148.
	"""
	
	def __init__(self, xmin, xmax):
		name = 'MaximinSampling'
		super(MaximinSampling,self).__init__(name, xmin, xmax)	
	
	
	# Example call:
	# ms = MaximinSampling([0,0], [1,1])
	# newPoints = ms.selectPoint(X)
	def selectPoint(self, X):
		"""
		Get top ranked candidate according to maximin sampling to add to current samples X
		"""
		# Set up stuff
		numSamples = X.shape[0]
		numDimensions = X.shape[1]
		candidatesRatio = 10
		numCandidates = numSamples * candidatesRatio
		
		# Generate MC candidates
		c = np.random.uniform(low=self.xmin, high=self.xmax, size=(numCandidates, numDimensions))
		
		# Compute distances
		# p = 1 implies Manhattan distance
		dist = distance_matrix(c, X, p=1)
		
		# Minimum distance...
		ranking = dist.min(axis=1)
		
		# ... is maximized
		idx = np.argsort(-ranking)
		
		# ta-da!
		return c[idx[0],:]
	
	
	def selectPoints(self, X, n):
		"""
		Get 'n' top ranked candidates according to maximin sampling to add to current samples X
		"""
		c = []
		for idx in range(0, n):
			cNew = self.selectPoint(X)
			X = np.vstack((X, cNew))
			c.append(cNew)
		return np.array(c)
		