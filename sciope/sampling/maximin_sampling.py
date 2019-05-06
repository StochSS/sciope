# Copyright 2017 Prashant Singh, Fredrik Wrede and Andreas Hellander
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
from sciope.sampling.sampling_base import SamplingBase
from scipy.spatial import distance_matrix
from sciope.utilities.housekeeping import sciope_logger as ml
import numpy as np

# Set up the logger
logger = ml.SciopeLogger().get_logger()


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
        super(MaximinSampling, self).__init__(name, xmin, xmax)
        logger.info("Maximin sequential sampler in {0} dimensions initialized".format(len(self.xmin)))

    # Example call:
    # ms = MaximinSampling([0,0], [1,1])
    # new_points = ms.select_point(X)
    def select_point(self, x):
        """
        Get top ranked candidate according to maximin sampling to add to current samples x
        """
        # Set up stuff
        num_samples = x.shape[0]
        num_dimensions = x.shape[1]
        candidates_ratio = 10
        num_candidates = num_samples * candidates_ratio

        # Generate MC candidates
        c = np.random.uniform(low=self.xmin, high=self.xmax, size=(num_candidates, num_dimensions))

        # Compute distances
        # p = 1 implies Manhattan distance
        dist = distance_matrix(c, x, p=1)

        # Minimum distance...
        ranking = dist.min(axis=1)

        # ... is maximized
        idx = np.argsort(-ranking)

        # ta-da!
        logger.info("Maximin sequential design: selected one new sample")
        return c[idx[0], :]

    def select_points(self, x, n):
        """
        Get 'n' top ranked candidates according to maximin sampling to add to current samples x
        """
        c = []
        for idx in range(0, n):
            c_new = self.selectPoint(x)
            x = np.vstack((x, c_new))
            c.append(c_new)
        logger.info("Maximin sequential design: selected {0} new samples".format(n))
        return np.array(c)
