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
The Uniform Prior
"""

# Imports
from sciope.utilities.priors.prior_base import PriorBase
import numpy as np


# Class definition: Uniform Prior
class UniformPrior(PriorBase):
    """
    The uniform prior draws a sample from the uniform distribution in a specified d-dimensional space described as:
    [min_i, max_i], i=1..d
    """

    def __init__(self, space_min, space_max):
        """
        Set up a uniform prior corresponding to the space bounded by:
        :param space_min: the lowerbound of each variable/dimension
        :param space_max: the upperbound of each variable/dimension
        """
        self.name = 'Uniform'
        self.lb = space_min
        self.ub = space_max
        super(UniformPrior, self).__init__(self.name)

    def draw(self, n=1):
        """
        Draw 'n' samples within self.lb and self.ub
        :param n: the desired number of samples
        :return: the n-sized vector of drawn samples
        """
        d = len(self.lb)

        # Generate samples in [-1,1]
        generated_samples = np.random.random((n, d)) * 2 - 1

        # scale from [-1,1] to problem range
        scaled_values = generated_samples
        for j in range(0, d):
            scaled_values[:, j] = abs((((generated_samples[:, j] + 1) *
                                        (self.ub[j] - self.lb[j])) / 2) + self.lb[j])

        return scaled_values
