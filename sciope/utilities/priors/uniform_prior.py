# Copyright 2020 Prashant Singh, Richard Jiang, Fredrik Wrede and Andreas Hellander
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
from sciope.utilities.housekeeping import sciope_logger as ml
from dask import delayed
import numpy as np


# Class definition: Uniform Prior
class UniformPrior(PriorBase):
    """
    The uniform prior draws a sample from the uniform distribution in a specified d-dimensional space described as:
    [min_i, max_i], i=1..d
    """

    def __init__(self, space_min, space_max, use_logger=False):
        """
        Set up a uniform prior corresponding to the space bounded by:
        :param space_min: the lowerbound of each variable/dimension
        :param space_max: the upperbound of each variable/dimension
        :param use_logger: whether logging is enabled or disabled
        """
        self.name = 'Uniform'
        self.lb = space_min
        self.ub = space_max
        super(UniformPrior, self).__init__(self.name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Uniform prior in {} dimensions initialized".format(len(self.lb)))

    def draw(self, n=1, chunk_size=1):
        """
        Draw 'n' samples within self.lb and self.ub
        :param n: the desired number of samples
        :return: the n-sized vector of drawn samples
        """
        assert n >= chunk_size, "chunk_size can not be larger than n"

        d = len(self.lb)

        generated_samples = []
        m = n % chunk_size
        if m > 0:
            generated_samples.append(self._uniform_scale(m, d))

        for i in range(0, n - m, chunk_size):
            generated_samples.append(self._uniform_scale(chunk_size, d))

        return generated_samples

    def pdf(self, x, log=False):
        if len(np.asarray(x).shape) == 1:
            z = np.asarray(x)
            if (z > self.lb).all() and (z < self.ub).all():
                v = np.prod(1 / (self.ub - self.lb))
                if log:
                    v = np.log(v)
            else:
                v = 0
                if log:
                    v = -np.inf
            return v
        else:
            z = np.asarray(x)
            vs = []
            for i in range(z.shape[0]):
                if (z[i] > self.lb).all() and (z[i] < self.ub).all():
                    v = np.prod(1 / (self.ub - self.lb))
                    if log:
                        vs.append(np.log(v))
                    else:
                        vs.append(v)
                else:
                    if log:
                        vs.append(-np.inf)
                    else:
                        vs.append(0)
            return np.asarray(vs)
        return v

    def get_dimension(self):
        return len(self.lb)

    @delayed
    def _uniform_scale(self, n, d):
        # Generate samples in [0,1)
        generated_samples = np.random.random((n, d))

        # scale from [0,1) to problem range
        scaled_values = generated_samples
        for j in range(0, d):
            scaled_values[:, j] = (generated_samples[:, j] * (self.ub[j] - self.lb[j])) + self.lb[j]
        if self.use_logger:
            self.logger.info("Uniform Prior: sampled {} points in {} dimensions".format(n, len(self.lb)))

        return scaled_values
