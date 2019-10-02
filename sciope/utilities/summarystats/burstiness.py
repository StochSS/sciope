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
The 'Burstiness' summary statistic
"""

# Imports
import numpy as np
import math as mt
from sciope.utilities.summarystats.summary_base import SummaryBase
from sciope.utilities.housekeeping import sciope_logger as ml


# Class definition: Burstiness Statistic
class Burstiness(SummaryBase):
    """
    Burstiness Summary statictics
    Burstiness = (sigma-mu)/(sigma+mu)

    Ref: Burstiness and memory in complex systems, Europhys. Let., 81, pp. 48002, 2008.
    """

    def __init__(self, mean_trajectories=False, improvement=False, use_logger=False):
        """
        [summary]
        
        Parameters
        ----------
        mean_trajectories : bool, optional
            [description], by default True
        improvement : bool, optional
            [description], by default False
        """
        self.name = 'Burstiness'
        self.improvement = improvement
        super(Burstiness, self).__init__(self.name, mean_trajectories, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Burstiness summary statistic initialized")

    def _compute(self, data):
        """
        Calculates the measure per specie
        Parameters
        ----------
        data : [type]
            simulated or data set in the form N x S X T - num data points x num species x num time steps

        Returns
        -------
        [type]
            computed statistic value
        """
        trajs = []
        for i in range(np.shape(data)[0]):
            y = data[i, :]
            r = np.std(y) / np.mean(y)
            if not self.improvement:
                # original burstiness due to Goh and Barabasi
                out = (r - 1) / (r + 1)
            else:
                # improvement by Kim & Ho, 2016 (arxiv)
                n = len(y)
                out = (mt.sqrt(n + 1) * r - mt.sqrt(n - 1)) / ((mt.sqrt(n + 1) - 2) * r + mt.sqrt(n - 1))

            trajs.append(out)

        out = np.array(trajs)
        res = np.reshape(out, (out.size, 1))
        return res

    def compute(self, data):
        """
        Calculate the value(s) of the summary statistic(s)
        
        Parameters
        ----------
        data : [type]
            simulated or data set in the form N x S X T - num data points x num species x num time steps
        
        Returns
        -------
        [type]
            computed statistic value
        
        """
        data_arr = np.array(data)
        assert len(data_arr.shape) == 3, "required input shape is (n_points, n_species, n_timepoints)"

        res = []
        for i in range(data_arr.shape[1]):
            bs_value = self._compute(data_arr[:, i, :].reshape((data_arr.shape[0], data_arr.shape[2])))
            res.append(bs_value.ravel())
        res = np.asarray(res)
        res = res.transpose()

        if self.mean_trajectories:
            res = np.asarray(np.mean(res, axis=0))  # returns a scalar, so we cast it as an array

        if self.use_logger:
            self.logger.info("Burstiness summary statistic: processed data matrix of shape {0} and generated summaries"
                             " of shape {1}".format(data.shape, res.shape))
        return res
