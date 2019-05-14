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
from dask import delayed
from sciope.utilities.summarystats.summary_base import SummaryBase


# Class definition: Burstiness Statistic
class Burstiness(SummaryBase):
    """
    Burstiness Summary statictics
    Burstiness = (sigma-mu)/(sigma+mu)

    Ref: Burstiness and memory in complex systems, Europhys. Let., 81, pp. 48002, 2008.
    """

    def __init__(self, mean_trajectories=True, improvement=False):
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
        super(Burstiness, self).__init__(self.name, mean_trajectories)

    @delayed
    def compute(self, data):
        """
        Calculate the value(s) of the summary statistic(s)
        
        Parameters
        ----------
        data : [type]
            simulated or data set
        
        Returns
        -------
        [type]
            computed statistic value
        
        """
        data_arr = np.array(data)
        trajs = []
        for i in range(np.shape(data)[0]):
            y = data_arr[i, :]
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
        np.testing.assert_equal(out.size, data_arr.shape[0], "Burstiness: expected summaries count mismatch!")
        return np.reshape(out, (out.size, 1))

