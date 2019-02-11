# Copyright 2019 Prashant Singh, Fredrik Wrede and Andreas Hellander
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
The global maximum summary statistic
"""

# Imports
import numpy as np
from summary_base import SummaryBase


# Class definition: Global Min Statistic
class GlobalMax(SummaryBase):
    """
    The maximum value observed across the entire time span
    """

    def __init__(self, mean_trajectories=True):
        self.name = 'GlobalMax'
        super(GlobalMax, self).__init__(self.name, mean_trajectories)

    def compute(self, data):
        """
        Calculate the value(s) of the summary statistic(s)
        :param data: simulated or data set
        :return: computed statistic value
        """
        if self.mean_trajectories:
            return np.mean(np.max(data, axis=1))
        else:
            return np.max(data, axis=1)
