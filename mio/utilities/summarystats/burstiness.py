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
from mio.utilities.summarystats.summary_base import SummaryBase


# Class definition: Burstiness Statistic
class Burstiness(SummaryBase):
    """
    Burstiness = (sigma-mu)/(sigma+mu)

    Ref: Burstiness and memory in complex systems, Europhys. Let., 81, pp. 48002, 2008.
    """

    def __init__(self, mean_trajectories=True, improvement=False):
        self.name = 'Burstiness'
        self.improvement = improvement
        super(Burstiness, self).__init__(self.name, mean_trajectories)

    def compute(self, data):
        """
        Calculate the value(s) of the summary statistic(s)
        :param data: simulated or data set
        :return: computed statistic value
        """
        if self.mean_trajectories:
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

            out = np.array(np.mean(np.array(trajs)))
            return out.reshape(1, 1)

        else:
            r = np.std(data) / np.mean(data)

            # original burstiness due to Goh and Barabasi
            if not self.improvement:
                out1 = np.asarray((r - 1) / (r + 1))
                return out1.reshape(1, 1)
            else:
                # improvement by Kim & Ho, 2016 (arxiv)
                n = len(data)
                out2 = np.asarray((mt.sqrt(n + 1) * r - mt.sqrt(n - 1)) / ((mt.sqrt(n + 1) - 2) * r + mt.sqrt(n - 1)))
                return out2.reshape(1, 1)
