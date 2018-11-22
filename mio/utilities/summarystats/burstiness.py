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
from summary_base import SummaryBase


# Class definition: Burstiness Statistic
class Burstiness(SummaryBase):
    """
    Burstiness = (sigma-mu)/(sigma+mu)

    Ref: Burstiness and memory in complex systems, Europhys. Let., 81, pp. 48002, 2008.
    """

    def __init__(self, name):
        self.name = 'Burstiness'
        super(Burstiness, self).__init__(self.name)

    @staticmethod
    def compute(self, data, improvement=False):
        r = np.std(data) / np.mean(data)

        # original burstiness due to Goh and Barabasi
        if not improvement:
            out1 = (r - 1) / (r + 1)
            return out1
        else:
            # improvement by Kim & Ho, 2016 (arxiv)
            n = len(data)
            out2 = (mt.sqrt(n + 1) * r - mt.sqrt(n - 1)) / ((mt.sqrt(n + 1) - 2) * r + mt.sqrt(n - 1))
            return out2
