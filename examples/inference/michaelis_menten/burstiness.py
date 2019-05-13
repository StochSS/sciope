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
Example: Michaelis-Menten chemical kinetics

The 'Burstiness' summary statistic

Burstiness = (sigma-mu)/(sigma+mu)

ver 0.1 08 Sep 2017
Notes: Burstiness and memory in complex systems, Europhys. Let., 81, pp. 48002, 2008.
"""

import numpy as np
import math as mt


def compute(y):
    r = np.std(y) / np.mean(y)

    # original burstiness due to Goh and Barabasi
    out1 = (r - 1) / (r + 1)
    return out1

# improvement by Kim & Ho, 2016 (arxiv)
# N = len(y)
# out2 = (mt.sqrt(N+1)*r - mt.sqrt(N-1))/((mt.sqrt(N+1)-2)*r + mt.sqrt(N-1))
# return out2
# '''
