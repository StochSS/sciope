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
Example: Michaelis-Menten simulator pass-through
Loads a fixed dataset and computes the distance from simulated values
"""
import mm
import numpy as np
import burstiness as bs


def compute(param):
    # param is n-d array passed from GPyOpt so we take param[0]
    # computed_value = model2.simulate(param[0])
    computed_value = mm.simulate(param)
    simulated_ss = bs.compute(computed_value)
    data = np.loadtxt("mm_dataset1000_t500.dat", delimiter=",")

    # Compute stat for each trajectory and then take the mean
    stats = np.apply_along_axis(bs.compute, 0, data)
    data_ss = np.nanmean(stats)

    # dist = (lines.item(randIdx) - computed_value) ** 2
    dist = (data_ss - simulated_ss) ** 2
    return dist
