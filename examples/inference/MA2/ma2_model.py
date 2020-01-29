# Copyright 2019 Mattias Åkesson, Prashant Singh, Fredrik Wrede and Andreas Hellander
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
The Moving Averages 2 model
"""
import numpy as np


def simulate(param, n=100):
    """
    Simulate a given parameter combination.

    Parameters
    ----------
    param : vector or 1D array
        Parameters to simulate (\theta).
    n : integer
        Time series length
    """
    m = len(param)
    g = np.random.normal(0, 1, n)
    gy = np.random.normal(0, 0.3, n)
    y = np.zeros(n)
    x = np.zeros(n)
    for t in range(0, n):
        # print("t: ", t)
        x[t] += g[t]
        # print("g: ", "{0:.2f}".format(g[t]))
        for p in range(0, np.minimum(t, m)):
            x[t] += g[t - 1 - p] * param[p]
        y[t] = x[t] + gy[t]

    return y


def prior(n=10):
    """
    Sample parameters or thetas from the prior.

    Parameters
    ----------
    n : integer
        Number of random samples to draw or sample.
    """
    p = []
    trials = 0
    acc = 0
    while acc < n:
        trials += 1
        r = np.random.rand(2) * np.array([4, 2]) + np.array([-2, -1])
        # print("r: ", r)
        if r[1] + r[0] >= -1 and r[1] - r[0] >= -1:
            p.append(r)
            acc += 1
    # print("trials: ", trials, ", acc: ", acc)
    return p
