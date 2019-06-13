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
Test suite for utilities - distance functions, MABs, prior functions and summary statistics
"""
from sciope.utilities.distancefunctions import euclidean, manhattan, naive_squared
from sciope.utilities.mab import mab_direct, mab_incremental, mab_halving, mab_sar
from sciope.utilities.priors import uniform_prior
from sciope.utilities.summarystats import burstiness, global_max, global_min, temporal_mean, temporal_variance
import numpy as np
from scipy.spatial.distance import cityblock
import pytest


def test_distance_functions():
    vec_length = 5
    v1 = np.random.rand(1, vec_length)
    v2 = np.random.rand(1, vec_length)

    # calculate Euclidean distance using sciope and numpy
    euc_func = euclidean.EuclideanDistance()
    euc_dist = euc_func.compute(v1, v2).compute()
    validation_val = np.linalg.norm(v1 - v2)
    assert euc_dist == validation_val, "EuclideanDistance functional test error, expected value mismatch"

    # now for Manhattan distance
    man_func = manhattan.ManhattanDistance()
    man_dist = man_func.compute(v1, v2).compute()
    validation_val = cityblock(v1, v2)
    assert man_dist == validation_val, "ManhattanDistance functional test error, expected value mismatch"

    # ... and naive squared distance
    ns_func = naive_squared.NaiveSquaredDistance()
    v3 = np.asarray([0, 0, 0])
    v4 = np.asarray([1, 1, 1])
    validation_val = v4
    ns_dist = ns_func.compute(v3, v4).compute()
    assert np.array_equal(ns_dist, validation_val), "NaiveSquaredDistance functional test error, " \
                                                    "expected value mismatch"


def test_uniform_prior():
    lb = np.asarray([1, 1])
    ub = np.asarray([5, 5])
    num_samples = 5
    prior_func = uniform_prior.UniformPrior(lb, ub)
    samples = prior_func.draw(num_samples).compute()
    assert samples.shape[0] == num_samples, "UniformPrior functional test error, expected sample count mismatch"
    assert samples.shape[1] == len(lb), "UniformPrior functional test error, dimension mismatch"
    axis_mins = np.min(samples, 0)
    axis_maxs = np.max(samples, 0)
    assert axis_mins[0] > lb[0] and axis_maxs[0] < ub[0] and axis_mins[1] > lb[1] and axis_maxs[1] < ub[1], \
        "UniformPrior functional test error, drawn samples out of bounds"
