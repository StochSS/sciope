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
Test suite for utilities - distance functions, prior functions and summary statistics
"""
from sciope.utilities.distancefunctions import euclidean, manhattan, naive_squared
from sciope.utilities.priors import uniform_prior
from sciope.utilities.summarystats import auto_tsfresh
from sciope.core import core
from dask.distributed import Client
import numpy as np
from scipy.spatial.distance import cityblock
import dask
import pytest


def test_distance_functions():
    vec_length = 5
    v1 = np.random.rand(1, vec_length)
    v2 = np.random.rand(1, vec_length)

    # calculate Euclidean distance using sciope and numpy
    euc_func = euclidean.EuclideanDistance()
    euc_dist = euc_func.compute(v1, v2)
    validation_val = np.linalg.norm(v1 - v2)
    assert euc_dist == validation_val, "EuclideanDistance functional test error, expected value mismatch"

    # now for Manhattan distance
    man_func = manhattan.ManhattanDistance()
    man_dist = man_func.compute(v1, v2)
    validation_val = cityblock(v1, v2)
    assert man_dist == validation_val, "ManhattanDistance functional test error, expected value mismatch"

    # ... and naive squared distance
    ns_func = naive_squared.NaiveSquaredDistance()
    v3 = np.asarray([0, 0, 0])
    v4 = np.asarray([1, 1, 1])
    validation_val = v4
    ns_dist = ns_func.compute(v3, v4)
    assert np.array_equal(ns_dist, validation_val), "NaiveSquaredDistance functional test error, " \
                                                    "expected value mismatch"


def test_uniform_prior():
    lb = np.asarray([1, 1])
    ub = np.asarray([5, 5])
    num_samples = 5
    prior_func = uniform_prior.UniformPrior(lb, ub)

    # multiprocessing mode
    samples = prior_func.draw(num_samples, chunk_size=1)
    assert len(samples) == 5, "UniformPrior functional test error, expected chunk count mismatch"
    samples, = dask.compute(samples)
    samples = np.asarray(samples)
    assert samples.shape[0] == num_samples, "UniformPrior functional test error, expected sample count mismatch"
    assert samples.shape[1] == 1, "UniformPrior functional test error, expected chunk size mismatch"
    assert samples.shape[2] == len(lb), "UniformPrior functional test error, dimension mismatch"
    samples = samples.reshape(-1, len(lb))
    axis_mins = np.min(samples, 0)
    axis_maxs = np.max(samples, 0)
    assert axis_mins[0] > lb[0] and axis_maxs[0] < ub[0] and axis_mins[1] > lb[1] and axis_maxs[1] < ub[1], \
        "UniformPrior functional test error, drawn samples out of bounds"

    # Cluster mode
    c = Client()
    samples = prior_func.draw(num_samples, chunk_size=1)
    assert len(samples) == 5, "UniformPrior functional test error, expected chunk count mismatch"
    samples, = dask.compute(samples)
    samples = np.asarray(samples)
    assert samples.shape[0] == num_samples, "UniformPrior functional test error, expected sample count mismatch"
    assert samples.shape[1] == 1, "UniformPrior functional test error, expected chunk size mismatch"
    assert samples.shape[2] == len(lb), "UniformPrior functional test error, dimension mismatch"
    samples = samples.reshape(-1, len(lb))
    axis_mins = np.min(samples, 0)
    axis_maxs = np.max(samples, 0)
    assert axis_mins[0] > lb[0] and axis_maxs[0] < ub[0] and axis_mins[1] > lb[1] and axis_maxs[1] < ub[1], \
        "UniformPrior functional test error, drawn samples out of bounds"

    # chunk_size = 2
    samples = prior_func.draw(num_samples, chunk_size=2)
    assert len(samples) == 3, "UniformPrior functional test error, expected chunk count mismatch"
    samples, = dask.compute(samples)
    samples = np.asarray(samples)
    assert samples.shape[0] == 3, "UniformPrior functional test error, expected sample count mismatch"
    assert samples[-1].shape[0] == 2, "UniformPrior functional test error, expected chunk size mismatch"
    assert samples[-1].shape[1] == len(lb), "UniformPrior functional test error, dimension mismatch"
    samples = core._reshape_chunks(samples)
    axis_mins = np.min(samples, 0)
    axis_maxs = np.max(samples, 0)
    assert axis_mins[0] > lb[0] and axis_maxs[0] < ub[0] and axis_mins[1] > lb[1] and axis_maxs[1] < ub[1], \
        "UniformPrior functional test error, drawn samples out of bounds"
    c.close()


def test_distance_functions_with_logging():
    vec_length = 5
    v1 = np.random.rand(1, vec_length)
    v2 = np.random.rand(1, vec_length)

    # calculate Euclidean distance using sciope and numpy
    euc_func = euclidean.EuclideanDistance(use_logger=True)
    euc_dist = euc_func.compute(v1, v2)
    validation_val = np.linalg.norm(v1 - v2)
    assert euc_dist == validation_val, "EuclideanDistance functional test error, expected value mismatch"

    # now for Manhattan distance
    man_func = manhattan.ManhattanDistance(use_logger=True)
    man_dist = man_func.compute(v1, v2)
    validation_val = cityblock(v1, v2)
    assert man_dist == validation_val, "ManhattanDistance functional test error, expected value mismatch"

    # ... and naive squared distance
    ns_func = naive_squared.NaiveSquaredDistance(use_logger=True)
    v3 = np.asarray([0, 0, 0])
    v4 = np.asarray([1, 1, 1])
    validation_val = v4
    ns_dist = ns_func.compute(v3, v4)
    assert np.array_equal(ns_dist, validation_val), "NaiveSquaredDistance functional test error, " \
                                                    "expected value mismatch"


def test_uniform_prior_with_logging():
    lb = np.asarray([1, 1])
    ub = np.asarray([5, 5])
    num_samples = 5
    prior_func = uniform_prior.UniformPrior(lb, ub, use_logger=True)

    # multiprocessing mode
    samples = prior_func.draw(num_samples, chunk_size=1)
    assert len(samples) == 5, "UniformPrior functional test error, expected chunk count mismatch"
    samples, = dask.compute(samples)
    samples = np.asarray(samples)
    assert samples.shape[0] == num_samples, "UniformPrior functional test error, expected sample count mismatch"
    assert samples.shape[1] == 1, "UniformPrior functional test error, expected chunk size mismatch"
    assert samples.shape[2] == len(lb), "UniformPrior functional test error, dimension mismatch"
    samples = samples.reshape(-1, len(lb))
    axis_mins = np.min(samples, 0)
    axis_maxs = np.max(samples, 0)
    assert axis_mins[0] > lb[0] and axis_maxs[0] < ub[0] and axis_mins[1] > lb[1] and axis_maxs[1] < ub[1], \
        "UniformPrior functional test error, drawn samples out of bounds"

    # Cluster mode
    c = Client()
    samples = prior_func.draw(num_samples, chunk_size=1)
    assert len(samples) == 5, "UniformPrior functional test error, expected chunk count mismatch"
    samples, = dask.compute(samples)
    samples = np.asarray(samples)
    assert samples.shape[0] == num_samples, "UniformPrior functional test error, expected sample count mismatch"
    assert samples.shape[1] == 1, "UniformPrior functional test error, expected chunk size mismatch"
    assert samples.shape[2] == len(lb), "UniformPrior functional test error, dimension mismatch"
    samples = samples.reshape(-1, len(lb))
    axis_mins = np.min(samples, 0)
    axis_maxs = np.max(samples, 0)
    assert axis_mins[0] > lb[0] and axis_maxs[0] < ub[0] and axis_mins[1] > lb[1] and axis_maxs[1] < ub[1], \
        "UniformPrior functional test error, drawn samples out of bounds"

    # chunk_size = 2
    samples = prior_func.draw(num_samples, chunk_size=2)
    assert len(samples) == 3, "UniformPrior functional test error, expected chunk count mismatch"
    samples, = dask.compute(samples)
    samples = np.asarray(samples)
    assert samples.shape[0] == 3, "UniformPrior functional test error, expected sample count mismatch"
    assert samples[-1].shape[0] == 2, "UniformPrior functional test error, expected chunk size mismatch"
    assert samples[-1].shape[1] == len(lb), "UniformPrior functional test error, dimension mismatch"
    samples = core._reshape_chunks(samples)
    axis_mins = np.min(samples, 0)
    axis_maxs = np.max(samples, 0)
    assert axis_mins[0] > lb[0] and axis_maxs[0] < ub[0] and axis_mins[1] > lb[1] and axis_maxs[1] < ub[1], \
        "UniformPrior functional test error, drawn samples out of bounds"
    c.close()


def test_summarystats_auto_tsfresh():
    samples = np.random.randn(2, 2, 10)
    # corrcoef = False, will compute mean
    at = auto_tsfresh.SummariesTSFRESH()
    stats = at.compute(samples)
    assert stats.shape == (1, 14), "summarystats auto_tsfresh test failed, dimension mismatch"

    # corrcoef = True
    at.corrcoef = True
    stats = at.compute(samples)
    assert stats.shape == (1, 15), "summarystats auto_tsfresh test failed, dimension mismatch"

    samples = np.random.randn(2, 3, 10)
    # corrcoef = True
    at.corrcoef = True
    stats = at.compute(samples)
    assert stats.shape == (1, 21 + 3), "summarystats auto_tsfresh test failed, dimension mismatch"

    samples = np.random.randn(1, 1, 10)
    # corrcoef = False, will compute mean
    at = auto_tsfresh.SummariesTSFRESH()
    stats = at.compute(samples)
    assert stats.shape == (1, 7), "summarystats auto_tsfresh test failed, dimension mismatch"

    # corrcoef = True, should raise AssertionError
    at.corrcoef = True
    with pytest.raises(AssertionError) as excinfo:
        stats = at.compute(samples)
    assert "corrcoef = True can only be used if the n_species > 1" in str(excinfo.value)

    # input wrong shape, should rasie AssertionError
    samples = np.random.randn(1, 10)
    with pytest.raises(AssertionError) as excinfo:
        stats = at.compute(samples)
    assert "required input shape is (n_points, n_species, n_timepoints)" in str(excinfo.value)
