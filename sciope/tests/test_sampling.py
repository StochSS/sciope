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
Test suite for sampling algorithms
"""
from sciope.designs import latin_hypercube_sampling as lhs
from sciope.sampling import maximin_sampling as ms
import numpy as np
import dask
import pytest


def test_maximin_functional():
    # Get 10 points from an initial design
    lb = np.asarray([1, 1])
    ub = np.asarray([9, 9])
    n_dims = ub.size
    n_initial_points = 10
    lhs_obj = lhs.LatinHypercube(lb, ub, use_logger=False)
    lhs_del = lhs_obj.generate(n_initial_points)
    lhs_points, = dask.compute(lhs_del)

    # Get 10 additional points using maximin sampling
    n_new_points = 10
    ms_obj = ms.MaximinSampling(lb, ub, use_logger=False)
    ms_del = ms_obj.select_points(lhs_points, n_new_points)
    ms_points, = dask.compute(ms_del)

    ms_points = np.asarray(ms_points)

    assert ms_points.shape[0] == n_new_points, "MaximinSampling test error, dimensions mismatch"
    assert ms_points.shape[1] == n_dims, "MaximinSampling test error, dimensions mismatch"


def test_maximin_functional_with_logging():
    # Get 10 points from an initial design
    lb = np.asarray([1, 1])
    ub = np.asarray([9, 9])
    n_dims = ub.size
    n_initial_points = 10
    lhs_obj = lhs.LatinHypercube(lb, ub, use_logger=True)
    lhs_del = lhs_obj.generate(n_initial_points)
    lhs_points, = dask.compute(lhs_del)

    # Get 10 additional points using maximin sampling
    n_new_points = 10
    ms_obj = ms.MaximinSampling(lb, ub, use_logger=False)
    ms_del = ms_obj.select_points(lhs_points, n_new_points)
    ms_points, = dask.compute(ms_del)

    ms_points = np.asarray(ms_points)

    assert ms_points.shape[0] == n_new_points, "MaximinSampling test error, dimensions mismatch"
    assert ms_points.shape[1] == n_dims, "MaximinSampling test error, dimensions mismatch"