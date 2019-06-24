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
Test suite for initial designs
"""
from sciope.designs import latin_hypercube_sampling as lhs
from sciope.designs import random_sampling as rs
from sciope.designs import factorial_design as fd
import numpy as np
from distributed import Client, LocalCluster
import pytest

# Set up dask
cluster = LocalCluster()
client = Client(cluster)


def test_lhs_functional():
    lb = np.asarray([1, 1, 1])
    ub = np.asarray([9, 9, 9])
    num_points = 30
    num_dims = lb.size
    lhs_obj = lhs.LatinHypercube(lb, ub, use_logger=False)
    lhs_delayed = lhs_obj.generate(num_points)
    lhs_points = lhs_delayed.compute()
    assert lhs_points.shape[0] == num_points, "LatinHypercube test failed, dimensions mismatch"
    assert lhs_points.shape[1] == num_dims, "LatinHypercube test failed, dimensions mismatch"

    num_samples = 10
    samples = lhs_obj.draw(n=num_points, n_samples=num_samples)

    assert len(samples) == 10, "LatinHypercube sampling test failed, dimensions mismatch"
    assert len(lhs_obj.random_idx) == 20, "LatinHypercube sampling test failed, dimensions mismatch"
    for d in samples:
         sample = d.compute()
         assert sample.shape == (num_dims,)
    
    samples = lhs_obj.draw(n_samples=5)
    assert len(samples) == 5, "LatinHypercube sampling test failed, dimensions mismatch"
    assert len(lhs_obj.random_idx) == 15, "LatinHypercube sampling test failed, dimensions mismatch"

    samples = lhs_obj.draw(n_samples=20)
    assert len(samples) == 15, "LatinHypercube sampling test failed, dimensions mismatch"
    assert len(lhs_obj.random_idx) == 0, "LatinHypercube sampling test failed, dimensions mismatch"

    samples = lhs_obj.draw(n=num_points, n_samples=5)
    assert len(samples) == 5, "LatinHypercube sampling test failed, dimensions mismatch"
    assert len(lhs_obj.random_idx) == 25, "LatinHypercube sampling test failed, dimensions mismatch"
    


def test_random_functional():
    lb = np.asarray([1, 1, 1])
    ub = np.asarray([9, 9, 9])
    num_points = 30
    num_dims = lb.size
    rs_obj = rs.RandomSampling(lb, ub, use_logger=False)
    rs_delayed = rs_obj.generate(num_points)
    rs_points = rs_delayed.compute()
    assert rs_points.shape[0] == num_points, "RandomSampling test failed, dimensions mismatch"
    assert rs_points.shape[1] == num_dims, "RandomSampling test failed, dimensions mismatch"


def test_factorial_functional():
    lb = np.asarray([1, 1, 1])
    ub = np.asarray([9, 9, 9])
    num_levels = 3
    num_dims = lb.size
    fd_obj = fd.FactorialDesign(num_levels, lb, ub, use_logger=False)
    fd_delayed = fd_obj.generate()
    fd_points = fd_delayed.compute()
    assert fd_points.shape[0] == np.power(num_levels, num_dims), "FactorialDesign test failed, dimensions mismatch"
    assert fd_points.shape[1] == num_dims, "FactorialDesign test failed, dimensions mismatch"


def test_lhs_functional_with_logging():
    lb = np.asarray([1, 1, 1])
    ub = np.asarray([9, 9, 9])
    num_points = 30
    num_dims = lb.size
    lhs_obj = lhs.LatinHypercube(lb, ub, use_logger=True)
    lhs_delayed = lhs_obj.generate(num_points)
    lhs_points = lhs_delayed.compute()
    assert lhs_points.shape[0] == num_points, "LatinHypercube test failed, dimensions mismatch"
    assert lhs_points.shape[1] == num_dims, "LatinHypercube test failed, dimensions mismatch"


def test_random_functional_with_logging():
    lb = np.asarray([1, 1, 1])
    ub = np.asarray([9, 9, 9])
    num_points = 30
    num_dims = lb.size
    rs_obj = rs.RandomSampling(lb, ub, use_logger=True)
    rs_delayed = rs_obj.generate(num_points)
    rs_points = rs_delayed.compute()
    assert rs_points.shape[0] == num_points, "RandomSampling test failed, dimensions mismatch"
    assert rs_points.shape[1] == num_dims, "RandomSampling test failed, dimensions mismatch"


def test_factorial_functional_with_logging():
    lb = np.asarray([1, 1, 1])
    ub = np.asarray([9, 9, 9])
    num_levels = 3
    num_dims = lb.size
    fd_obj = fd.FactorialDesign(num_levels, lb, ub, use_logger=True)
    fd_delayed = fd_obj.generate()
    fd_points = fd_delayed.compute()
    assert fd_points.shape[0] == np.power(num_levels, num_dims), "FactorialDesign test failed, dimensions mismatch"
    assert fd_points.shape[1] == num_dims, "FactorialDesign test failed, dimensions mismatch"
