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
Test suite for inference algorithms
"""
from sciope.inference import abc_inference
from sciope.inference import bandits_abc
from sciope.utilities.priors import uniform_prior
from sciope.utilities.summarystats import burstiness as bs
import numpy as np
import sys
from sciope.utilities.mab import mab_halving as mh, mab_sar as sar, mab_direct as md, mab_incremental as mi
from sciope.utilities.distancefunctions import naive_squared as ns
from sciope.utilities.distancefunctions import euclidean as euc
import pytest

sys.path.append('../../examples/inference/vilar')
import vilar
import summaries_tsa as tsa
import summaries_ensemble as se
from sklearn.metrics import mean_absolute_error
from distributed import Client, LocalCluster

# Load data
data = np.loadtxt("../../examples/inference/vilar/datasets/vilar_dataset_specieA_50trajs_15time.dat", delimiter=",")

# True parameter
true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]

# Set up the prior, distance functions and summary statistics
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
mm_prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
bs_stat = bs.Burstiness(mean_trajectories=True, use_logger=False)

# Set up dask
cluster = LocalCluster()
client = Client(cluster)

# For Bandits ABC
dist_fun = ns.NaiveSquaredDistance(use_logger=False)
sum_stats = tsa.SummariesTSFRESH()
mab_algo = mh.MABHalving(bandits_abc.arm_pull)


def test_abc_functional():
    abc_instance = abc_inference.ABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior,
                                     summaries_function=bs_stat)
    abc_instance.infer(num_samples=30, batch_size=10)
    mae_inference = mean_absolute_error(true_params, abc_instance.results['inferred_parameters'])
    assert abc_instance.results['trial_count'] > 0 and abc_instance.results['trial_count'] < 300, \
        "ABC inference test failed, trial count out of bounds"
    assert mae_inference < 15, "ABC inference test failed, error too high"


def test_bandits_abc_functional():
    abc_instance = bandits_abc.BanditsABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior, k=1,
                                          distance_function=dist_fun,
                                          summaries_function=sum_stats,
                                          mab_variant=mab_algo)
    abc_instance.infer(num_samples=30, batch_size=10)
    mae_inference = mean_absolute_error(true_params, abc_instance.results['inferred_parameters'])
    assert abc_instance.results['trial_count'] > 0 and abc_instance.results['trial_count'] < 300, \
        "Bandits ABC inference test failed, trial count out of bounds"
    assert mae_inference < 15, "Bandits ABC inference test failed, error too high"


def test_bandits_abc_functional_direct():
    mab_algo = md.MABDirect(bandits_abc.arm_pull)
    abc_instance = bandits_abc.BanditsABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior, k=1,
                                          distance_function=dist_fun,
                                          summaries_function=sum_stats,
                                          mab_variant=mab_algo)
    abc_instance.infer(num_samples=30, batch_size=10)
    mae_inference = mean_absolute_error(true_params, abc_instance.results['inferred_parameters'])
    assert abc_instance.results['trial_count'] > 0 and abc_instance.results['trial_count'] < 300, \
        "Bandits ABC inference test failed, trial count out of bounds"
    assert mae_inference < 15, "Bandits ABC inference test failed, error too high"


def test_bandits_abc_functional_sar():
    mab_algo = sar.MABSAR(arm_pull=bandits_abc.arm_pull, p=50, b=500)
    abc_instance = bandits_abc.BanditsABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior, k=1,
                                          distance_function=dist_fun,
                                          summaries_function=sum_stats,
                                          mab_variant=mab_algo)
    abc_instance.infer(num_samples=30, batch_size=10)
    mae_inference = mean_absolute_error(true_params, abc_instance.results['inferred_parameters'])
    assert abc_instance.results['trial_count'] > 0 and abc_instance.results['trial_count'] < 300, \
        "Bandits ABC inference test failed, trial count out of bounds"
    assert mae_inference < 15, "Bandits ABC inference test failed, error too high"


def test_abc_with_logging():
    abc_instance = abc_inference.ABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior,
                                     summaries_function=bs_stat, use_logger=True)
    abc_instance.infer(num_samples=30, batch_size=10)
    mae_inference = mean_absolute_error(true_params, abc_instance.results['inferred_parameters'])
    assert abc_instance.results['trial_count'] > 0 and abc_instance.results['trial_count'] < 300, \
        "ABC inference test failed, trial count out of bounds"
    assert mae_inference < 15, "ABC inference test failed, error too high"


def test_bandits_abc_with_logging():
    mab_algo = mh.MABHalving(arm_pull=bandits_abc.arm_pull, use_logger=True)
    abc_instance = bandits_abc.BanditsABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior, k=1,
                                          distance_function=dist_fun,
                                          summaries_function=sum_stats,
                                          mab_variant=mab_algo,
                                          use_logger=True)
    abc_instance.infer(num_samples=30, batch_size=10)
    mae_inference = mean_absolute_error(true_params, abc_instance.results['inferred_parameters'])
    assert abc_instance.results['trial_count'] > 0 and abc_instance.results['trial_count'] < 300, \
        "Bandits ABC inference test failed, trial count out of bounds"
    assert mae_inference < 15, "Bandits ABC inference test failed, error too high"


def test_bandits_abc_functional_direct_with_logging():
    mab_algo = md.MABDirect(arm_pull=bandits_abc.arm_pull, use_logger=True)
    abc_instance = bandits_abc.BanditsABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior, k=1,
                                          distance_function=dist_fun,
                                          summaries_function=sum_stats,
                                          mab_variant=mab_algo,
                                          use_logger=True)
    abc_instance.infer(num_samples=30, batch_size=10)
    mae_inference = mean_absolute_error(true_params, abc_instance.results['inferred_parameters'])
    assert abc_instance.results['trial_count'] > 0 and abc_instance.results['trial_count'] < 300, \
        "Bandits ABC inference test failed, trial count out of bounds"
    assert mae_inference < 15, "Bandits ABC inference test failed, error too high"


def test_bandits_abc_functional_sar_with_logging():
    mab_algo = sar.MABSAR(arm_pull=bandits_abc.arm_pull, p=50, b=500, use_logger=True)
    abc_instance = bandits_abc.BanditsABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior, k=1,
                                          distance_function=dist_fun,
                                          summaries_function=sum_stats,
                                          mab_variant=mab_algo,
                                          use_logger=True)
    abc_instance.infer(num_samples=30, batch_size=10)
    mae_inference = mean_absolute_error(true_params, abc_instance.results['inferred_parameters'])
    assert abc_instance.results['trial_count'] > 0 and abc_instance.results['trial_count'] < 300, \
        "Bandits ABC inference test failed, trial count out of bounds"
    assert mae_inference < 15, "Bandits ABC inference test failed, error too high"


def test_simple_summary_stats():
    abc_instance = abc_inference.ABC(data, vilar.simulate, epsilon=0.1, prior_function=mm_prior,
                                     summaries_function=se.SummariesEnsemble(),
                                     distance_function=euc.EuclideanDistance(), use_logger=False)
    abc_instance.infer(num_samples=30, batch_size=10)
    mae_inference = mean_absolute_error(true_params, abc_instance.results['inferred_parameters'])
    assert abc_instance.results['trial_count'] > 0 and abc_instance.results['trial_count'] < 500, \
        "ABC inference test failed, trial count out of bounds"
    assert mae_inference < 15, "ABC inference test failed, error too high"