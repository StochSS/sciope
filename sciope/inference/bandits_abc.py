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
Multi-Armed Bandit - Approximate Bayesian Computation
"""

# Imports
from sciope.inference.abc_inference import ABC
from sciope.utilities.mab import mab_direct as md
from sciope.utilities.distancefunctions import euclidean as euc
from sciope.utilities.summarystats import burstiness as bs
from sciope.utilities.housekeeping import sciope_logger as ml
from sciope.utilities.housekeeping import sciope_profiler
from sciope.data.dataset import DataSet
from toolz import partition_all
import multiprocessing as mp  # remove dependency
import numpy as np
import dask


# The following variable stores n normalized distance values after n summary statistics have been calculated
normalized_distances = None


def arm_pull(arm_idx):
    """
    Used by MAB algorithms; Each arm corresponds to a summary statistic and an arm pull is simply selection of one
    (or more) summary statistics in inference. Here that corresponds to simply returning the desired arm.
    :param arm_idx: The index into the vector of arms
    :return: -1 * distance value from distances corresponding to the arm_idx, as reward is to be maximized according to
    MABs but in inference we minimize distance.
    """
    global normalized_distances
    return -1 * normalized_distances[-1, arm_idx]


# Class definition: Bandits-ABC rejection sampling
class BanditsABC(ABC):
    """
    ABC rejection sampling with dynamic multi-armed bandit (MAB) assisted summary statistic selection.
    """

    def __init__(self, data, sim, prior_function, mab_variant=md.MABDirect(arm_pull), k=1, epsilon=0.1,
                 summaries_function=bs.Burstiness(), distance_function=euc.EuclideanDistance(), use_logger=False):
        super().__init__(data, sim, prior_function, epsilon, summaries_function, distance_function, use_logger)
        self.name = 'BanditsABC'
        self.mab_variant = mab_variant
        self.k = k
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Multi-Armed Bandits Approximate Bayesian Computation initialized")

    def scale_distance(self, dist):
        """
         Performs scaling in [0,1] of a given distance vector/value with respect to historical distances

        Parameters
        ----------
        dist : ndarray, float
            distance

        Returns
        -------
        ndarray
            scaled distance
        """
        dist = np.asarray(dist)
        global normalized_distances
        self.historical_distances.append(dist.ravel())
        all_distances = np.array(self.historical_distances)
        divisor = np.asarray(np.nanmax(all_distances, axis=0))
        normalized_distances = all_distances
        for j in range(0, len(divisor), 1):
            if divisor[j] > 0:
                normalized_distances[:, j] = normalized_distances[:, j] / divisor[j]

        return normalized_distances[-1, :]

    #@sciope_profiler.profile
    def rejection_sampling(self, num_samples, batch_size, chunk_size):
        """
        * overrides rejection_sampling of ABC class *
        Perform ABC inference according to initialized configuration.

        Parameters
        ----------
        num_samples : int
            The number of required accepted samples
        batch_size : int
            The batch size of samples for performing rejection sampling
        chunk_size : int
            the partition size when splitting the fixed data. For avoiding many individual tasks
            in dask if the data is large.

        Returns
        -------
        dict
            Keys
            'accepted_samples: The accepted parameter values',
            'distances: Accepted distance values',
            'accepted_count: Number of accepted samples',
            'trial_count: The number of total trials performed in order to converge',
            'inferred_parameters': The mean of accepted parameter samples
        """
        accepted_count = 0
        trial_count = 0
        accepted_samples = []
        distances = []

        # if fixed_mean has not been computed
        if not self.fixed_mean:
            self.compute_fixed_mean(chunk_size)

        # Get dask graph
        graph_dict = self.get_dask_graph(batch_size)

        # do rejection sampling
        while accepted_count < num_samples:

            res_param, res_dist = dask.compute(graph_dict["parameters"], graph_dict["distances"])

            # Normalize distances between [0,1]
            sim_dist_scaled = np.asarray([self.scale_distance(dist) for dist in res_dist])

            # Use MAB arm selection to identify the best 'k' arms or summary statistics
            num_arms = sim_dist_scaled.shape[1]
            arms = range(num_arms)
            top_k_arms_idx = self.mab_variant.select(arms, self.k)

            print("top_k_arms_idx: ", top_k_arms_idx)

            top_k_distances = np.asarray([sim_dist_scaled[:, i] for i in top_k_arms_idx])
            top_k_distances = top_k_distances.transpose()


            #Mattias lines
            print("top_k_distances shape: ", top_k_distances.shape)
            #

            # Take the norm to combine the distances, if more than one summary is used
            if top_k_distances.shape[1] > 1:
                combined_distance = [dask.delayed(np.linalg.norm)(scaled) for scaled in top_k_distances]
                result, = dask.compute(combined_distance)
            else:
                result = top_k_distances.ravel()

            # Accept/Reject
            for e, res in enumerate(result):
                if self.use_logger:
                    self.logger.debug("Bandits-ABC Rejection Sampling: trial parameter(s) = {}".format(res_param[e]))
                    self.logger.debug("Bandits-ABC Rejection Sampling: trial distance(s) = {}".format(res_dist[e]))
                if res <= self.epsilon:
                    accepted_samples.append(res_param[e])
                    distances.append(res_dist[e])
                    accepted_count += 1
                    if self.use_logger:
                        self.logger.info("Bandits-ABC Rejection Sampling: accepted a new sample, "
                                         "total accepted samples = {0}".format(accepted_count))

            trial_count += batch_size

        self.results = {'accepted_samples': accepted_samples, 'distances': distances, 'accepted_count': accepted_count,
                        'trial_count': trial_count, 'inferred_parameters': np.mean(accepted_samples, axis=0)}
        return self.results
