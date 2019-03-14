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
from abc_inference import ABC
import numpy as np
from data.dataset import DataSet
from utilities.distancefunctions import euclidean as euc
from utilities.summarystats import burstiness as bs
from utilities.mab import mab_direct as md
from utilities.housekeeping import mio_logger as ml
from utilities.housekeeping import mio_profiler as mp


# The following variable stores n normalized distance values after n summary statistics have been calculated
normalized_distances = None

# Set up the logger and profiler
logger = ml.MIOLogger().get_logger()


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
                 parallel_mode=True, summaries_function=bs.Burstiness(), distance_function=euc.EuclideanDistance()):
        super().__init__(data, sim, prior_function, epsilon, parallel_mode, summaries_function, distance_function)
        self.name = 'BanditsABC'
        self.mab_variant = mab_variant
        self.k = k
        logger.info("Multi-Armed Bandits Approximate Bayesian Computation initialized")

    def scale_distance(self, dist):
        """
        Performs scaling in [0,1] of a given distance vector/value with respect to historical distances
        :param dist: a distance value or vector
        :return: scaled distance value or vector
        """
        global normalized_distances
        self.historical_distances.append(dist.ravel())
        all_distances = np.array(self.historical_distances)
        divisor = np.asarray(all_distances.max(axis=0))
        normalized_distances = all_distances
        for j in range(0, len(divisor), 1):
            if divisor[j] > 0:
                normalized_distances[:, j] = normalized_distances[:, j] / divisor[j]

        return normalized_distances[-1, :]

    @mp.profile
    def rejection_sampling(self, num_samples):
        """
        * overrides rejection_sampling of ABC class *
        Perform ABC inference with dynamic summary statistic selection using MABs.
        :return:
        posterior: The posterior distribution (samples)
        distances: Accepted distance values
        accepted_count: Number of accepted samples
        trial_count: The number of total trials performed in order to converge
        """
        accepted_count = 0
        trial_count = 0
        accepted_samples = []
        distances = []
        fixed_dataset = DataSet('Fixed Data')
        sim_dataset = DataSet('Simulated Data')
        fixed_dataset.add_points(targets=self.data, summary_stats=self.summaries_function.compute(self.data))

        while accepted_count < num_samples:
            # Rejection sampling
            # Draw from the prior
            trial_param = self.prior_function.draw()

            # Perform the trial
            sim_result = self.sim(trial_param)

            # Get the statistic(s)
            # In case of multiple summaries, a numpy array of k summaries should be returned
            # ToDo: add exception handling to enforce it
            sim_stats = self.summaries_function.compute(sim_result)

            # Set/Update simulated dataset
            sim_dataset.add_points(targets=sim_result, summary_stats=sim_stats)

            # Calculate the distance between the dataset and the simulated result
            # In case of multiple summaries, a numpy array of k distances should be returned
            sim_dist = self.distance_function.compute(fixed_dataset.s, sim_stats)

            # Normalize distances between [0,1]
            sim_dist_scaled = self.scale_distance(sim_dist)

            # Use MAB arm selection to identify the best 'k' arms or summary statistics
            num_arms = len(sim_dist_scaled)
            arms = range(num_arms)
            top_k_arms_idx = self.mab_variant.select(arms, self.k)
            top_k_distances = np.asarray([sim_dist_scaled[i] for i in top_k_arms_idx])

            # Take the norm to combine the top k distances
            combined_distance = np.linalg.norm(top_k_distances)
            logger.debug("Rejection Sampling: trial parameter = [{0}], distance = [{1}]".format(trial_param,
                                                                                                combined_distance))

            # Accept/Reject
            if combined_distance <= self.epsilon:
                accepted_samples.append(trial_param)
                distances.append(sim_dist)
                accepted_count += 1
                logger.info("Rejection Sampling: accepted a new sample, total accepted samples = {0}".
                            format(len(accepted_samples)))

            trial_count += 1

        self.results = {'accepted_samples': accepted_samples, 'distances': distances, 'accepted_count': accepted_count,
                        'trial_count': trial_count, 'inferred_parameters': np.mean(accepted_samples, axis=0)}
        return self.results

