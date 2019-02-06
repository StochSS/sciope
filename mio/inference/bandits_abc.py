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
from abc import ABC
import numpy as np
from sklearn.preprocessing import scale
from utilities.distancefunctions import euclidean as euc
from utilities.summarystats import burstiness as bs
from utilities.mab import mab_direct as md


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
    return -1 * normalized_distances[arm_idx]


# Class definition: Bandits-ABC rejection sampling
class BanditsABC(ABC):
    """
    ABC rejection sampling with dynamic multi-armed bandit (MAB) assisted summary statistic selection.
    """

    def __init__(self, data, sim, prior_function, mab_variant=md.MABDirect(arm_pull), k=1, epsilon=0.1,
                 parallel_mode=True, summaries_function=bs.Burstiness(), distance_function=euc.EuclideanDistance()):
        self.mab_variant = mab_variant
        super(BanditsABC, self).__init__(data, sim, prior_function, epsilon, parallel_mode, summaries_function,
                                         distance_function)
        self.name = 'BanditsABC'

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
        global normalized_distances
        dataset_stats = self.summaries_function.compute(self.data)

        while accepted_count < num_samples:
            # Rejection sampling
            # Draw from the prior
            trial_param = self.prior_function.draw()

            # Perform the trial
            sim_result = self.sim(trial_param)

            # Get the statistic(s)
            sim_stats = self.summaries_function.compute(sim_result)

            # Calculate the distance between the dataset and the simulated result
            sim_dist = self.distance_function.compute(dataset_stats, sim_stats)

            # Normalize distances between [0,1]
            sim_dist_scaled = scale(sim_dist)
            normalized_distances = sim_dist_scaled

            # Use MAB arm selection to identify the best 'k' arms or summary statistics
            num_arms = len(normalized_distances)
            arms = range(num_arms)
            top_k_distances = self.mab_variant.select(arms, self.k)

            # Take the norm to combine the top k distances
            combined_distance = np.linalg.norm(top_k_distances)

            # Accept/Reject
            if combined_distance <= self.epsilon:
                accepted_samples.append(trial_param)
                distances.append(sim_dist)
                accepted_count += 1

            trial_count += 1

        return accepted_samples, distances, accepted_count, trial_count
