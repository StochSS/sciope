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
from sklearn.preprocessing import scale


# Class definition: Bandits-ABC rejection sampling
class BanditsABC(ABC):
    """
    ABC rejection sampling with dynamic multi-armed bandit (MAB) assisted summary statistic selection
    """

    def __init__(self, data, sim, epsilon=0.1, parallel_mode=False, summaries_function,
                 distance_function, prior_function, mab_variant):
        self.mab_variant = mab_variant
        super(BanditsABC, self).__init__(data, sim, epsilon, parallel_mode, summaries_function,
                                  distance_function, prior_function)
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
        dataset_stats = self.summaries_function(self.data)

        while accepted_count < num_samples:
            # Rejection sampling
            # Draw from the prior
            trial_param = self.prior_function.draw()

            # Perform the trial
            sim_result = self.sim(trial_param)

            # Get the statistic
            sim_stats = self.summaries_function.compute(sim_result)

            # Calculate the distance between the dataset and the simulated result
            sim_dist = self.distance_function.compute(dataset_stats, sim_stats)

            # Normalize distances between [0,1]
            sim_dist_scaled = scale(sim_dist)

            # ToDo: Put in MAB call here

            # Accept/Reject
            if sim_dist <= self.epsilon:
                accepted_samples.append(trial_param)
                distances.append(sim_dist)
                accepted_count += 1

            trial_count += 1

        return accepted_samples, distances, accepted_count, trial_count