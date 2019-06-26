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
Multi-Armed Bandits: Halving Algorithm Solution Class
"""

# Imports
from sciope.utilities.mab.mab_base import MABBase
from sciope.utilities.housekeeping import sciope_logger as ml
import numpy as np
import math


# Class definition: HALVING MAB arm selection
class MABHalving(MABBase):
    """
    Halving improves over the incremental method by eliminating half of the arms from round l to the round l+1. Arms are
    sampled enough times in each round to ensure that at least k(epsilon, k)-optimal arms are likely to survive each
    round.
    Ref: "Efficient Selection of Multiple Bandit Arms: Theory and Practice", Shivaram Kalyanakrishnan and Peter Stone
    """

    def __init__(self, arm_pull, epsilon=1, delta=0.4, use_logger=False):
        """
        Set up local variables,
        :param arm_pull: function handle returning distance between fixed and simulated data for a sample from prior.
        The prior function must be used within the definition of arm_pull
        :param epsilon: algorithm-specific (optimality) constant
        :param delta: algorithm-specific constant (controls solution confidence)
        """
        self.name = 'HALVING'
        self.epsilon = epsilon  # defaults to 1
        self.delta = delta  # starting probability of 0.6 seems loose enough
        super(MABHalving, self).__init__(self.name, arm_pull, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Halving MAB summary statistic selection initialized")

    def select(self, arms, k=1):
        """
        Select k arms using the HALVING algorithm.
        :param arms: array of arm labels/indices
        :param k: number of arms to select
        :return: k selected arms
        """
        r = arms
        n = len(arms)
        epsilon_i = self.epsilon / 4
        delta_i = self.delta / 2
        num_rounds = int(np.ceil(math.log(n / k, 2)))

        for l in range(0, num_rounds):
            num_pulls = int(np.ceil((2 / (epsilon_i ** 2)) * math.log(3 * k / delta_i)))
            rewards = np.empty([num_pulls, len(r)])
            for a in range(0, len(r)):
                for p in range(0, num_pulls):
                    rewards[p, a] = self.arm_pull(arms[a])
                    self.num_pulls += 1

            mean_rewards = np.nanmean(rewards, axis=0)

            # replace nan with inf
            mean_rewards[np.isnan(mean_rewards)] = -1 * np.inf
            if self.use_logger:
                self.logger.debug("MABDirect: reward values are {}".format(mean_rewards))

            halving_point = int(min(np.ceil(len(r) / 2), k))
            print("halving_point: ", halving_point)
            top_half_arms = np.argpartition(mean_rewards, -halving_point)[-halving_point:]
            r = top_half_arms.tolist()
            epsilon_i = 3 / 4 * epsilon_i
            delta_i = delta_i / 2

        if self.use_logger:
            self.logger.debug("MABHalving: selected top {} arm(s) with distances {}".format(k, mean_rewards[r]))
        return r
