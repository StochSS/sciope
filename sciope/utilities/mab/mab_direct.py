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
Multi-Armed Bandits: Direct Algorithm Solution Class
"""

# Imports
from sciope.utilities.mab.mab_base import MABBase
from sciope.utilities.housekeeping import sciope_logger as ml
import numpy as np
import math


# Class definition: DIRECT MAB arm selection
class MABDirect(MABBase):
    """
    The DIRECT algorithm pulls each arm a fixed number of times such that with high probability (1-delta), the k
    selected arms with highest empirical averages are all (epsilon, k)-optimal.
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
        self.name = 'DIRECT'
        self.epsilon = epsilon  # defaults to 1
        self.delta = delta  # starting probability of 0.6 seems loose enough
        super(MABDirect, self).__init__(self.name, arm_pull, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("DIRECT MAB summary statistic selection initialized")

    def select(self, arms, k=1):
        """
        Select k arms using the DIRECT algorithm.
        :param arms: array of arm labels/indices
        :param k: number of arms to select
        :return: k selected arms
        """
        print("select starts")
        n = len(arms)
        num_pulls = int(np.ceil((2 / (self.epsilon ** 2)) * math.log(n / self.delta)))
        print("num_pulls: ", num_pulls)
        rewards = np.empty([num_pulls, n])

        # pull arm 'a' 'num_pulls' times
        for a in range(0, n):
            for p in range(0, num_pulls):
                rewards[p, a] = self.arm_pull(arms[a])
                self.num_pulls += 1

        print("rewards: ", rewards)
        # find the 'k' sized subset with the highest estimated mean reward
        mean_rewards = np.nanmean(rewards, axis=0)

        # replace nan with inf
        mean_rewards[np.isnan(mean_rewards)] = -1 * np.inf
        if self.use_logger:
            self.logger.debug("MABDirect: reward values are {}".format(mean_rewards))

        top_k_arms = np.argpartition(mean_rewards, -k)[-k:]
        if self.use_logger:
            self.logger.debug("MABDirect: selected top {} arm(s) with distances {}".format(k, mean_rewards[top_k_arms]))
        return top_k_arms
