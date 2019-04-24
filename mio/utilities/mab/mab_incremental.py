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
Multi-Armed Bandits: Incremental Algorithm Solution Class
"""

# Imports
from sciope.utilities.mab.mab_base import MABBase
from sciope.utilities.housekeeping import sciope_logger as ml
import numpy as np
import math

# Set up the logger and profiler
logger = ml.MIOLogger().get_logger()


# Class definition: HALVING MAB arm selection
class MABIncremental(MABBase):
    """
    The incremental algorithm consists of 'k' rounds. In a given round l, S_l is the set of selected arms, while R_l is
    the set of arms remaining to be processed. During round l, an (epsilon,1) optimal arm in R_l is selected with high
    probability using the median elimination algorithm (Even-Dar et. al. 2006)
    Ref: "Efficient Selection of Multiple Bandit Arms: Theory and Practice", Shivaram Kalyanakrishnan and Peter Stone
    """

    def __init__(self, arm_pull, epsilon=1, delta=0.4):
        """
        Set up local variables,
        :param arm_pull: function handle returning distance between fixed and simulated data for a sample from prior.
        The prior function must be used within the definition of arm_pull
        :param epsilon: algorithm-specific (optimality) constant
        :param delta: algorithm-specific constant (controls solution confidence)
        """
        self.name = 'INCREMENTAL'
        self.epsilon = epsilon  # defaults to 1
        self.delta = delta  # starting probability of 0.6 seems loose enough
        super(MABIncremental, self).__init__(self.name, arm_pull)

    def select(self, arms, k=1):
        """
        Select k arms using the INCREMENTAL algorithm.
        :param arms: array of arm labels/indices
        :param k: number of arms to select
        :return: k selected arms
        """
        s = []
        r = arms

        for l in range(0, k):
            # Find arm 'a' using median elimination: MEDIAN-ELIMINATION(R, epsilon, delta/k)
            epsilon_i = self.epsilon / 4
            delta_i = self.delta / (2 * k)
            r_i = r
            while len(r_i) >= 2:
                num_pulls = int((1 / ((epsilon_i / 2) ** 2)) * math.log(3 / delta_i))
                rewards = np.empty([num_pulls, len(r_i)])
                for a in range(0, len(r_i)):
                    for p in range(0, num_pulls):
                        rewards[p, a] = self.arm_pull(r_i[a])
                        self.num_pulls += 1

                median_value = np.median(rewards)
                mean_rewards = np.nanmean(rewards, axis=0)

                # replace nan with inf
                mean_rewards[np.isnan(mean_rewards)] = -1 * np.inf
                logger.debug("MABDirect: reward values are {}".format(mean_rewards))

                # remove all arms with mean rewards < median_value
                r_new = []
                for i in range(0, len(r_i)):
                    if mean_rewards[i] >= median_value:
                        r_new.append(r_i[i])

                r_i = r_new
                epsilon_i = epsilon_i / 4 * 3.0
                delta_i = delta_i / 2

            s.append(r_i[0])
            r.remove(r_i[0])

        logger.debug("MABDirect: selected top {} arm(s) with distances {}".format(k, mean_rewards[s]))
        return s
