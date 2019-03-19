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
Multi-Armed Bandits: Successive Accepts and Rejects (SAR) Algorithm Solution Class
"""

# Imports
from mio.utilities.mab.mab_base import MABBase
from mio.utilities.housekeeping import mio_logger as ml
import numpy as np

# Set up the logger and profiler
logger = ml.MIOLogger().get_logger()


# Class definition: SAR MAB arm selection
class MABSAR(MABBase):
    """
    SAR divides n rounds into p-1 phases. At the end of each phase, the algorithm either accepts the arm with the
    highest empirical mean reward, or rejects the arm with the lowest empirical mean reward. The arm in question is
    deactivated. In the next phase, each active arm is pulled equally often.
    Ref: "Multiple Identifications in Multiple-Armed Bandits", Bubeck, Wang and Vishwanathan (2012)
    """

    def __init__(self, arm_pull, p, b):
        """
        Set up local variables,
        :param arm_pull: function handle returning distance between fixed and simulated data for a sample from prior.
        The prior function must be used within the definition of arm_pull
        :param p: algorithm-specific constant (number of phases)
        :param b: algorithm-specific constant (total pull budget)
        """
        self.name = 'SAR'
        self.p = p
        self.b = b
        super(MABSAR, self).__init__(self.name, arm_pull)

    def select(self, arms, k=1):
        """
        Select k arms using the SAR algorithm.
        :param arms: array of arm labels/indices
        :param k: number of arms to select
        :return: k selected arms
        """
        active_arms = list(arms)
        selected_arms = []
        sum_i = 0
        n_prev = 0

        for i in range(2, self.p):
            sum_i = sum_i + (1 / i)
        log_x = 0.5 + sum_i

        for x in range(0, self.p - 1):
            n_x = int(np.ceil(1 / log_x * (self.b - self.p) / (self.p + 1 - x)))
            num_pulls = n_x - n_prev

            # handle '0' arm pulls
            if num_pulls < 1:
                n_prev = n_x
                continue

            # if we need more arms than valid & available, we return 'last standing' arms
            if k >= len(active_arms):
                selected_arms = selected_arms + active_arms
                break

            rewards = np.empty([num_pulls, len(active_arms)])
            for a in range(0, len(active_arms)):
                for p in range(0, num_pulls):
                    rewards[p, a] = self.arm_pull(arms[a])
                    self.num_pulls += 1

            mean_rewards = np.nanmean(rewards, axis=0)
            logger.debug("MABDirect: reward values are {}".format(mean_rewards))

            # replace nan with inf
            mean_rewards[np.isnan(mean_rewards)] = -1 * np.inf

            sorted_idx = np.argsort(-mean_rewards)
            k_th_best_arm = sorted_idx[k - 1]
            k_star_th_best_arm = sorted_idx[k]
            gaps = np.zeros(len(active_arms))

            # for each active design
            for a in range(0, len(active_arms)):
                if a in sorted_idx[0:k - 1]:
                    gaps[a] = mean_rewards[a] - mean_rewards[k_star_th_best_arm]
                else:
                    gaps[a] = mean_rewards[k_th_best_arm] - mean_rewards[a]

            # find the active arm with the highest gap
            arm_highest_gap = gaps.argmax()

            # if the arm with the largest gap was indeed the best arm then accept it
            if arm_highest_gap == sorted_idx[0]:
                selected_arms.append(active_arms[arm_highest_gap])
                k = k - 1

            # mark 'arm_highest_gap' as inactive; remove it from A
            active_arms.remove(active_arms[arm_highest_gap])

            # check if we have met our goal
            if k == 0:
                break

            # increment ...
            n_prev = n_x

        logger.debug("MABDirect: selected top {} arm(s) with distances {}".format(k, mean_rewards[selected_arms]))
        return selected_arms
