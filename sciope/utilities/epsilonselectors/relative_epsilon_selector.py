# Copyright 2020 Richard Jiang, Prashant Singh, Fredrik Wrede and Andreas Hellander
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
An epsilon selector based on relative percentiles.
"""
from sciope.utilities.epsilonselectors.epsilon_selector import *
import numpy as np


class RelativeEpsilonSelector(EpsilonSelector):
    """
    Creates an epsilon selector based on a relative percentile.  For
    each round, it computes the new epsilon as the epsilon-percentile
    of the last set of ABC distances.
    """

    def __init__(self, epsilon_percentile, max_rounds=None):
        """
        Parameters
        ----------
        epsilon_percentile : float [0, 100]
            The percentile of the distances to use in each round. Specified
            as a percent between 0 and 100.
        max_rounds : int
            The maximum number of rounds before stopping. If None, doesn't end.
        """

        self.epsilon_percentile = epsilon_percentile
        self.max_rounds = max_rounds

    def get_initial_epsilon(self):
        """
        Gets the initial epsilon, interpreted as a percentile
        Returns
        -------
        epsilon : float
            The epsilon percentile, because there is no initial epsilon
            without history
        percentile : bool
            Whether the epsilon should be interpreted as a percentile
        has_more : bool
            Whether there are more epsilons after this one
        """
        return self.epsilon_percentile, True, self.max_rounds == 0

    def get_epsilon(self, round, abc_history):
        """Returns the new epsilon based on the n-th round.

        Parameters
        ----------
        round : int
            the n-th round of the sequence
        abc_history : type
            A list of dictionaries with keys `accepted_samples` and `distances`
            representing the history of all ABC runs up to this point.

        Returns
        -------
        epsilon : float
            The epsilon value for ABC-SMC
        percentile : bool
            Whether the epsilon should be interpreted as a percentile
        terminate : bool
            Whether to stop after this epsilon
        """
        if round > len(abc_history):
            t = np.percentile(abc_history[-1]['distances'], self.epsilon_percentile)
        else:
            t = np.percentile(abc_history[round - 1]['distances'], self.epsilon_percentile)
        return t, False, self.max_rounds and round + 1 == self.max_rounds
