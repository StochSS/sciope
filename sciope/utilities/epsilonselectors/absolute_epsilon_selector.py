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
Absolute epsilon selector
"""

from sciope.utilities.epsilonselectors.epsilon_selector import *


class AbsoluteEpsilonSelector(EpsilonSelector):
    """
    Creates an epsilon selector based on a fixed sequence.
    """

    def __init__(self, epsilon_sequence):
        """

        Parameters
        ----------
        epsilon_sequence : Seqeunce[float]
            Sequence of epsilons to use.
        """

        assert (len(epsilon_sequence) > 0)
        self.epsilon_sequence = epsilon_sequence
        self.last_round = len(self.epsilon_sequence) - 1

    def get_initial_epsilon(self):
        """Gets the first epsilon in the sequence.

        Returns
        -------
        epsilon : float
            The initial epsilon value of this sequence
        percentile : bool
            Whether the epsilon should be interpreted as a percentile
        has_more : bool
            Whether there are more epsilons after this one
        """
        return self.epsilon_sequence[0], False, len(self.epsilon_sequence) == 1

    def get_epsilon(self, round, abc_history):
        """Returns the n-th epsilon in the seqeunce.

        Parameters
        ----------
        round : int
            the round to get the epsilon for
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
        return self.epsilon_sequence[round], False, round == last_round
