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
Epsilon Selector base class
"""

from abc import ABCMeta, abstractmethod


class EpsilonSelector(object):
    """
    Base class for creating epsilon sequences for use in ABC-SMC.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_initial_epsilon(self):
        """
        Returns
        -------
        epsilon : float
            The initial epsilon value for ABC-SMC
        percentile : bool
            Whether the epsilon should be interpreted as a percentile
        has_more : bool
            Whether there are more epsilons after this one
        """

        pass

    @abstractmethod
    def get_epsilon(self, round, abc_history):
        """
        Parameters
        ----------
        round : int
            The round number
        abc_results :
            A list of dictionaries with keys `accepted_samples` and `distances`
            representing the history of all ABC runs up to this point

        Returns
        -------
        epsilon : float
            The epsilon value for ABC-SMC
        percentile : bool
            Whether the epsilon should be interpreted as a percentile
        terminate : bool
            Whether to stop after this epsilon
        """
        pass
