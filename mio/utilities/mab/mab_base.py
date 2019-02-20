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
Multi-Armed Bandit Solution Base Class
"""

# Imports
from abc import ABCMeta, abstractmethod


# Class definition
class MABBase(object):
    """
    Base class for MAB solutions like epsilon-first, DIRECT, Incremental, etc.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, arm_pull):
        """
        Set up local variables,
        :param name: should signify a flavour of MAB solution type
        :param arm_pull: function handle returning distance between fixed and simulated data for a sample from prior.
        The prior should be in the same vein as in ABC and should be used within the definition of arm_pull
        """
        self.name = name
        self.arm_pull = arm_pull
        # ToDo: implement assertions here
        self.num_pulls = 0

    @abstractmethod
    def select(self, arms, k=1):
        """
        Select best 'k' arms (summary stats) from n=len(arms) total arms.
        :param arms: array of arm labels/indices
        :param k: scalar value; number of desired stats to be selected
        :return: k indices of selected arms
        """
