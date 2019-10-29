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
Base Class for Summary Statistics
"""

# Imports
from abc import ABCMeta, abstractmethod


# Class definition
class SummaryBase(object):
    """
    Base class for summary statistics.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, mean_trajectories=False, use_logger=False):
        """
        Set up local variables
        :param name: unique identifier for the statistic
        :param mean_trajectories: if enabled, it returns the mean statistic value computed over all trajectories
        :param use_logger: if enabled, logs the summary statistic calculation in a file and on screen
        """
        self.name = name
        self.summaries_names = None
        self.mean_trajectories = mean_trajectories
        self.use_logger = use_logger

    def compute(self, data):
        """
        Calculate the summary statistic value for given 'data'
        :param data: a fixed data set or simulation result
        :return: the computed summary statistic value
        """
