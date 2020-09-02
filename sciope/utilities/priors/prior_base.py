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
Base Class for Prior Functions
"""

# Imports
from abc import ABCMeta, abstractmethod


# Class definition
class PriorBase(object):
    """
    Base class for prior functions used by ABC inference algorithms and MAB-based statistic selection.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, use_logger=False):
        """
        Set up local variables
        :param name: unique identifier for the prior
        """
        self.name = name
        self.use_logger = use_logger

    @abstractmethod
    def pdf(self, x, log=False):
        """
        Evaluate the PDF of the sample
        :param x: the point or collection of points to evaluate the pdf at
        :param log: whether to return the log pdf
        :return: the pdf evaluated at x
        """

    @abstractmethod
    def draw(self, n=1):
        """
        Draw 'n' samples from the prior
        :param n: number of desired samples from prior; defaults to 1
        :return: the 'n' drawn samples as a vector
        """

    @abstractmethod
    def get_dimension(self):
        """
        Get the dimension of the prior.
        """
