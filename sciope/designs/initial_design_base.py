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
Initial Design Base Class
"""

# Imports
from abc import ABCMeta, abstractmethod
import numpy as np


# Class definition
class InitialDesignBase(object):
    """
    Base class for initial designs.
    Must not be used directly!
    Each initial design type must implement the methods described herein:

    * InitialDesignBase.generate(n,domain)
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, xmin, xmax, use_logger=False):
        self.name = name
        np.testing.assert_array_less(xmin, xmax, err_msg=("Please validate the values and ensure shape equality of "
                                                          "domain lower and upper bounds."))
        self.xmin = xmin
        self.xmax = xmax
        self.use_logger = use_logger

    @abstractmethod
    def generate(self, n):
        """
        Sub-classable method for generating 'n' points within a given domain. Each derived class must implement.
        """

    @staticmethod
    def scale_variable(x, old_min, old_max, new_min, new_max):
        """
        Scales a dimension from the specified old range to a new range.
        :param x:       vector representing a variable / dimension
        :param old_min: the old lower bound
        :param old_max: the old upper bound
        :param new_min: the new lower bound
        :param new_max: the new upper bound
        :return:        a vector scaled to the new range
        """
        if old_max - old_min == 0:
            return new_min      # this is the rare case of a scalar with old range being 0
        else:
            return (((x - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

    @staticmethod
    def scale_to_new_domain(x, new_min, new_max):
        """
        Scales a given array/matrix to a new range
        :param x:       given data matrix
        :param new_min: new min (np array - one element per dimension)
        :param new_max: new max (np array - one element per dimension)
        :return:        scaled x
        """
        old_min = np.min(x, axis=0)         # this gives min across each dimension
        old_max = np.max(x, axis=0)         # and max...
        idx = 0
        for column in x.T:
            x[:, idx] = InitialDesignBase.scale_variable(column, old_min[idx], old_max[idx], new_min[idx], new_max[idx])
            idx += 1

        return x
        # # We will use np.iter, so pre-process min/max in the format required by np.iter
        # l_to = list(map(list, zip(new_min, new_max)))
        # to_range = list(chain.from_iterable(l_to))
        #
        # old_min = np.min(x, axis=0)
        # old_max = np.max(x, axis=0)
        # l_from = list(map(list, zip(old_min, old_max)))
        # from_range = list(chain.from_iterable(l_from))
        #
        # return np.interp(x, from_range, to_range)
