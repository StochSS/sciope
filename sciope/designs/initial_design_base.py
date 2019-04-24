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

    def __init__(self, name, xmin, xmax, use_logger=True):
        self.name = name
        np.testing.assert_array_less(xmin, xmax, err_msg=("Please validate the values and ensure shape equality of "
                                                          "domain lower and upper bounds."))
        self.xmin = xmin
        self.xmax = xmax
        self.use_logger= use_logger

    @abstractmethod
    def generate(self, n):
        """
        Sub-classable method for generating 'n' points within a given domain. Each derived class must implement.
        """
