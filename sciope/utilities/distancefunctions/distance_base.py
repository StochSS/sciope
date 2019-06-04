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
Distance functions base class
"""

# Imports
from abc import ABCMeta, abstractmethod


# Class definition
class DistanceBase(object):
    """
    Base class for creating distance functions used by parameter inference algorithms.
    Must not be used directly!
    Each distance function type must implement the methods described herein:

    * DistanceBase.compute()
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, use_logger=False):
        """[summary]
        
        Parameters
        ----------
        name : [type]
            [description]
        use_logger : bool, optional
            [description], by default True
        """
        self.name = name
        self.use_logger = use_logger

    @abstractmethod
    def compute(self, data, sim):
        """
        Sub-classable method for calculating distances between fixed and simulated data.
        Each derived class must implement.

        Parameters
        ----------
        data : [type]
            [description]
        sim : [type]
            [description]
        """
