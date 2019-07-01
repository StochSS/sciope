# Copyright 2017 Prashant Singh, Andreas Hellander and Fredrik Wrede
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
Parameter Inference Base Class
"""

# Imports
from abc import ABCMeta, abstractmethod


# Class definition
class InferenceBase(object):
    """
    Base class for parameter inference algorithms.
    Must not be used directly!
    Each inference algorithm must implement the methods described herein:

    * InferenceBase.infer()
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, data, sim, use_logger=False):
        """
        Base class initializer
        
        Parameters
        ----------
        name : string
            Inference algorithm name; set by the derived class
        data : nd-array
            Observed data or fixed data
        sim : function handle
            The simulator
        use_logger : bool
            Flag controlling enabling / disabling of logging
        """
        self.name = name
        # ToDo: implement assertions here
        self.data = data
        self.sim = sim
        self.use_logger = use_logger
        self.results = dict()

    @abstractmethod
    def infer(self):
        """
        Sub-classable method for performing parameter inference. Each derived class must implement.
        """
