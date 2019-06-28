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
Surrogate Model Base Class
"""

# Imports
from abc import ABCMeta, abstractmethod
import numpy as np


# Class definition
class ModelBase(object):
    """
    Base class for surrogate models.
    Must not be used directly!
    Each model type must implement the methods described herein:

    * ModelBase.train(x,y)
    * ModelBase.predict(xt)

    The following variables are available to derived classes:

    * self.x			(training inputs)
    * self.y			(training targets)
    * self.my			(mean for scaling)
    * self.sy			(std dev for scaling)
    * self.n			(num training points)
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, use_logger=False):
        """
        Initialize the model.

        Parameters
        ----------
        name : string
            Model name; set by the derived class
        use_logger : bool, optional
            Controls whether logging is enabled or disabled, by default False
        """
        self.name = name
        self.model = None
        self.use_logger = use_logger

    def scale_training_data(self, x, y):
        """
        pre-process training data

        x : nd-array
            inputs or independent variables
        y : nd-array
            output or dependent variable
        """
        # Take care of NaNs
        if np.isnan(y).any():
            yr = y[~np.isnan(y)]
            my = np.mean(yr)
            y[np.isnan(y)] = my

        # ...imaginaries
        y = np.real(y)
        self.n = x.shape[0]

        # scale
        self.my = np.mean(y)
        self.sy = np.std(y)
        ry = np.ones((self.n, 1)) * self.my
        y = (y - ry) / self.sy

        # set data
        self.x = x
        self.y = y.ravel()

    @abstractmethod
    def train(self, inputs, targets):
        """
        Sub-classable method for training a given surrogate. Each derived class must implement.
        """

    @abstractmethod
    def predict(self, test_input):
        """
        Sub-classable method for predicting test data. Each derived class must implement.
        """
