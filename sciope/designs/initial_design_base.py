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

    Properties/variables:
    * name						(FactorialDesign)
    * xmin						(lower bound of multi-dimensional space encompassing generated points)
    * xmax						(upper bound of multi-dimensional space encompassing generated points)
    * logger                    (a logging object to display/save events - set by derived classes)
    * use_logger     			(a boolean variable controlling whether logging is enabled or disabled)


    Methods:
    * generate					(returns the generated samples)
    * scale_variable            (scales a variable from an old domain range to a new domain range)
    * scale_to_new_domain       (scales a matrix from an old domain range to a new domain range)
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, xmin, xmax, use_logger=False):
        """
        Initialize a design with specified parameters

        Parameters
        ----------
        name : string
            Set by the derived class - typically same as the type of design
        xmin : vector or 1D array
            Specifies the lower bound of the hypercube within which the design is generated
        xmax : vector or 1D array
            Specifies the upper bound of the hypercube within which the design is generated
        use_logger : bool, optional
            controls whether logging is enabled or disabled, by default False
        """
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

        Parameters
        ----------
        x : vector
            represents a variable or dimension to operate upon
        old_min : vector
            the old lower bound of the domain
        old_max : vector
            the old upper bound of the domain
        new_min : vector
            the new lower bound of the domain
        new_max : vector
            the new upper bound of the domain

        Returns
        -------
        vector
            scaled to the new range
        """
        if old_max - old_min == 0:
            return new_min      # this is the rare case of a scalar with old range being 0
        else:
            return (((x - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

    @staticmethod
    def scale_to_new_domain(x, new_min, new_max):
        """
        Scales a given array/matrix to a new range

        Parameters
        ----------
        x : multidimensional array/matrix
            to operate upon
        new_min : vector
            the new lower bound of the domain, one element per dimension
        new_max : vector
            the new upper bound of the domain, one element per dimension

        Returns
        -------
        matrix
            scaled to the new range
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
