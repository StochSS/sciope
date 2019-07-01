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
Random Sampling Initial Design
"""

# Imports
from sciope.designs.initial_design_base import InitialDesignBase
from sciope.utilities.housekeeping import sciope_logger as ml
import numpy as np
from dask import delayed


# Class definition
class RandomSampling(InitialDesignBase):
    """
    Random Sampling implemented through gpflowopt

    Properties/variables:
    * name						(RandomSampling)
    * xmin						(lower bound of multi-dimensional space encompassing generated points)
    * xmax						(upper bound of multi-dimensional space encompassing generated points)
    * logger                    (a logging object to display/save events - set by derived classes)
    * use_logger     			(a boolean variable controlling whether logging is enabled or disabled)


    Methods:
    * generate					(returns the generated samples)
    """

    def __init__(self, xmin, xmax, use_logger=False):
        """[summary]
        
        Parameters
        ----------
        xmin : vector or 1D array
            Specifies the lower bound of the hypercube within which the design is generated
        xmax : vector or 1D array
            Specifies the upper bound of the hypercube within which the design is generated
        use_logger : bool, optional
            controls whether logging is enabled or disabled, by default False
        """
        name = 'RandomSampling'
        super(RandomSampling, self).__init__(name, xmin, xmax, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Random design in {0} dimensions initialized".format(len(self.xmin)))

    @delayed
    def generate(self, n):
        """
        Sub-classable method for generating 'n' points in the given 'domain'.
        
        Parameters
        ----------
        n : integer
            number of desired points
        
        Returns
        -------
        dask.delayed
        """
        num_variables = len(self.xmin)

        # Generate in [0,1] space
        x = np.random.rand(n, num_variables)

        # Scale from [0,1] to [self.xmin, self.xmax]
        x_scaled = self.scale_to_new_domain(x, self.xmin, self.xmax)
        if self.use_logger:
            self.logger.info("Random design: generated {0} points in {1} dimensions".format(n, num_variables))
        return x_scaled
