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
The Euclidean distance function
"""

# Imports
from sciope.utilities.distancefunctions.distance_base import DistanceBase
from sciope.utilities.housekeeping import sciope_logger as ml
from dask import delayed
import numpy as np


# Class definition: Euclidean distance function
class EuclideanDistance(DistanceBase):
    """
    Calculates Euclidean distance between two given datasets

    * DistanceBase.compute()
    """

    def __init__(self, use_logger=False):
        """
        We just set the name here and call the superclass constructor.
        """
        self.name = 'Euclidean'
        super(EuclideanDistance, self).__init__(self.name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("EuclideanDistance distance function initialized")

    
    def compute(self, data, sim):
        """
        Summary
        
        Parameters
        ----------
        data : [type]
            [description]
        sim : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        # Make sure we have numpy arrays
        data = np.asarray(data)
        sim = np.asarray(sim)

        # Check that we have equal shapes
        np.testing.assert_equal(sim.shape, data.shape, "Please validate the values and ensure shape equality of the "
                                                       "arguments.")

        res = np.linalg.norm(data - sim)

        if self.use_logger:
            self.logger.info("EuclideanDistance: processed data matrices of shape {0} and calculated distance"
                             " of {1}".format(data.shape, res))
        return res
