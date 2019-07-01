# Copyright 2019 Prashant Singh, Fredrik Wrede and Andreas Hellander
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
The naive squared function: (a-b ** 2)
"""

# Imports
from sciope.utilities.distancefunctions.distance_base import DistanceBase
from sciope.utilities.housekeeping import sciope_logger as ml
import numpy as np
from dask import delayed


# Class definition: NaiveSquared distance function
class NaiveSquaredDistance(DistanceBase):
    """
    Calculates squared element-wise distance between two given datasets

    * DistanceBase.compute()
    """

    def __init__(self, use_logger=False):
        """
        We just set the name here and call the superclass constructor.
        """
        self.name = 'NaiveSquared'
        super(NaiveSquaredDistance, self).__init__(self.name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("NaiveSquaredDistance distance function initialized")

    
    def compute(self, data, sim):
        """[summary]
        
        Parameters
        ----------
        data : [type]
            [description]
        sim : [type]
            [description]
        
        Returns
        -------
        [type]
            the squared element-wise distance
        """
        # Make sure we have numpy arrays
        data = np.asarray(data)
        sim = np.asarray(sim)

        # Check that we have equal shapes
        np.testing.assert_equal(sim.shape, data.shape, "Please validate the values and ensure shape equality of the "
                                                       "arguments.")

        res = (data - sim) ** 2

        if self.use_logger:
            self.logger.info("NaiveSquaredDistance: processed data matrices of shape {0} and calculated distance"
                             " of {1}".format(data.shape, res))
        return res
