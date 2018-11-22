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
The Manhattan distance function
"""

# Imports
from distance_base import DistanceBase
from scipy.spatial.distance import cdist


# Class definition: Manhattan distance function
class ManhattanDistance(DistanceBase):
    """
    Calculates Manhattan distance between two given datasets

    * DistanceBase.compute()
    """

    def __init__(self):
        """
        We just set the name here and call the superclass constructor.
        """
        self.name = 'Manhattan'
        super(ManhattanDistance, self).__init__(self.name)

    @staticmethod
    def compute(data, sim):
        """
        The arguments should either be provided with the function call or during instantiation.
        :param data: as in init
        :param sim: as in init
        :return: the distance calculated using numpy (found to be more efficient than scipy.distance)
        """
        return cdist(data, sim, metric='cityblock')
