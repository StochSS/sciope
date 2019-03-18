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
from mio.designs.initial_design_base import InitialDesignBase
from mio.utilities.housekeeping import mio_logger as ml
import gpflowopt

# Set up the logger
def get_logger():
    return ml.MIOLogger().get_logger()


# Class definition
class RandomSampling(InitialDesignBase):
    """
    Random Sampling implemented through gpflowopt

    * InitialDesignBase.generate(n)
    """

    def __init__(self, xmin, xmax, use_logger=True):
        name = 'RandomSampling'
        super(RandomSampling, self).__init__(name, xmin, xmax, use_logger)
        if self.use_logger:
            self.logger = get_logger()
            self.logger.info("Random design in {0} dimensions initialized".format(len(self.xmin)))

    def generate(self, n):
        """
        Sub-classable method for generating 'n' points in the given 'domain'.
        """
        num_variables = len(self.xmin)
        gpf_domain = gpflowopt.domain.ContinuousParameter('x0', self.xmin[0], self.xmax[0])
        for i in range(1, num_variables):
            var_name = 'x' + repr(i)
            gpf_domain = gpf_domain + gpflowopt.domain.ContinuousParameter(var_name, self.xmin[i], self.xmax[i])

        design = gpflowopt.design.RandomDesign(n, gpf_domain)
        if self.use_logger:
            self.logger.info("Random design: generated {0} points in {1} dimensions".format(n, num_variables))
        return design.generate()
