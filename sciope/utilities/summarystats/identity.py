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
The 'Identity' summary statistic
"""

# Imports
import numpy as np
import math as mt
from sciope.utilities.summarystats.summary_base import SummaryBase
from sciope.utilities.housekeeping import sciope_logger as ml

class Identity(SummaryBase):

    def __init__(self, normalization = None, mean_trajectories = False, use_logger = False):

        self.name = 'Identity'
        self.normalization = normalization

        super(Identity, self).__init__(self.name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Identity summary statistic initialized")

    def compute(self, data):
        
        if self.normalization is not None:
            return self.normalization(data)
        return data
