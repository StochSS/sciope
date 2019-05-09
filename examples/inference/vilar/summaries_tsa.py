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
This summary statistic returns an ensemble of summaries calculated using time series analysis / tsfresh
"""

# Imports
from sciope.utilities.summarystats.summary_base import SummaryBase
import numpy as np
from sciope.features.feature_extraction import generate_tsfresh_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters


# Class definition: SummariesEnsemble
class SummariesTSFRESH(SummaryBase):
    """
    An ensemble of different statistics from TSFRESH
    """

    def __init__(self):
        self.name = 'SummariesTSFRESH'
        self.features = None
        super(SummariesTSFRESH, self).__init__(self.name)

    def compute(self, data, features=EfficientFCParameters()):
        self.features = features
        feature_values = generate_tsfresh_features(data, features)
        return feature_values.reshape(1, feature_values.size)
