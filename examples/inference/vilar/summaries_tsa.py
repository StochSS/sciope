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
import dask
from sciope.features.feature_extraction import generate_tsfresh_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
import numpy as np


# Summary statistics from TSFRESH
class SummariesTSFRESH(SummaryBase):
    """
    Class for computing features/statistics on time series data.
    An ensemble of different statistics from TSFRESH are supported.
    """

    def __init__(self):
        self.name = 'SummariesTSFRESH'
        self.features = EfficientFCParameters() # MinimalFCParameters()
        self.features.pop('length')
        super(SummariesTSFRESH, self).__init__(self.name)

    @dask.delayed
    def compute(self, point):
        """
        Computes features for one point (time series).

        Parameters
        ---------

        point : numpy.ndarray of shape n_timepoints x 1

        Returns
        params : list of features

        """
        # f = MinimalFCParameters()
        # f.pop('length')
        #return list(generate_tsfresh_features(data=point, features=self.features))
        return np.asarray(generate_tsfresh_features(data=point, features=self.features))
