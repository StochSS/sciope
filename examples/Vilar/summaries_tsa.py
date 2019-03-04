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
import sys
sys.path.append("../../mio/utilities/")
sys.path.append("../../mio/utilities/summarystats")
sys.path.append("../../mio/features")
from summary_base import SummaryBase
import numpy as np
import tsfresh as tsf
import time_series_analysis as tsa


# Class definition: SummariesEnsemble
class SummariesTSFRESH(SummaryBase):
    """
    An ensemble of different statistics
    """

    def __init__(self):
        self.name = 'SummariesTSFRESH'
        super(SummariesTSFRESH, self).__init__(self.name)

    @staticmethod
    def compute(data):
        fe = tsa.TimeSeriesAnalysis(name="Vilar", columns=['A'])
        if len(data.shape) == 3:
            # Extract the time series for specie A
            data_reshaped = data[:, :, [0, -2]]
        else:
            data_reshaped = data.reshape(data.shape[0], data.shape[1], 1)
        fe.put(data_reshaped)
        df = fe.get_data()
        features = tsf.extract_features(df, column_id="index", column_sort="time")
        effective_features = features.dropna(axis=1, how='any')
        return np.asarray(effective_features).ravel()
