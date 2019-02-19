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
This summary statistic returns an ensemble of summaries calculated using existing statistics
"""

# Imports
import sys
sys.path.append("../../mio/utilities/")
sys.path.append("../../mio/utilities/summarystats")
from summary_base import SummaryBase
import numpy as np
from utilities.summarystats import burstiness as bs
from utilities.summarystats import global_max as mx
from utilities.summarystats import global_min as mn
from utilities.summarystats import temporal_mean as tm
from utilities.summarystats import temporal_variance as tv


# Class definition: SummariesEnsemble
class SummariesEnsemble(SummaryBase):
    """
    An ensemble of different statistics
    """

    def __init__(self):
        self.name = 'SummariesEnsemble'
        super(SummariesEnsemble, self).__init__(self.name)

    @staticmethod
    def compute(data):
        ensemble = []
        ensemble.append(bs.Burstiness(mean_trajectories=False).compute(data))
        ensemble.append(mx.GlobalMax().compute(data))
        ensemble.append(mn.GlobalMin().compute(data))
        ensemble.append(tm.TemporalMean().compute(data))
        ensemble.append(tv.TemporalVariance().compute(data))
        return np.asarray(ensemble).ravel()
