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
from sciope.utilities.summarystats.summary_base import SummaryBase
import numpy as np
from sciope.utilities.summarystats import burstiness as bs
from sciope.utilities.summarystats import global_max as mx
from sciope.utilities.summarystats import global_min as mn
from sciope.utilities.summarystats import temporal_mean as tm
from sciope.utilities.summarystats import temporal_variance as tv


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
        bs_vals = bs.Burstiness(mean_trajectories=True).compute(data).compute()
        mx_vals = mx.GlobalMax(mean_trajectories=True).compute(data).compute()
        mn_vals = mn.GlobalMin(mean_trajectories=True).compute(data).compute()
        tm_vals = tm.TemporalMean(mean_trajectories=True).compute(data).compute()
        tv_vals = tv.TemporalVariance(mean_trajectories=True).compute(data).compute()
        ensemble.append(bs_vals)
        ensemble.append(mx_vals)
        ensemble.append(mn_vals)
        ensemble.append(tm_vals)
        ensemble.append(tv_vals)
        ensemble = np.asarray(ensemble)
        return ensemble.reshape(1, ensemble.size)
