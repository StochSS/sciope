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

from sciope.features.feature_extraction import generate_tsfresh_features, _get_tsfresh_features_names
from sciope.utilities.summarystats.summary_base import SummaryBase
from itertools import combinations
from tsfresh.feature_extraction.settings import EfficientFCParameters, MinimalFCParameters
import numpy as np


class SummariesTSFRESH(SummaryBase):
    """
    Class for computing features/statistics on time series data.
    An ensemble of different statistics from TSFRESH are supported.
    """

    def __init__(self, features='minimal', corrcoef=False, use_logger=False):
        self.name = 'SummariesTSFRESH'
        super(SummariesTSFRESH, self).__init__(self.name, use_logger=use_logger)
        
        if type(features) is str:
            allowed_str = ['minimal', 'full']
            assert features in allowed_str,"{0} is not recognized, supported sets are 'minimal' and 'full'".format(features)
            if features == 'minimal':
                self.features = MinimalFCParameters()
                self.features.pop('length')
            else:
                self.features = EfficientFCParameters()
        else:
            self.features = features
        
        self.corrcoef = corrcoef

        self.summaries_names = _get_tsfresh_features_names(self.features)
        
        

    def _compute_tsfresh(self, point):
        """
        Computes features for one point (time series).
        
        Parameters
        ----------
        point : ndarray
            trajectory of shape n_timepoints x 1

        Returns
        -------
        list
            list of generated features 
        """
        return generate_tsfresh_features(point, features=self.features)

    def _compute_corrcoef(self, x, y):
        """
        Computes the Pearson correlation coefficient between two trajectories
        
        Parameters
        ---------

        x : ndarray 
            Trajectory of shape n_timepoints x 1 

        y: ndarray 
            Trajectory of shape n_timepoints x 1 

        Returns
        list
            list of generated feature
        """
        return [np.corrcoef(x, y)[0, 1]]

    def compute(self, point):
        """[summary]
        
        Parameters
        ----------
        point : [type]
            [description]
        """
        point = np.asarray(point)
        assert len(point.shape) == 3, "required input shape is (n_points, n_species, n_timepoints)" 
        tsfresh_summaries = self._compute_tsfresh(point)
        tsfresh_summaries = np.asarray(tsfresh_summaries)
        tsfresh_summaries = np.mean(tsfresh_summaries, axis=0, keepdims=True)
        if self.corrcoef:
            assert point.shape[1] > 1, "corrcoef = True can only be used if the n_species > 1"
            corrcoef_summaries = []
            n_species = range(point.shape[1])
            for n in point:
                corr = []
                for s in combinations(n_species, 2):
                    x = n[s[0]]
                    y = n[s[1]]
                    corr.append(self._compute_corrcoef(x, y)[0])
                corrcoef_summaries.append(corr)
            corrcoef_summaries = np.asarray(corrcoef_summaries)
            corrcoef_summaries = np.mean(corrcoef_summaries, axis=0, keepdims=True)
            tot = np.hstack((tsfresh_summaries, corrcoef_summaries))
            return tot
        else:
            return tsfresh_summaries


