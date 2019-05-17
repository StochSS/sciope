# Copyright 2017  Fredrik Wrede, Prashant Singh, and Andreas Hellander
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
Feature Extraction
"""
# Import
from tsfresh.feature_extraction import feature_calculators
import numpy as np


def generate_tsfresh_features(data, features):
    """
    Generates time series features from TSFRESH library

    Parameters
    ----------
    data : ndarray
        time series collections of shape N x T, where T is number of time points
    features : dict, optional
        dict containing tsfresh features
        for exempel:  features = {'variance': None,
                                                'absolute_sum_of_changes': None,
                                                'agg_autocorrelation': [{'f_agg': 'mean'},
                                                                        {'f_agg': 'var'}]}
        
        See TSFRESH documentation for full set of supported features:
        https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html
    
    Returns
    -------
    ndarray
        shape N x (Nr of total features)
    """
    for key in features.keys():
        assert hasattr(feature_calculators, key), "%s does not exist as a feature supported by tsfresh" % key

    def _f(x):
        for function_name, parameter_list in features.items():
            func = getattr(feature_calculators, function_name)

            if func.fctype == "combiner":
                res = func(x, param=parameter_list)  ## returns a list of tuples with string and value
                for item in res:
                    yield item[1]
                
            else:
                if parameter_list:
                    res = [func(x, **param) for param in parameter_list]
                    for item in res:
                        yield item
                else:
                    res = func(x)
                    yield res

    def _wrapper(data):
            return [list(_f(x)) for x in data]

    return np.array(_wrapper(data))



def remove_nan_features(x, features):
    """
    Method to remove features containing NaN values.
    :param x: numpy array of calculated features
    :param features: list of calculated features
    :return: non-nan features dict, and feature values
    """
    nan_idx = np.any(np.isnan(x), axis=0)
    key_idx = 0
    non_nan_features = dict()
    non_nan_feature_values = x[:, ~nan_idx]

    for key, value in features.items():
        if not nan_idx[key_idx]:
            # Not a nan feature
            non_nan_features[key] = value
        key_idx += 1

    return non_nan_feature_values, non_nan_features
