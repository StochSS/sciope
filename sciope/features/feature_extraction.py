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


# Work in progress, no test case coverage for now
def generate_tsfresh_features(data, features):  # pragma: no cover
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
        total = []
        for point in data:
            total.append([list(_f(species)) for species in point])
        return total

    res = np.array(_wrapper(data))
    return res.reshape(-1, res.shape[1] * res.shape[2])


def _get_tsfresh_features_names(features):
    """
    Generates time series features names from TSFRESH library

    Parameters
    ----------
    features : dict, optional
        dict containing tsfresh features
        for exempel:  features = {'variance': None,
                                                'absolute_sum_of_changes': None,
                                                'agg_autocorrelation': [{'f_agg': 'mean'},
                                                                        {'f_agg': 'var'}]}
        
    
    Returns
    -------
    list
        shape: Nr of total features
    """
    f_names = []

    for key in features.keys():
        assert hasattr(feature_calculators, key), "%s does not exist as a feature supported by tsfresh" % key

    for function_name, parameter_list in features.items():
        func = getattr(feature_calculators, function_name)
        if parameter_list:
            for param in parameter_list:
                f_names.append(function_name + str(param))
        else:
            f_names.append(function_name)

    return f_names


def remove_nan_features(x, features):  # pragma: no cover
    """
    Method to remove features containing NaN values.

    Parameters
    ----------
    x: vector/array-like
        array of calculated features
    features : dict
        list of calculated features

    Returns
    -------
    tuple
        non-nan features dict, and feature values
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
