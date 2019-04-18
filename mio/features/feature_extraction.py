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
from toolz import partition_all


def generate_tsfresh_features(data, features=None, dask_client=None, chunk_size=10):
    """Method to generate time series features
        input: 
            data -  numpy array of shape 2D  N x T, where T is number of time points
            features - dict containing tsfresh features
                       
                       for exempel:  features = {'variance': None,
                                                'absolute_sum_of_changes': None,
                                                'agg_autocorrelation': [{'f_agg': 'mean'},
                                                                        {'f_agg': 'var'}]}
        return: numpy array of shape N x (Nr of total features)
        """

    for key in features.keys():
        assert hasattr(feature_calculators, key), "%s does not exist as a feature" % key

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

    if dask_client is None:
        return np.array(_wrapper(data))
    else:
        
        #data_chunks = partition_all(chunk_size, data)
        futures = dask_client.map(_wrapper, data)
        return futures


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
