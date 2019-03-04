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


def generate_tsfresh_features(X, features=None):
    """Method to generate time series features
        input: 
            X -  2D numpy array (Nr trajectories x Nr time points) 
            features - dict containing tsfresh features
                       
                       for exempel:  features = {'variance': None,
                                                'absolute_sum_of_changes': None,
                                                'agg_autocorrelation': [{'f_agg': 'mean'},
                                                                        {'f_agg': 'var'}]}
        return: numpy array of shape Nr of total features
        """

    assert type(features) == dict "features has to be of type dict"
    for key in features.keys():
    assert hasattr(feature_calculators, key) "%s does not exist as a feature" % key

    def _f():
        for function_name, parameter_list in features.items():
            func = getattr(feature_calculators, function_name)

            if func.fctype == "combiner":
                res = func(X, param=parameter_list)  ## returns a list of tuples with string and value
                for item in res:
                    yield item[1]
                
            else:
                if parameter_list:
                    result = ((convert_to_output_format(param), func(X, **param)) for param in parameter_list) # TODO: convert to array
                else:
                    result = func(X)
            yield result
    return np.array(_f())
        
        
