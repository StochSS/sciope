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
Dataset Class
"""

# Imports
import numpy as np
from collections import OrderedDict
from scipy.stats.mstats import mquantiles
from scipy.stats import zscore


# Class definition
# Work in progress, no test case coverage for now
class DataSet(object):  # pragma: no cover
    """
    Class for defining a dataset for a modeling/optimization/inference run

    Properties/variables:
    * x							(inputs)
    * y							(targets)
    * ts						(time series)
    * s 						(summary statistics)
    * outlier_column_indices	(columns containing outliers)
    * size
    * configurations 			(OrderedDict with relevant information)


    Methods:
    * get_size					(returns current size of the dataset)
    * add_points				(add data to the dataset, data can be added incrementally)
    * process_outliers			(check summary stats that contain outliers, and apply log scaling)
    * apply_func_to_columns     (Applies a transformation function to selected column indices of a matrix)


    """

    def __init__(self, name):
        """
        Initialize a dataset with parameters specified above
        
        Parameters
        ----------
        see above
        """
        self.name = name
        self.x = None
        self.y = None
        self.ts = None
        self.s = None
        self.outlier_column_indices = None
        self.outlier_detection = False
        self.configurations = OrderedDict()
        self.size = 0

    def get_size(self):
        """
        Returns the current number of points in the dataset
        
        Returns
        -------
        int
            The current number of points in the dataset
        """
        return self.size

    def add_points(self, inputs=None, targets=None, time_series=None, summary_stats=None):
        """
        Updates the dataset to include new points
        
        Parameters
        ----------
        inputs : ndarray, optional
            Usually parameter points, by default None
        targets : ndarray, optional
            The target for inferene/optimazation/exploration, by default None
        time_series : ndarray, optional
            Simulation output trajectories, by default None
        summary_stats : ndarray, optional
            The summary statistics, by default None
        
        Raises
        ------
        ValueError
            If all function args are None
        """
        if all(v is None for v in [inputs, targets, time_series, summary_stats]):
            raise ValueError('Dataset:add_points: no arguments specified.')

        if inputs is not None:
            if self.x is not None:
                np.testing.assert_equal(self.x.shape[1], inputs.shape[1], "Please validate the values and ensure the \
                                                                          shape equality of new samples to be added.")
                self.x = np.concatenate((self.x, inputs), axis=0)
            else:
                self.x = inputs

        if targets is not None:
            if self.y is not None:
                np.testing.assert_equal(self.y.shape[1], targets.shape[1], "Please validate the values and ensure the \
                                                                           shape equality of new samples to be added.")
                self.y = np.concatenate((self.y, targets), axis=0)
            else:
                self.y = targets

        if time_series is not None:
            if self.ts is not None:
                self.ts = np.concatenate((self.ts, time_series), axis=0)
            else:
                self.ts = time_series

        if summary_stats is not None:
            if self.s is not None:
                np.testing.assert_equal(self.s.shape[1], summary_stats.shape[1], "Please validate the values and \
                                                                                 ensure the shape equality of new \
                                                                                 samples to be added.")
                self.s = np.concatenate((self.s, summary_stats), axis=0)
            else:
                self.s = summary_stats

            if self.outlier_detection and len(self.s) > 1:
                self.process_outliers()

    def process_outliers(self, mode='zscore'):
        """
        Check for outliers in calculated summary stats. Outliers are the few very high or very low values that can
        potentially introduce bias in tasks such as parameter inference. One can either remove them, replace with mean
        value, or use log scale for the statistic in question. This choice is left to the user.
        
        Parameters
        ----------
        mode : str, optional
            Either use 'z-score' or inter-quantile range 'iqr', by default 'zscore'
        
        Returns
        -------
        array
            Indices of dataset.s columns containing outliers
        """
        if mode == 'zscore':
            # This will give us per-feature/per-statistic z-scores
            zscores = zscore(self.s, axis=0)

            # Find columns where abs(zscore) > threshold
            zscore_threshold = 3
            violation_indices = np.argwhere(np.abs(zscores) > zscore_threshold)
            if len(violation_indices) < 1:
                return
            outlier_indices = np.unique(np.argwhere(np.abs(zscores) > zscore_threshold)[:, 1])
        else:
            # Outlier detection using IQR
            quants = mquantiles(self.s)
            iqr = quants[2] - quants[0]
            iqr_factor = 1.5
            violations_left = self.s < quants[0] - iqr_factor * iqr
            violations_right = self.s > quants[2] + iqr_factor * iqr
            violation_indices = np.argwhere(violations_left | violations_right)
            if len(violation_indices) < 1:
                return
            outlier_indices = np.unique(np.argwhere(violations_left | violations_right)[:, 1])

        if len(outlier_indices) > 0:
            self.outlier_column_indices = outlier_indices
            print('Dataset:process_outliers: found outliers at indice(s) {}'.format(outlier_indices))
            print('Outliers can be transformed using the function Dataset.apply_func_to_outlier_columns if so desired.')

        return self.outlier_column_indices

    @staticmethod
    def apply_func_to_columns(func, matrix, idx):
        """
        Applies a transformation function to selected column indices of a matrix
        
        Parameters
        ----------
        func : callable
            the transformation function
        matrix : ndarray
            matrix to be processed
        idx : ndarray
            indices of the matrix to be transformed
        
        Returns
        -------
        ndarray
            the transformed matrix
        
        Raises
        ------
        ValueError
            [description]
        """
        if any(v is None for v in [func, matrix, idx]):
            raise ValueError('Dataset:apply_func_to_columns: all three function arguments are required.')
        return func(matrix[:, idx])
