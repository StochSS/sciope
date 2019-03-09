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
class DataSet(object):
	"""
	Class for defining a dataset for a modeling/optimization/inference run 
	
	Properties/variables:
	* x							(inputs)
	* y							(targets)
	* ts						(time series)
	* s 						(summary statistics)
	* outlier_column_indices	(columns containing outliers)
	* size
	* configurations 			(OrderedDict with relavant information)
	* ensembles? 

	
	Methods:
	* impute 					(treat missing values in summary statistics data)
	* scaler? 
	* set_data					(set inputs and targets)
	* get_size					(returns current size of the dataset)
	* add_points				(updates the dataset to include new points)
	* process_outliers			(check summary stats that contain outliers, and apply log scaling)

	
	"""
	
	def __init__(self, name):
		self.name = name
		self.x = None
		self.y = None
		self.ts = None
		self.s = None
		self.outlier_column_indices = None
		self.configurations = OrderedDict()
		self.size = 0
		
	def set_data(self, inputs, targets, time_series=None, summary_stats=None):
		"""
		Sets the inputs and target variables
		"""
		self.x = inputs
		self.y = targets
		self.ts = time_series
		self.s = summary_stats
		self.size = self.x.shape[0]
		self.process_outliers()
	
	def get_size(self):
		"""
		Returns the current number of points in the dataset
		"""
		return self.size
		
	def add_points(self, inputs, targets, time_series=None, summary_stats=None):
		"""
		Updates the dataset to include new points
		"""

		self.x = np.concatenate((self.x, inputs))
		self.y = np.concatenate((self.y, targets))
		
		if time_series.any() is not None:
			self.ts = np.concatenate((self.ts, time_series))

		if summary_stats.any() is not None:
			if self.outlier_column_indices is not None:
				summary_stats[:, self.outlier_column_indices] = np.log(summary_stats[:, self.outlier_column_indices])
			self.s = np.concatenate((self.s, summary_stats))
		
		self.size = self.x.shape[0]

	def process_outliers(self, mode='iqr'):
		"""
		Check for outliers in calculated summary stats. Outliers are the few very high or very low values that can
		potentially introduce bias in tasks such as parameter inference. One can either remove them, replace with mean
		value, or use log scale for the statistic in question.
		@ToDo: add removal and imputations as options in addition to iqr and z-score
		:param mode: either use 'z-score' or inter-quantile range 'iqr'
		:return: -
		"""
		if mode == 'zscore':
			# This will give us per-feature/per-statistic z-scores
			zscores = zscore(self.s, axis=0)

			# Find columns where abs(zscore) > threshold
			zscore_threshold = 3
			self.outlier_column_indices = np.unique(np.argwhere(np.abs(zscores) > zscore_threshold)[:, 1])
		else:
			# Outlier detection using IQR
			quants = mquantiles(self.s)
			iqr = quants[2] - quants[0]
			violations_left = x2 < quants[0] - 1.5 * iqr
			violations_right = x2 > quants[2] + 1.5 * iqr
			self.outlier_column_indices = np.unique(np.argwhere(violations_left | violations_right)[:, 1])

		self.s[:, self.outlier_column_indices] = np.log(self.s[:, self.outlier_column_indices])
