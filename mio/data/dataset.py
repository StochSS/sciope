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

	
	Methods:
	* impute 					(treat missing values in summary statistics data) 
	* get_size					(returns current size of the dataset)
	* add_points				(add data to the dataset, data can be added incrementally)
	* process_outliers			(check summary stats that contain outliers, and apply log scaling)

	
	"""
	
	def __init__(self, name):
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
		TODO:
		Returns the current number of points in the dataset
		"""
		return self.size
		
	def add_points(self, inputs=None, targets=None, time_series=None, summary_stats=None):
		"""
		Updates the dataset to include new points
		@ToDo: Put in validation and exception handling
		"""
		if inputs is not None:
			if self.x is not None:
				self.x = np.concatenate((self.x, inputs), axis=0)
			else:
				self.x = inputs
				
		if targets is not None:
			if self.y is not None:
				self.y = np.concatenate((self.y, targets), axis=0)
			else:
				self.y = targets
		
		if time_series is not None:
			if self.ts is not None:
				self.ts = np.concatenate((self.ts, time_series), axis=0)
			else:
				self.ts = time_series
		
		if summary_stats is not None:
			if self.outlier_detection and self.outlier_column_indices is not None:
				summary_stats[:, self.outlier_column_indices] = np.log(summary_stats[:, self.outlier_column_indices])
			
			if self.s is not None:
				self.s = np.concatenate((self.s, summary_stats), axis=0)
			else:
				self.s = summary_stats

			if self.outlier_detection and len(self.s) > 1:
				self.process_outliers()

	def process_outliers(self, mode='zscore'):
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

		# Check if the indices have previously been processed
		# We do not want to get into a cycle of logloglog...
		if self.outlier_column_indices is not None:
			indices_to_process = np.setdiff1d(np.union1d(self.outlier_column_indices, outlier_indices),
														np.intersect1d(self.outlier_column_indices, outlier_indices))
			self.outlier_column_indices = np.union1d(self.outlier_column_indices, indices_to_process)
		else:
			indices_to_process = outlier_indices
			self.outlier_column_indices = indices_to_process

		self.s[:, indices_to_process] = np.log(self.s[:, indices_to_process])

	@staticmethod
	def calculate_distance(fixed_ds, sim_ds, sim_stats):
		"""
		Compares the fixed and simulated datasets for outlier indices, and returns the union of outlier indices.
		:param fixed_ds: the fixed dataset
		:param sim_ds: the simulated dataset
		:param sim_stats: the simulated stats, subset of sim_ds
		:return: the indices of the fixed dataset that will need to be log-scaled based on outliers in both datasets
		"""
		if fixed_ds.outlier_column_indices is None and sim_ds.outlier_column_indices is None:
			# No outliers anywhere
			fixed_stats = fixed_ds.s
		elif fixed_ds.outlier_column_indices is None and sim_ds.outlier_column_indices is not None:
			# outliers only in simulated data
			sim_stats[:, sim_ds.outlier_column_indices] = np.log(sim_stats[:, sim_ds.outlier_column_indices])
			fixed_stats = fixed_ds.s
			fixed_stats[:, sim_ds.outlier_column_indices] = np.log(fixed_stats[:, sim_ds.outlier_column_indices])
		elif fixed_ds.outlier_column_indices is not None and sim_ds.outlier_column_indices is None:
			# outliers only in fixed dataset
			fixed_stats = fixed_ds.s
			sim_stats[:, fixed_ds.outlier_column_indices] = np.log(sim_stats[:, fixed_ds.outlier_column_indices])
		else:
			# outliers in both datasets
			outlier_indices = np.union1d(fixed_ds.outlier_column_indices, sim_ds.outlier_column_indices)
			sim_stats[:, outlier_indices] = np.log(sim_stats[:, outlier_indices])
			outliers_for_fixed_ds = np.setdiff1d(sim_ds.outlier_column_indices, fixed_ds.outlier_column_indices)
			fixed_stats = fixed_ds.s
			fixed_stats[:, outliers_for_fixed_ds] = np.log(fixed_stats[:, outliers_for_fixed_ds])

		return fixed_stats, sim_stats
