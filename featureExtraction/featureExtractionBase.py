from pandas import  DataFrame
import tsfresh
import numpy as np

class FeatureExtractionBase():

	def __init__(self, gillespy_model):
		self.model = gillespy_model
		self.columns = np.concatenate((['index', 'computed', 'time'], self.model.listOfSpecies.keys()))
		self.dataframe = DataFrame(columns = self.columns)
		self.features = DataFrame()

	def put(self, data):
		"""TODO"""
		nr_datapoints = len(self.features)
		for enum, datapoint in enumerate(data):
			enum += nr_datapoints
			df = DataFrame(data=map(lambda x: np.concatenate(([enum, 0], x)), datapoint), columns = self.columns)
			self.dataframe = self.dataframe.append(df)
			

	def delete_row(self):
		"""TODO"""

	def delete_column(self):
		"""TODO"""

	def generate(self):
		"""TODO"""

		non_computed = self.dataframe.loc[self.dataframe['computed'] == 0] #filter non-computed rows
		if len(non_computed) == 0:
			print 'All datapoints in __.dataframe has already been computed. Add new datapoints first.'
		else:
			non_computed.drop(['computed'], axis=1) #remove the 'computed' column  (required by tsfresh)

			#compute the features using tsfresh
			f_subset = tsfresh.extract_features(non_computed, column_id = "index", column_sort = "time", column_kind = None, column_value = None)
			self.features = self.features.append(f_subset)
			self.dataframe.loc[self.dataframe['computed'] == 0, 'computed'] = 1 #warning code redundancy 

	def normalize(self):
		"""TODO"""
