from pandas import  DataFrame
import tsfresh
import numpy as np

class FeatureExtractionBase():

	def __init__(self, gillespy_model):
		self.model = gillespy_model
		self.columns = np.concatenate((['index','time'], self.model.listOfSpecies.keys()))
		self.dataframe = DataFrame(columns = self.columns)
		self.features = DataFrame()

	def put(self, data):
		nr_datapoints = len(self.features)
		for enum, datapoint in enumerate(data):
			enum += nr_datapoints
			df = DataFrame(data=map(lambda x: np.concatenate(([enum], x)), datapoint), columns = self.columns)
			self.dataframe = self.dataframe.append(df)

	def generate(self):
		self.features = tsfresh.extract_features(self.dataframe, column_id="index", column_sort="time", column_kind=None, column_value=None)

	def normalize(self):
		"""TODO"""
