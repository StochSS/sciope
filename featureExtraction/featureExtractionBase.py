from pandas import  DataFrame
import tsfresh
import numpy as np

class FeatureExtractionBase():

	def __init__(self, gillespy_model):
		self.model = gillespy_model
		self.columns = np.concatenate((['index','time'], self.model.listOfSpecies.keys()))
		self.dataframe = DataFrame(columns = self.columns)

	def put(self, data):
		for enum, datapoint in enumerate(data):
			df = DataFrame(data=map(lambda x: np.concatenate(([enum], x)), datapoint), columns = self.columns)
			print df
			self.dataframe = self.dataframe.append(df)
