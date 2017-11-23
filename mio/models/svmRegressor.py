# Copyright 2017 Fredrik Wrede, Prashant Singh and Andreas Hellander
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
Support Vector Machine Regression Surrogate Model
"""

# Imports
from modelBase import ModelBase
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# Class definition
class SVRModel(ModelBase):
	"""
	We use the sklearn SVM implementation here.
	"""
	def __init__(self):
		self.name = 'SVRModel'
		
	# Tune parameters of the model
	def tuneParameters(self, X, y, nfolds):
		Cs = [0.001, 0.01, 0.1, 1, 10]
		gammas = [0.001, 0.01, 0.1, 1]
		param_grid = {'C': Cs, 'gamma' : gammas}
		grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=nfolds)
		grid_search.fit(X, y)
		grid_search.best_params_
		return grid_search.best_params_
	
	# train the SVR model given the data
	def train(self, inputs, targets):
		# Scale the training data
		self.scaleTrainingData(inputs, targets)
		
		# Tune parameters using 5-fold CV and grid-search
		params = self.tuneParameters(self.X, self.y, 5)
		
		# Train the model
		self.model = SVR(C=params['C'], gamma=params['gamma'])
		self.model.fit(self.X, self.y)
		
	# Predict
	def predict(self, Xt):
		# predict
		yp = self.model.predict(Xt)
		
		# scale back
		Nt = Xt.shape[0]
		yp = yp.reshape(Nt,1)
		yp = yp * self.sy + self.my
		return yp
		