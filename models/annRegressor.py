"""
Artificial Neural Network (ANN) Regression Surrogate Model
"""

# Imports
from modelBase import ModelBase
from sklearn.neural_network import MLPRegressor


# Class definition
class ANNModel(ModelBase):
	"""
	We use the sklearn MLP Regressor implementation here.
	"""
	def __init__(self):
		self.name = 'ANNModel'
		
	
	# train the ANN model given the data
	def train(self, inputs, targets):
		# Scale the training data
		self.scaleTrainingData(inputs, targets)
		
		# Train the model
		self.model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 2), max_iter=50000, learning_rate = 'adaptive', activation = 'logistic')
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
		