"""
Gaussian Process Regression Surrogate Model
"""

# Imports
from modelBase import ModelBase
from sklearn.gaussian_process import GaussianProcessRegressor


# Class definition
class GPRModel(ModelBase):
	"""
	We use the sklearn GP Regressor implementation here.
	"""
	def __init__(self):
		self.name = 'GPRModel'
		
	
	# train the GP model given the data
	def train(self, inputs, targets):
		# Scale the training data
		self.scaleTrainingData(inputs, targets)
		
		# Train the model
		self.model = GaussianProcessRegressor(n_restarts_optimizer=100)
		self.model.fit(self.X, self.y)
		
	# Predict
	# * NOTE * 
	# GP returns the mean and variance of prediction, so handle it accordingly while using predict
	def predict(self, Xt):
		# Predict
		yp, sigma = self.model.predict(Xt, return_std=True)
		
		# Scale back
		Nt = Xt.shape[0]
		yp = yp.reshape(Nt,1)
		yp = yp * self.sy + self.my
		return (yp, sigma)
		