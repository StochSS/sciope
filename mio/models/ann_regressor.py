# Copyright 2017 Prashant Singh, Andreas Hellander and Fredrik Wrede
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
Artificial Neural Network (ANN) Regression Surrogate Model
"""

# Imports
from model_base import ModelBase
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
        self.model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 2), max_iter=50000,
                                  learning_rate='adaptive', activation='logistic')
        self.model.fit(self.x, self.y)

    # Predict
    def predict(self, xt):
        # predict
        yp = self.model.predict(xt)

        # scale back
        nt = xt.shape[0]
        yp = yp.reshape(nt, 1)
        yp = yp * self.sy + self.my
        return yp
