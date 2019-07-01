# Copyright 2017 Prashant Singh, Fredrik Wrede and Andreas Hellander
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
from sciope.models.model_base import ModelBase
from sklearn.neural_network import MLPRegressor
from sciope.utilities.housekeeping import sciope_logger as ml


# Class definition
class ANNModel(ModelBase):
    """
    We use the sklearn MLP Regressor implementation here.
    """

    def __init__(self, use_logger=False):
        """
        Initialize the model.

        Parameters
        ----------
        use_logger : bool, optional
            Controls whether logging is enabled or disabled, by default False
        """
        self.name = 'ANNModel'
        super(ANNModel, self).__init__(self.name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Artificial Neural Network regression model initialized")

    def train(self, inputs, targets):
        """
        Train the ANN model given the data

        Parameters
        ----------
        inputs : nd-array
            independent variables
        targets : vector
            dependent variable
        """
        # Scale the training data
        self.scale_training_data(inputs, targets)

        # Train the model
        self.model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 2), max_iter=50000,
                                  learning_rate='adaptive', activation='logistic')
        self.model.fit(self.x, self.y)
        if self.use_logger:
            self.logger.info("Artificial Neural Network regression model trained with {} samples".format(len(self.y)))

    def predict(self, xt):
        """
        Predict unseen data using the trained model

        Parameters
        ----------
        xt : nd-array
            unseen data to be predicted

        Returns
        -------
        vector
            predictions
        """
        # predict
        yp = self.model.predict(xt)

        # scale back
        nt = xt.shape[0]
        yp = yp.reshape(nt, 1)
        yp = yp * self.sy + self.my
        return yp
