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
Support Vector Machine Regression Surrogate Model
"""

# Imports
from sciope.models.model_base import ModelBase
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sciope.utilities.housekeeping import sciope_logger as ml


# Class definition
class SVRModel(ModelBase):
    """
    We use the sklearn SVM implementation here.
    """

    def __init__(self, use_logger=False):
        """
        Initialize the model.

        Parameters
        ----------
        name : string
            Model name; set by the derived class
        use_logger : bool, optional
            Controls whether logging is enabled or disabled, by default False
        """
        self.name = 'SVRModel'
        super(SVRModel, self).__init__(self.name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Support Vector Regression model initialized")

    def tune_parameters(self, x, y, nfolds):
        """
        Tune hyper-parameters of the model

        x : inputs or independent variables
        y : output or dependent variable
        nfolds : number of cross-validation folds

        Returns
        -------
        vector
            pseudo-optimal parameters
        """
        cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': cs, 'gamma': gammas}
        grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(x, y)
        grid_search.best_params_
        return grid_search.best_params_

    def train(self, inputs, targets):
        """
        Train the SVR model given the data

        Parameters
        ----------
        inputs : nd-array
            independent variables
        targets : vector
            dependent variable
        """
        # Scale the training data
        self.scale_training_data(inputs, targets)

        # Tune parameters using 5-fold CV and grid-search
        params = self.tune_parameters(self.x, self.y, 5)

        # Train the model
        self.model = SVR(C=params['C'], gamma=params['gamma'])
        self.model.fit(self.x, self.y)
        if self.use_logger:
            self.logger.info("Support Vector Regression model trained with {} samples".format(len(self.y)))

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
