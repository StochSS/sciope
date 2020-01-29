# Copyright 2019 Mattias Åkesson, Prashant Singh, Fredrik Wrede and Andreas Hellander
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
Deep Neural Network (DNN) Regression Surrogate Model
Ref:
Jiang, Bai, Tung-yu Wu, Charles Zheng, and Wing H. Wong.
"Learning summary statistic for approximate Bayesian computation via deep neural network."
Statistica Sinica (2017): 1595-1618.
"""
from sciope.models.dnn_base import DNNBase
from tensorflow import keras
from sciope.utilities.housekeeping import sciope_logger as ml
import numpy as np


# Class definition
class DNNModel(DNNBase):
    """
    The deep neural network for learning summary statistics. The DNN learns the mapping time_series_trajs -> thetas.
    The train and predict methods are defined in the base class. Here we initialize the model and define the model
    construction routine.
    """

    def __init__(self, input_shape, output_shape, layers=[100, 100, 100], use_logger=False, problem_name="None"):
        self.name = 'DNNModel_l' + str(layers)
        super(DNNModel, self).__init__(self.name, problem_name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Deep Neural Network regression model initialized")
        self.model = self._construct_model(input_shape, output_shape, layers=layers)

    @staticmethod
    def _construct_model(input_shape, output_shape, layers=[100, 100, 100]):
        # TODO: add a **kwargs to specify the hyperparameters
        dense_activation = 'relu'
        batch_mom = 0.99
        model = keras.Sequential()
        new_shape = np.prod(input_shape)
        model.add(keras.layers.Reshape((new_shape,), input_shape=input_shape))

        # Add 3 layers of Dense layers with activation function and Batch Norm.
        for i in range(0, len(layers)):
            if i == 0:
                model.add(keras.layers.Dense(layers[i], input_dim=100))
            else:
                model.add(keras.layers.Dense(layers[i]))

            model.add(keras.layers.BatchNormalization(momentum=batch_mom))
            model.add(keras.layers.Activation(dense_activation))

        # Add output layer without Activation or Batch Normalization
        model.add(keras.layers.Dense(output_shape))

        # model.summary()
        return model
