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
Convolutional Neural Network (CNN) Regression Surrogate Model
"""
from sciope.models.dnn_base import DNNBase
from tensorflow import keras
from sciope.utilities.housekeeping import sciope_logger as ml


# Class definition
class CNNModel(DNNBase):
    """
    The CNN architecture for learning summary statistics. The CNN learns the mapping time_series_trajs -> thetas.
    The train and predict methods are defined in the base class. Here we initialize the model and define the model
    construction routine.
    """

    def __init__(self, input_shape, output_shape, con_len=3, con_layers=[25, 50],
                 last_pooling=keras.layers.AvgPool1D, dense_layers=[100, 100], pooling_len=3,
                 problem_name='noname', use_logger=False):
        self.name = 'CNNModel_con_len' + str(con_len) + '_con_layers' + str(con_layers) + '_pl' + str(
            pooling_len) + '_dense_layers' + str(dense_layers) + '_data' + problem_name
        super(CNNModel, self).__init__(self.name, problem_name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Convolutional Neural Network regression model initialized")
        self.model = self._construct_model(input_shape, output_shape, con_len=con_len, con_layers=con_layers,
                                           pooling_len=pooling_len, last_pooling=last_pooling,
                                           dense_layers=dense_layers)

    @staticmethod
    def _construct_model(input_shape, output_shape, con_len=3, con_layers=[25, 50, 100], pooling_len=3,
                         last_pooling=keras.layers.AvgPool1D, dense_layers=[100, 100]):
        # TODO: add a **kwargs to specify the hyperparameters
        activation = 'relu'
        dense_activation = 'relu'
        padding = 'same'
        poolpadding = 'valid'

        maxpool = con_len
        levels = 3
        batch_mom = 0.99
        reg = None
        # pool = keras.layers.AvgPool1D #
        pool = keras.layers.MaxPooling1D
        model = keras.Sequential()
        depth = input_shape[0]

        # Add levels nr of CNN layers
        model.add(keras.layers.Conv1D(con_layers[0], con_len, strides=1,
                                      padding=padding, activity_regularizer=reg,
                                      input_shape=input_shape))
        model.add(keras.layers.Activation(activation))
        model.add(keras.layers.Conv1D(con_layers[0], con_len, strides=1,
                                      padding=padding, activity_regularizer=reg))
        model.add(keras.layers.Activation(activation))

        model.add(pool(pooling_len, padding=poolpadding))
        if padding == 'valid':
            depth -= (pooling_len - 1) * 3
        depth = depth // pooling_len

        for i in range(1, len(con_layers)):
            model.add(keras.layers.Conv1D(con_layers[i], con_len, strides=1,
                                          padding=padding,
                                          activity_regularizer=reg))
            model.add(keras.layers.Activation(activation))
            model.add(keras.layers.Conv1D(con_layers[i], con_len, strides=1,
                                          padding=padding,
                                          activity_regularizer=reg))
            model.add(keras.layers.Activation(activation))

            if padding == 'valid':
                depth -= (pooling_len - 1) * 2
            if i < len(con_layers) - 1:
                model.add(pool(pooling_len, padding=poolpadding))
                depth = depth // pooling_len

        # Using Maxpooling to downsample the temporal dimension size to 1.
        # model.add(keras.layers.MaxPooling1D(depth,padding=poolpadding))
        model.add(last_pooling(depth, padding=poolpadding))
        # Reshape previous layer to 1 dimension (feature state).
        model.add(keras.layers.Flatten())

        # Add 3 layers of Dense layers with activation function and Batch Norm.
        for i in range(len(dense_layers)):
            model.add(keras.layers.Dense(dense_layers[i]))
            model.add(keras.layers.BatchNormalization(momentum=batch_mom))
            model.add(keras.layers.Activation(dense_activation))

        # Add output layer without Activation or Batch Normalization
        model.add(keras.layers.Dense(output_shape))

        # model.summary()
        return model
