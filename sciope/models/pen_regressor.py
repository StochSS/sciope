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
Partially Exchangeable Networks (PEN) Regression Surrogate Model

Ref:
Wiqvist, Samuel, Pierre-Alexandre Mattei, Umberto Picchini, and Jes Frellsen.
"Partially Exchangeable Networks and Architectures for Learning Summary Statistics in Approximate Bayesian Computation."
 In International Conference on Machine Learning, pp. 6798-6807. 2019.
"""
from sciope.models.dnn_base import DNNBase
from tensorflow import keras
from sciope.utilities.housekeeping import sciope_logger as ml


# Class definition
class PENModel(DNNBase):
    """
    The PEN neural network for learning summary statistics. The PEN model learns the mapping
    time_series_trajs -> thetas.
    The train and predict methods are defined in the base class. Here we initialize the model and define the model
    construction routine.
    """

    def __init__(self, input_shape, output_shape, pen_nr=3, con_layers=[25, 50],
                 dense_layers=[100, 100, 100], use_logger=False, problem_name="None"):
        self.name = 'PENModel' + str(pen_nr) + '_conlayers_' + str(con_layers)
        super(PENModel, self).__init__(self.name, problem_name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Partially Exchangeable Network regression model initialized")

        self.model = self._construct_model(input_shape, output_shape, pen_nr, con_layers, dense_layers=dense_layers)
        self.pen_nr = pen_nr

    @staticmethod
    def _construct_model(input_shape, output_shape, pen_nr=3, con_layers=[25, 50, 100],
                         dense_layers=[100, 100, 100]):
        # TODO: add a **kwargs to specify the hyperparameters

        activation = 'relu'
        dense_activation = 'relu'
        padding = 'same'
        poolpadding = 'valid'
        con_len = 1

        maxpool = con_len

        batch_mom = 0.99
        reg = None
        pool = keras.layers.MaxPooling1D
        model = keras.Sequential()
        depth = input_shape[0]

        Input = keras.Input(shape=input_shape)
        # Add levels nr of CNN layers
        layer = keras.layers.Conv1D(con_layers[0], pen_nr + 1, strides=1,
                                    padding='valid', activity_regularizer=reg,
                                    input_shape=input_shape)(Input)
        layer = keras.layers.Activation(activation)(layer)

        for i in range(1, len(con_layers)):
            layer = keras.layers.Conv1D(con_layers[i], con_len, strides=1,
                                        padding=padding,
                                        activity_regularizer=reg)(layer)
            layer = keras.layers.Activation(activation)(layer)

        poolsize = input_shape[0] - (pen_nr)
        # Using Avgpooling to downsample the temporal dimension size to 1.
        layer = keras.layers.AvgPool1D(poolsize, padding=poolpadding)(layer)

        # Reshape previous layer to 1 dimension (feature state).
        layer = keras.layers.Flatten()(layer)
        cut_Input = keras.layers.Lambda(lambda x: x[:, 0:pen_nr + 1, :], )(Input)
        cut_Input_1d = keras.layers.Lambda(lambda x: keras.backend.reshape(x, (-1, (pen_nr + 1) * input_shape[1],)))(
            cut_Input)
        layer = keras.layers.concatenate([layer, cut_Input_1d])

        # Add 3 layers of Dense layers with activation function and Batch Norm.
        for i in range(len(dense_layers)):
            layer = keras.layers.Dense(dense_layers[i])(layer)
            layer = keras.layers.BatchNormalization(momentum=batch_mom)(layer)
            layer = keras.layers.Activation(dense_activation)(layer)

        # Add output layer without Activation or Batch Normalization
        layer = keras.layers.Dense(output_shape)(layer)
        model = keras.models.Model(inputs=Input, outputs=layer)

        # model.summary()
        return model
