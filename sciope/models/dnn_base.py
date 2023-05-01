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
from sciope.models.model_base import ModelBase
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from abc import abstractmethod


# Class definition
class DNNBase(ModelBase):
    """
    We use keras to define CNN and DNN layers to the model
    """

    def __init__(self, name, problem_name='noname', use_logger=False):
        super(DNNBase, self).__init__(name, use_logger)
        self.target_scaler = None
        self.scale_output = False
        self.scale_input = False
        self._r_min = None
        self._r_min = None
        self.problem_name = problem_name

    # train the CNN model given the data
    def train(self, inputs, targets, batch_size, epochs, learning_rate=0.001,
              val_freq=1, early_stopping_patience=5, validation_split=0.2,
              validation_inputs=None, validation_targets=None, verbose=0, scale_output=False, scale_input=False):
        # pre-process input
        num_species = inputs.shape[1]
        num_timestamps = inputs.shape[2]
        num_samples = inputs.shape[0]

        # scale the input in [0, 1] if desired
        if scale_input:
            self.scale_input = True
            self._r_min = np.tile(np.expand_dims(np.min(inputs, (0, 2)), 1), num_timestamps)
            self._r_max = np.tile(np.expand_dims(np.max(inputs, (0, 2)), 1), num_timestamps)
            inputs = (inputs - self._r_min) / (self._r_max - self._r_min)
            if validation_inputs is not None:
                validation_inputs = (validation_inputs - self._r_min) / (self._r_max - self._r_min)

        # reshape from NxSxT to NxTxS
        inputs = inputs.transpose((0, 2, 1))
        if validation_inputs is not None:
            validation_inputs = validation_inputs.transpose((0, 2, 1))

        # scale the thetas/targets
        if scale_output:
            self.scale_output = True
            self.target_scaler = MinMaxScaler()
            targets = self.target_scaler.fit_transform(targets)
            if validation_inputs is not None:
                validation_targets = self.target_scaler.transform(validation_targets)

        es = keras.callbacks.EarlyStopping(monitor='val_mae', mode='min', verbose=verbose,
                                           patience=early_stopping_patience)
        # Using Adam optimizer
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                           loss='mean_squared_error', metrics=['mae'])

        if validation_inputs is not None and validation_targets is not None:
            history = self.model.fit(inputs, targets, validation_data=(validation_inputs, validation_targets),
                                     epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[es],
                                     validation_freq=val_freq, verbose=verbose)
        else:
            history = self.model.fit(inputs, targets, validation_split=validation_split,
                                     epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[es],
                                     verbose=verbose)

        return history

    # Predict
    def predict(self, xt):
        # scale the input in [0, 1] if desired
        if self.scale_input:
            xt = (xt - self._r_min) / (self._r_max - self._r_min)

        # reshape from NxSxT to NxTxS
        xt = xt.transpose((0, 2, 1))

        yt = self.model.predict(xt)
        if self.scale_output:
            yt = self.target_scaler.inverse_transform(yt)
        return yt


@abstractmethod
def _construct_model():
    """
    Each derived class must put in the model construction routine here.
    :return: an object of the model
    """
