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
Bayesian Neural Network (BNN) classifier
Ref:

"""

from sciope.models.model_base import ModelBase
from sciope.utilities.housekeeping import sciope_logger as ml
import tensorflow.compat.v2 as tf
#import tensorflow as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np


class BNNModel(ModelBase):

    def __init__(self, input_shape, output_shape, num_train_examples, conv_channel=6, 
                 kernel_size=5,
                 pooling_len=10,
                 problem_name='noname', use_logger=False):

        self.name = 'BNNModel'
        #super(BNNModel, self).__init__(self.name, use_logger) #TODO: use at production ready
        self.use_logger = use_logger                           #TODO: use super at production ready
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Bayesian Neural Network classifier model initialized")
        self.model = self._construct_model(input_shape, output_shape, conv_channel, kernel_size,
                                           num_train_examples, pooling_len)

    def _construct_model(self, input_shape, output_shape, conv_channel, kernel_size, NUM_TRAIN_EXAMPLES, pooling_len):
            """BCNN used in manuscript"""
            
            poolpadding = 'valid'
            pool = tf.keras.layers.MaxPooling1D
            
            kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /
                                    tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
            
            model_in = tf.keras.layers.Input(shape=input_shape)
            conv_1 = tfp.layers.Convolution1DFlipout(conv_channel, kernel_size=kernel_size, padding="same", strides=1,
                                                    kernel_divergence_fn=kl_divergence_function,
                                                    activation=tf.nn.relu)
            x = conv_1(model_in)
            
            x = pool(pooling_len, padding=poolpadding)(x)
            x = tf.keras.layers.Flatten()(x)
            
                
            dense = tfp.layers.DenseFlipout(output_shape, kernel_divergence_fn=kl_divergence_function,
                                            activation=tf.nn.softmax)
            
            model_out = dense(x)
            model = tf.keras.Model(model_in, model_out)
            
            return model

    def train(self, inputs, targets, val_inputs, val_targets, 
              batch_size=256, 
              learning_rate=0.001, 
              patience=5, 
              min_delta=0.001, 
              verbose=False):
        tf.keras.backend.clear_session()
        tf.keras.backend.set_floatx('float32')
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=1, min_delta=0.001,
                                              patience=5)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss = 'categorical_crossentropy'
        self.model.compile(optimizer, loss=loss,
                         metrics=['accuracy'], experimental_run_tf_function=False)
        
        self.model.fit(inputs, targets, batch_size=batch_size, epochs=1000, verbose=verbose,
                       validation_freq=1, validation_data=(val_inputs, val_targets),
                       callbacks=[es])
    
    def predict(self, inputs, verbose=False):
        return self.model.predict(inputs, verbose=verbose)
    
    def mc_sampling(self, inputs, num_monte_carlo=500):
        if self.use_logger:
            self.logger.info('Running monte carlo inference of BNN posterior')
        probs = tf.stack([self.predict(inputs, verbose=False)
                        for _ in range(num_monte_carlo)], axis=0)
        return probs
